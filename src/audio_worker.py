import collections
import io
import json
import os
import platform
import queue
import subprocess
import time
import urllib.request
import wave
import zipfile

import numpy as np
import sounddevice as sd
from groq import Groq
from scipy.io import wavfile

from src.coordinator import StateCoordinator


class AudioWorker:
    def __init__(self, coordinator: StateCoordinator, alert_queue: queue.Queue):
        self.coordinator = coordinator
        self.alert_queue = alert_queue
        
        self.model_name = "en_US-lessac-medium.onnx"
        self.config_name = f"{self.model_name}.json"
        self._ensure_piper_model()
        
        print("[Audio AI] Booting Native C++ Daemon and locking ONNX matrix in RAM...")
        piper_cmd = [
            self.binary_path,
            "--model", self.model_name,
            "--config", self.config_name,
            "--json-input"
        ]
        
        self.piper_daemon = subprocess.Popen(
            piper_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,           
            encoding='utf-8'
        )
        print("[Audio AI] IPC Pipeline established. C++ Engine is listening.")

        self.groq_key = os.environ.get("GROQ_API_KEY")
        self.client = Groq(api_key=self.groq_key) if self.groq_key else None
        
        self.SAMPLE_RATE = 16000  
        self.CHUNK_SIZE = 1024    
        self.VAD_THRESHOLD = 0.02 
        self.SILENCE_TIMEOUT = 0.5      
        
        self.pre_roll_buffer = collections.deque(maxlen=4)
        self.emergency_handled = False
        self.med_success_handled = False
        
        self.chat_history = [{"role": "system", "content": (
            "You are a compassionate, ultra-concise mobile assistant robot inside a dementia patient's home. "
            "Keep your answers limited to a maximum of 2 sentences under all circumstances."
        )}]

    def _ensure_piper_model(self) -> None:
        base_voice_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/"
        if not os.path.exists(self.model_name):
            urllib.request.urlretrieve(base_voice_url + self.model_name, self.model_name)
        if not os.path.exists(self.config_name):
            urllib.request.urlretrieve(base_voice_url + self.config_name, self.config_name)

        self.os_type = platform.system()
        if self.os_type == "Windows":
            self.binary_path = os.path.join("piper_bin", "piper", "piper.exe")
            if not os.path.exists(self.binary_path):
                print("[System] Windows detected. Fetching standalone pre-compiled C++ Piper engine (~21MB)...")
                zip_path = "piper_windows.zip"
                binary_url = "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip"
                urllib.request.urlretrieve(binary_url, zip_path)
                os.makedirs("piper_bin", exist_ok=True)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall("piper_bin")
                os.remove(zip_path)
                print("[System] Standalone C++ binary engine unpacked successfully.")
        else:
            self.binary_path = "piper"

    def _speak(self, text: str, stream: sd.InputStream) -> sd.InputStream:
        self.coordinator.update_state(is_agent_speaking=True)
        
        output_file = "tts_response.wav"
        try:
            if stream is not None:
                stream.stop()
                stream.close()

            payload = json.dumps({"text": text, "output_file": output_file}) + "\n"
            self.piper_daemon.stdin.write(payload)
            self.piper_daemon.stdin.flush() 
            
            _ack = self.piper_daemon.stdout.readline()

            sample_rate, audio_data = wavfile.read(output_file)
            audio_data_float = audio_data.astype(np.float32) / 32768.0
            
            sd.play(audio_data_float, sample_rate)
            sd.wait() 
            
        except Exception as e:
            print(f"[Audio Error] IPC Neural TTS execution failed: {e}")
        finally:
            new_stream = sd.InputStream(samplerate=self.SAMPLE_RATE, blocksize=self.CHUNK_SIZE, channels=1, dtype='float32')
            new_stream.start()
            self.coordinator.update_state(is_agent_speaking=False)
            return new_stream

    def _execute_priority_intercepts(self, stream: sd.InputStream) -> sd.InputStream:
        snapshot = self.coordinator.get_snapshot()
        
        # Priority 1: Hardware Fall Safety
        if snapshot.is_falling and not self.emergency_handled:
            self.emergency_handled = True
            print("\n[Audio Intercept] Safety State Latched! Executing primary emergency verbal feedback.")
            return self._speak("Attention. I have detected a potential fall. Sending alert notifications to your family now. Are you alright?", stream)
            
        # Priority 2: Positive Reinforcement (Meds verified)
        if snapshot.medication_completed and not self.med_success_handled:
            self.med_success_handled = True
            print("\n[Audio Intercept] Medication sequence verified! Executing positive reinforcement.")
            return self._speak("Great job taking your medication. I have securely logged it in your daily health record.", stream)
            
        # Priority 3: Scheduled Intervention
        if snapshot.medication_reminder_active and not snapshot.medication_reminder_spoken:
            print("\n[Audio Intercept] Schedule Triggered! Audibly reminding patient.")
            self.coordinator.update_state(medication_reminder_spoken=True)
            return self._speak("Excuse me, it is time to take your scheduled medication. Please take it now so I can log it for you.", stream)

        return None

    def _compile_in_memory_wav(self, audio_data: np.ndarray) -> io.BytesIO:
        wav_buffer = io.BytesIO()
        int_data = (audio_data * 32767).astype(np.int16)
        wavfile.write(wav_buffer, self.SAMPLE_RATE, int_data)
        wav_buffer.seek(0)
        return wav_buffer

    def run(self) -> None:
        print("[Audio HW] Opening hardware microphone input ring buffers...")
        try:
            stream = sd.InputStream(samplerate=self.SAMPLE_RATE, blocksize=self.CHUNK_SIZE, channels=1, dtype='float32')
            stream.start()
        except Exception as err:
            print(f"[Audio HW] CRITICAL: Failed to link local audio device nodes: {err}")
            self.coordinator.update_state(is_running=False)
            return

        print("[Audio System] Live Mic Input Stream operational. VAD monitoring active.")
        voice_buffer = []
        silence_start_time = None
        is_recording_phrase = False

        while self.coordinator.get_snapshot().is_running:
            intercept_stream = self._execute_priority_intercepts(stream)
            if intercept_stream is not None:
                stream = intercept_stream
                voice_buffer.clear()
                self.pre_roll_buffer.clear()
                is_recording_phrase = False
                continue

            chunk, available_frames = stream.read(self.CHUNK_SIZE)
            chunk = chunk.flatten()

            if self.coordinator.get_snapshot().is_agent_speaking:
                voice_buffer.clear()
                self.pre_roll_buffer.clear()
                is_recording_phrase = False
                silence_start_time = None
                continue

            rms_energy = np.sqrt(np.mean(chunk**2))

            if rms_energy > self.VAD_THRESHOLD:
                if not is_recording_phrase:
                    voice_buffer.extend(self.pre_roll_buffer)
                    self.coordinator.update_state(is_user_speaking=True)
                    is_recording_phrase = True
                    
                voice_buffer.append(chunk)
                silence_start_time = None 
            else:
                if not is_recording_phrase:
                    self.pre_roll_buffer.append(chunk)
                else:
                    if silence_start_time == None:
                        silence_start_time = time.time()
                    
                    voice_buffer.append(chunk)
                    
                    if time.time() - silence_start_time >= self.SILENCE_TIMEOUT:
                        self.coordinator.update_state(is_user_speaking=False)
                        is_recording_phrase = False
                        
                        if len(voice_buffer) > 0 and self.client:
                            full_audio = np.concatenate(voice_buffer)
                            voice_buffer.clear()
                            virtual_file = self._compile_in_memory_wav(full_audio)
                            
                            try:
                                transcription = self.client.audio.transcriptions.create(
                                    model="whisper-large-v3",
                                    file=("speech.wav", virtual_file),
                                    language="en"
                                )
                                user_text = transcription.text.strip()
                                
                                if user_text:
                                    print(f"\n[User]: \"{user_text}\"")
                                    
                                    if self.coordinator.get_snapshot().is_falling:
                                        print("[Audio AI] Target confirmed safety. De-latching fallback sequences.")
                                        self.coordinator.update_state(is_falling=False)
                                        self.emergency_handled = False
                                        self.chat_history.append({"role": "system", "content": "CRITICAL CONTEXT: The user just experienced a physical fall. Based on their next message, assess if they need help or are okay."})
                                    
                                    self.chat_history.append({"role": "user", "content": user_text})
                                    
                                    print(f"[Robot]: ", end="", flush=True)
                                    response_stream = self.client.chat.completions.create(
                                        model="llama-3.1-8b-instant",
                                        messages=self.chat_history,
                                        max_tokens=60,
                                        temperature=0.6,
                                        stream=True 
                                    )
                                    
                                    response = ""
                                    for token_chunk in response_stream:
                                        if token_chunk.choices[0].delta.content is not None:
                                            token = token_chunk.choices[0].delta.content
                                            response += token
                                            print(token, end="", flush=True)
                                    print() 
                                    
                                    self.chat_history.append({"role": "assistant", "content": response})
                                    stream = self._speak(response, stream)
                                    
                            except Exception as pipeline_err:
                                print(f"[Audio Error] Exception occurred: {pipeline_err}")
                        else:
                            voice_buffer.clear()

        self.piper_daemon.stdin.close()
        self.piper_daemon.terminate()
        stream.stop()
        stream.close()