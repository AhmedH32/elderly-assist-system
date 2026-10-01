import math
import queue
import time

import cv2
import numpy as np
from ultralytics import YOLO

from src.action_recognizer import MedicationTracker
from src.coordinator import StateCoordinator


class VisionWorker:
    def __init__(self, coordinator: StateCoordinator, display_queue: queue.Queue, alert_queue: queue.Queue):
        self.coordinator = coordinator
        self.display_queue = display_queue
        self.alert_queue = alert_queue
        
        self.current_state = "IDLE"
        self.consecutive_frames = 0
        self.missed_frames = 0 
        self.cooldown_start_time = 0.0
        
        self.DEBOUNCE_THRESHOLD = 8  
        self.COOLDOWN_DURATION = 30.0 
        
        self.prev_center_y = None
        self.prev_timestamp = None

        print("[Vision AI] Loading localized YOLO26-Pose weights into volatile RAM...")
        self.model = YOLO("yolo26n-pose.pt")
        print("[Vision AI] Weights compiled successfully.")
        
        # Initialize the 4-stage biomechanical tracker with hand-to-mouth offset compensation
        self.med_tracker = MedicationTracker(proximity_ratio=0.25)

    def run(self) -> None:
        print("[Vision HW] Initializing camera hardware descriptor...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            self.coordinator.update_state(is_running=False)
            return

        print("[Vision] System online. Ingesting physical streams...")
        
        while self.coordinator.get_snapshot().is_running:
            ret, frame = cap.read()
            if not ret:
                self.coordinator.update_state(is_running=False)
                break

            results = self.model(frame, verbose=False)
            is_anomaly_detected = False
            current_timestamp = time.time()
            
            # Atomic fetch of current system state flags to coordinate vision execution gates
            snap = self.coordinator.get_snapshot()
            
            if len(results) > 0 and results[0].keypoints is not None:
                keypoints_tensor = results[0].keypoints.data.cpu().numpy()
                
                if keypoints_tensor.shape[0] > 0:
                    for person_idx in range(keypoints_tensor.shape[0]):
                        kpts = keypoints_tensor[person_idx]
                        
                        # Safely calculate bounding box scaling factors
                        bboxes = results[0].boxes.xywh.cpu().numpy() if results[0].boxes else None
                        bbox_height = bboxes[person_idx][3] if (bboxes is not None and len(bboxes) > person_idx) else 400.0
                        
                        # --- GATED MEDICATION TRACKING EVALUATION ---
                        if snap.medication_reminder_active and not snap.medication_completed:
                            
                            # Execute O(1) state transitions using raw pose vectors
                            if self.med_tracker.process_frame(kpts, bbox_height):
                                print("\n[AI Logic] Biomechanical sequence verified: Medication ingested.")
                                try:
                                    self.alert_queue.put_nowait("medication_taken")
                                    self.alert_queue.put_nowait("activity_Monitoring Patient Safely")
                                    self.coordinator.update_state(medication_completed=True)
                                except queue.Full:
                                    pass

                            # --- DYNAMIC MEDICATION GUI TELEMETRY (ACTIVE MODE) ---
                            dynamic_threshold = int(bbox_height * self.med_tracker.proximity_ratio)
                            nose_x, nose_y = int(kpts[0][0]), int(kpts[0][1])
                            cv2.circle(frame, (nose_x, nose_y), dynamic_threshold, (255, 255, 0), 2)
                            
                            if kpts[9][2] > 0.4: 
                                cv2.circle(frame, (int(kpts[9][0]), int(kpts[9][1])), 6, (0, 255, 255), -1)
                            if kpts[10][2] > 0.4: 
                                cv2.circle(frame, (int(kpts[10][0]), int(kpts[10][1])), 6, (0, 255, 255), -1)
                            
                            telemetry = self.med_tracker.get_telemetry()
                            state_labels = ["0: IDLE", "1: PILL (Checking...)", "2: GAP (Waiting for Water...)", "3: WATER (Drinking...)"]
                            
                            cv2.putText(frame, f"MED TRACKER: {state_labels[telemetry['state']]}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                            if telemetry['state'] > 0:
                                cv2.putText(frame, f"Timer: {telemetry['elapsed']:.1f}s", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        
                        else:
                            # --- DYNAMIC MEDICATION GUI TELEMETRY (STANDBY / COMPLETE MODES) ---
                            if snap.medication_completed:
                                cv2.putText(frame, "MED TRACKER: LOGGED (Completed)", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            else:
                                cv2.putText(frame, "MED TRACKER: STANDBY (Awaiting Window)", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                        # -------------------------------------------------------------------------
                        
                        # --- FALL DETECTION PROCESSING PIPELINE ---
                        valid_shoulders = []
                        if kpts[5, 2] > 0.5: valid_shoulders.append(kpts[5, :2])
                        if kpts[6, 2] > 0.5: valid_shoulders.append(kpts[6, :2])
                        
                        valid_hips = []
                        if kpts[11, 2] > 0.5: valid_hips.append(kpts[11, :2])
                        if kpts[12, 2] > 0.5: valid_hips.append(kpts[12, :2])
                        
                        if len(valid_shoulders) > 0 and len(valid_hips) > 0:
                            shoulder_center = np.mean(valid_shoulders, axis=0)
                            hip_center = np.mean(valid_hips, axis=0)
                            
                            dx = hip_center[0] - shoulder_center[0]
                            dy = hip_center[1] - shoulder_center[1]
                            
                            angle_rad = math.atan2(abs(dx), abs(dy))
                            angle_deg = math.degrees(angle_rad)
                            
                            current_center_y = (shoulder_center[1] + hip_center[1]) / 2.0
                            velocity_y = 0.0
                            
                            if self.prev_center_y is not None and self.prev_timestamp is not None:
                                delta_t = current_timestamp - self.prev_timestamp
                                if delta_t > 0:
                                    velocity_y = (current_center_y - self.prev_center_y) / delta_t
                            
                            self.prev_center_y = current_center_y
                            self.prev_timestamp = current_timestamp
                            
                            cv2.line(frame, (int(shoulder_center[0]), int(shoulder_center[1])), 
                                     (int(hip_center[0]), int(hip_center[1])), (255, 255, 0), 3)
                            cv2.putText(frame, f"Angle: {int(angle_deg)}deg", (int(shoulder_center[0]) + 10, int(shoulder_center[1])),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                            if angle_deg > 60.0 and velocity_y > 40.0:
                                is_anomaly_detected = True
                                break 

            # Fall Detector State Machine Logic
            if self.current_state == "IDLE":
                if is_anomaly_detected:
                    self.current_state = "TRACKING"
                    self.consecutive_frames = 1
                    self.missed_frames = 0

            elif self.current_state == "TRACKING":
                if is_anomaly_detected:
                    self.consecutive_frames += 1
                    self.missed_frames = 0
                    if self.consecutive_frames >= self.DEBOUNCE_THRESHOLD:
                        self.current_state = "LATCHED"
                else:
                    self.missed_frames += 1
                    if self.missed_frames > 4:
                        self.current_state = "IDLE"
                        self.consecutive_frames = 0

            elif self.current_state == "LATCHED":
                print("\n[AI Logic] Critical Fall conditions verified. Triggering network sdk event routing...")
                try:
                    self.alert_queue.put_nowait("fall_detected")
                    self.alert_queue.put_nowait("activity_Emergency: Responding to Physical Fall")
                    self.coordinator.update_state(is_falling=True)
                except queue.Full:
                    pass
                
                self.cooldown_start_time = current_timestamp
                self.current_state = "COOLDOWN"

            elif self.current_state == "COOLDOWN":
                if current_timestamp - self.cooldown_start_time >= self.COOLDOWN_DURATION:
                    self.current_state = "IDLE"
                    self.consecutive_frames = 0
                    try:
                        self.alert_queue.put_nowait("activity_Monitoring Patient Safely")
                    except queue.Full:
                        pass
                    self.coordinator.update_state(is_falling=False)

            # System Status HUD Header Rendering
            status_text = f"ENGINE: LIVE | STATE: {self.current_state}"
            color = (0, 255, 0) if self.current_state != "COOLDOWN" else (255, 0, 0)
            if self.coordinator.get_snapshot().is_falling:
                status_text += " [CRITICAL REPORT FIRED]"
                color = (0, 0, 255)
            cv2.putText(frame, status_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            try:
                self.display_queue.put_nowait(frame)
            except queue.Full:
                try: self.display_queue.get_nowait()
                except queue.Empty: pass
                try: self.display_queue.put_nowait(frame)
                except queue.Full: pass

        cap.release()