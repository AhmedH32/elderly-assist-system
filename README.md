# Edge-to-Cloud Assist & Fall Detection Engine

An asynchronous, multi-threaded computer vision and assistive intelligence engine engineered for real-time edge monitoring, kinematic fall detection, and context-aware conversational intervention.

---

## System Architecture

The engine is decoupled into two parallel execution lanes coordinated by a lock-free centralized thread supervisor (`coordinator.py`), ensuring sub-50ms vision loop execution without UI, disk, or network blocking:

```
                  +-----------------------------------------+
                  |         Camera Frame Stream (RAM)       |
                  +--------------------+--------------------+
                                       |
             +-------------------------+-------------------------+
             | (Sub-50ms Vision Loop)                            | (Asynchronous Event Bus)
             v                                                   v
+----------------------------+                     +---------------------------+
|    Vision Worker Lane      |                     |     Coordination Lane     |
|    (src/vision_worker.py)  |                     |    (src/coordinator.py)   |
| - YOLO Pose Estimation     |                     | - Thread Orchestration    |
| - COCO Keypoint Extraction |                     | - Rising-Edge Debouncing  |
+--------------+-------------+                     +-------------+-------------+
               |                                                 |
               v                                                 v
+----------------------------+                     +---------------------------+
|  Action Recognizer Engine  |                     |  Conversational Pipeline  |
| (src/action_recognizer.py) |                     | - Cloud: Groq API (Llama) |
| - Torso Angle & CoM Drop   |                     | - Edge Roadmap: Gemma 2B  |
| - Medication Trajectories  |                     |   (LiteRT-LM / Pi 5 NVMe) |
+----------------------------+                     +-------------+-------------+
                                                                 |
                                                                 v
                                                   +---------------------------+
                                                   |     Local Speech Output   |
                                                   |   (src/audio_worker.py)   |
                                                   | - Piper Neural TTS (ONNX) |
                                                   | - Software Barge-In Mute  |
                                                   +---------------------------+
```

---

## Technical Specifications

### 1. Vision & Kinematic Safety (Highway A)
* **Zero-Storage Memory Policy:** Video frames are captured and held exclusively in volatile memory buffers. Frames are immediately overwritten and purged following pose inference; zero image data is serialized to disk or exposed externally.
* **Kinematic Fall Detection:** Computes the angular deviation of the spine vector $\vec{u}$ relative to the vertical reference axis $\vec{v} = (0, -1)$:
  $$\theta = \arccos\left(\frac{\vec{u} \cdot \vec{v}}{\Vert{}\vec{u}\Vert{} \Vert{}\vec{v}\Vert{}}\right)$$
  A fall event triggers if $\theta > 70^\circ$ concurrently with downward pelvis Center-of-Mass velocity $v_{CoM} > 1.5\text{ m/s}$ across a rolling temporal buffer.
* **Medication Adherence Tracking:** Computes Euclidean distance $d(W, M) = \Vert{}(X_W, Y_W) - (X_M, Y_M)\Vert{}_2$ between wrist keypoints and the oral cavity center, validating adherence when $d(W, M) < \epsilon$ for a sustained period of $t \ge 3.0\text{ s}$.

### 2. Acoustic & Conversational Engine (Highway B)
* **Neural Speech Synthesis:** Generates natural speech locally using an embedded ONNX runtime executing Piper TTS (`en_US-lessac-medium.onnx`).
* **Barge-In Protection:** Implements an atomic mute barrier that pauses microphone ingestion during TTS playback, preventing acoustic feedback loops.
* **Debounce State Machine:** Enforces a 30-second refractory period following any triggered incident to avoid redundant safety alerts.
* **Telemetry Sync:** Asynchronously transmits incident status packets over Firebase Cloud Messaging to the client layer.

---

## Hardware Architecture & Migration Roadmap

The system is transitioning from a host workstation prototype to a distributed, multi-SBC physical chassis:

```
+-----------------------------------------------------------------------------+
|                          ~30 cm Mobile Robotic Chassis                      |
|                                                                             |
|  [ PRIMARY BRAIN: Raspberry Pi 5 (8GB) ]                                    |
|   - Storage: PCIe Gen 2/3 NVMe M.2 HAT (High-Throughput SSD)                |
|   - Vision: Local YOLO Pose Estimation (Sub-50ms)                           |
|   - Local Reasoning: Quantized Gemma 2B via LiteRT-LM (Zero-Cloud Offline)  |
|   - Voice: Piper Neural TTS (ONNX Engine)                                   |
|   - Peripherals: Wide-angle USB Camera, Mic Array, Active Speaker           |
|                                     |                                       |
|                                     | UART / Serial Command Bus             |
|                                     v                                       |
|  [ SECONDARY COPROCESSOR: Raspberry Pi 4 ]                                  |
|   - Kinematics: Person-Following Visual Servoing                            |
|   - Safety: Ultrasonic & LiDAR Obstacle Collision Avoidance                 |
|   - Actuation: L298N Dual H-Bridge Motor Driver + DC Gearmotors             |
+-----------------------------------------------------------------------------+
```

---

## Ecosystem Integration: Companion App

The platform pairs with **`MedRobotApp`**, a decoupled cross-platform mobile client built with React Native and Expo:
* Receives asynchronous emergency push notifications via Firebase.
* Displays live operational metrics, fall event timestamps, and medication compliance history.
* Interfaces via decoupled network contracts; maintained in an independent repository.

---

## Directory Layout

```
├── config/                  # Threshold parameters, kinematics, and engine configs
├── src/
│   ├── coordinator.py       # Core thread orchestrator & rising-edge state machine
│   ├── vision_worker.py     # In-memory frame capture & YOLO pose inference
│   ├── action_recognizer.py # Vector biomechanics (fall & medication heuristics)
│   ├── audio_worker.py      # Piper ONNX voice synthesis & barge-in mute logic
│   └── network_worker.py    # Remote telemetry dispatch & cloud client integration
├── main.py                  # Entrypoint, thread lifecycle & graceful shutdown
└── requirements.txt         # Minimal production dependency manifest
```

---

## Setup & Execution

### 1. Environment Setup
```bash
python -m venv venv
source venv/Scripts/activate  # On Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

### 2. Model Assets (Local)
Place the required inference models in the project root:
* YOLO pose weights: `yolo26n-pose.pt`
* Piper TTS model: `en_US-lessac-medium.onnx` and `en_US-lessac-medium.onnx.json`

*(Note: Model weights and environment credentials are excluded from source tracking via `.gitignore`).*

### 3. Execution
```bash
python main.py
```