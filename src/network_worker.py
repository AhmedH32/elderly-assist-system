import datetime
import queue
import time

import firebase_admin
from firebase_admin import credentials, messaging

from src.coordinator import StateCoordinator


class NetworkWorker:
    def __init__(self, coordinator: StateCoordinator, alert_queue: queue.Queue, robot_id: str):
        """
        Initializes the background network dispatch engine.
        
        Args:
            coordinator: The thread-safe state synchronization engine.
            alert_queue: Thread-safe FIFO channel containing verified string alerts.
            robot_id: The unique identifier matching the mobile application subscription topic.
        """
        self.coordinator = coordinator
        self.alert_queue = alert_queue
        
        # Enforces case-sensitive string matching for the Firebase topic mapping rule: Robot_{robotID}
        self.topic_name = f"Robot_{robot_id}"
        
        try:
            cred = credentials.Certificate("firebase-credentials.json")
            firebase_admin.initialize_app(cred)
            print(f"[Network] Firebase Admin SDK initialized successfully.")
            print(f"[Network] Pub/Sub subscription target locked to topic: '{self.topic_name}'")
        except Exception as e:
            print(f"[Network] CRITICAL: Failed to initialize Firebase SDK. Alerts will fall back to stdout logs. Error: {e}")

    def run(self) -> None:
        """
        Infinite execution loop running on an isolated thread.
        Ingests internal alert tokens and formats them into strict, type-safe JSON payloads
        matching the mobile client specification.
        """
        print("[Network] Background worker thread active. Awaiting event ingestion tokens...")
        
        while self.coordinator.get_snapshot().is_running:
            try:
                # Non-blocking pull with a 0.5s timeout to allow clean thread shutdown evaluation
                alert_input = self.alert_queue.get(block=True, timeout=0.5)
                print(f"[Network] Alert token extracted from queue: '{alert_input}'")
                
                # Generate strict ISO 8601 timestamp strings required by the mobile date-fns parsing engine
                iso_timestamp = datetime.datetime.now().isoformat()
                
                # Initialize default structural parameters for the Mobile RobotEvent contract
                event_type = 'normal'
                description = str(alert_input)
                trigger_system_notification = False
                
                # Map internal string tokens to the strict structural type interfaces of the mobile client
                if alert_input == "fall_detected":
                    event_type = 'urgent'
                    description = "CRITICAL EMERGENCY: A potential physical fall anomaly has been detected."
                    trigger_system_notification = True
                    
                elif alert_input == "medication_missed":
                    event_type = 'urgent'
                    description = "URGENT: Scheduled medication administration window has expired without ingestion verification."
                    trigger_system_notification = True
                    
                elif alert_input == "medication_taken":
                    event_type = 'normal'
                    description = "Health Log: Scheduled medication successfully ingested and verified by biomechanical tracking."
                    
                elif alert_input.startswith("activity_"):
                    event_type = 'activity'
                    description = alert_input.replace("activity_", "")
                
                # Construct data dictionary payload - all keys and values must map strictly to strings for FCM data parameters
                data_payload = {
                    "type": event_type,
                    "time": iso_timestamp,
                    "description": description
                }
                
                # Build an OS-level system notification block if the event type is urgent
                notification_block = None
                if trigger_system_notification:
                    notification_block = messaging.Notification(
                        title="Elderly Care Safety Alert",
                        body=description
                    )
                
                # Compile the final Message package targeting the case-sensitive pub/sub topic
                message = messaging.Message(
                    notification=notification_block,
                    data=data_payload,
                    topic=self.topic_name
                )
                
                try:
                    response = messaging.send(message)
                    print(f"[Network] FCM Broadcast successful. Server message transaction ID: {response}")
                except Exception as send_err:
                    print(f"[Network] Transmission Exception hitting Firebase cloud endpoints: {send_err}")
                
                # Signal task completion to release the queue resource
                self.alert_queue.task_done()
                
            except queue.Empty:
                # Queue empty for this polling slice; pass through to continue monitoring thread loop flag
                pass

        print("[Network] Background execution thread cleanly uncoiled.")