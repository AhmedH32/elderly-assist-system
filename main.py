import datetime
import os
import queue
import threading
import time  # Added to allow network buffer flushing before shutdown

import cv2
from dotenv import load_dotenv

load_dotenv(override=True)
from src.audio_worker import AudioWorker
from src.coordinator import StateCoordinator
from src.network_worker import NetworkWorker
from src.vision_worker import VisionWorker

# =====================================================================
# SYSTEM CONFIGURATION MATRIX
# =====================================================================
TARGET_MEDICATION_TIME = datetime.datetime.now() + datetime.timedelta(minutes=2)
GRACE_PERIOD_SECONDS = 45 

def main():
    global TARGET_MEDICATION_TIME
    print("[System] Initializing Multi-Threaded Elderly Assist Architecture...")
    ROBOT_ID = os.environ.get("ROBOT_ID", "99999")
    print(f"[System] Target Node Routing Identifier: robot_{ROBOT_ID}")
    print(f"[System] Scheduled Medication Time locked for: {TARGET_MEDICATION_TIME.strftime('%H:%M:%S')}")
    
    coordinator = StateCoordinator()
    display_queue = queue.Queue(maxsize=2)
    alert_queue = queue.Queue(maxsize=10)
    
    vision_worker = VisionWorker(coordinator, display_queue, alert_queue)
    audio_worker = AudioWorker(coordinator, alert_queue)
    network_worker = NetworkWorker(coordinator, alert_queue, ROBOT_ID)
    
    threads = [
        threading.Thread(target=vision_worker.run, name="VisionThread"),
        threading.Thread(target=audio_worker.run, name="AudioThread"),
        threading.Thread(target=network_worker.run, name="NetworkThread")
    ]
    
    print("[System] Spawning background worker threads...")
    for thread in threads:
        thread.start()
        
    print("[System] All systems operational. Launching OS GUI Event Pump.")

    try:
        alert_queue.put_nowait("activity_System Online - Monitoring")
    except queue.Full:
        pass

    schedule_triggered = False
    missed_alert_fired = False

    while coordinator.get_snapshot().is_running:
        try:
            frame = display_queue.get(block=True, timeout=0.02)
            cv2.imshow("Elderly Assist System - Live Workstation Stream", frame)
        except queue.Empty:
            pass
            
        # -------------------------------------------------------------
        # THE BACKGROUND SCHEDULER LOOP (O(1) Evaluation)
        # -------------------------------------------------------------
        now = datetime.datetime.now()
        snapshot = coordinator.get_snapshot()

        if now >= TARGET_MEDICATION_TIME and not schedule_triggered and not snapshot.medication_completed:
            schedule_triggered = True
            coordinator.update_state(medication_reminder_active=True)
            try:
                alert_queue.put_nowait("activity_Reminding Patient to Ingest Medication")
            except queue.Full:
                pass
            print(f"\n[Schedule {now.strftime('%H:%M:%S')}] Medication time reached. Triggering verbal reminder.")

        if schedule_triggered and not snapshot.medication_completed and not missed_alert_fired:
            if now >= (TARGET_MEDICATION_TIME + datetime.timedelta(seconds=GRACE_PERIOD_SECONDS)):
                alert_queue.put_nowait("medication_missed")
                try:
                    alert_queue.put_nowait("activity_Alert: Scheduled Dose Missed")
                except queue.Full:
                    pass
                missed_alert_fired = True
                print("\n[Schedule] Grace period expired. Firing URGENT missed medication alert to Firebase.")

        # -------------------------------------------------------------
        # HARDWARE SIMULATION BINDS FOR LIVE PRESENTATION
        # -------------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n[Main] 'q' key registered. Dispatched final telemetry offline token...")
            try:
                # Enqueue the final status token to cleanly clear the mobile app HUD status
                alert_queue.put_nowait("activity_Robot Offline - System Shutdown")
            except queue.Full:
                pass
            
            print("[Main] Flushing socket streams... Holding shutdown sequence for 1.0s...")
            time.sleep(1.0)  # Holds main thread to give NetworkWorker time to push the packet over HTTPS
            
            print("[Main] Broadcasting cooperative thread shutdown execution signal.")
            coordinator.update_state(is_running=False)
            break
            
        if key == ord('f'):
            print("\n[Main HW Overide] Simulating physical fall vector anomaly...")
            coordinator.update_state(is_falling=True)
            try:
                # CRITICAL FIX: Explicitly pass the token down the pipeline to trigger NetworkWorker
                alert_queue.put_nowait("fall_detected")
            except queue.Full:
                pass
            
        if key == ord('s'):
            print("\n[Main HW Overide] Fast-forwarding time to scheduled medication event...")
            try:
                alert_queue.put_nowait("activity_Simulating Scheduled Ingestion Window")
            except queue.Full:
                pass
            TARGET_MEDICATION_TIME = datetime.datetime.now()
            
        if key == ord('m'):
            print("\n[Main HW Overide] Flushing medication state machine for next demonstration run...")
            vision_worker.med_tracker.reset_daily_tracker()
            audio_worker.med_success_handled = False
            try:
                alert_queue.put_nowait("activity_System Online - Monitoring")
            except queue.Full:
                pass
            coordinator.update_state(
                medication_completed=False,
                medication_reminder_active=False,
                medication_reminder_spoken=False
            )
            schedule_triggered = False
            missed_alert_fired = False
            TARGET_MEDICATION_TIME = datetime.datetime.now() + datetime.timedelta(minutes=2)

    print("\n[Main] Destroying active UI contexts...")
    cv2.destroyAllWindows()
    
    print("[Main] Synchronizing thread joins. Awaiting background uncoils...")
    for thread in threads:
        thread.join(timeout=3.0)
        print(f"[Main] Thread '{thread.name}' joined safely.")
        
    print("[System] Execution matrix cleared. Clean exit achieved.")

if __name__ == "__main__":
    main()