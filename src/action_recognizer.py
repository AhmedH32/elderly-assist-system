import time
from typing import Tuple

import numpy as np


class MedicationTracker:
    def __init__(self, proximity_ratio: float = 0.25):
        """
        Two-Stage Sequential Temporal Heuristic.
        Models the physical biomechanics of "Pill -> Gap -> Water".
        
        Updated proximity_ratio from 0.15 to 0.25 to compensate for 
        the anatomical offset between the COCO wrist keypoint and the fingertips.
        """
        self.proximity_ratio = proximity_ratio
        
        # System States: 0=IDLE, 1=PILL_MOTION, 2=WAITING_FOR_WATER, 3=WATER_MOTION
        self.state = 0 
        
        # Biomechanical Timing Constraints (in seconds)
        self.pill_min_time = 0.2    
        self.pill_max_time = 1.5    
        self.water_min_time = 1.5   
        self.max_gap_time = 6.0     
        
        self.state_start_time = 0.0
        self.gap_start_time = 0.0
        self.medication_taken_today = False

    def _calculate_l2_norm(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def process_frame(self, keypoints: np.ndarray, bbox_height: float) -> bool:
        if self.medication_taken_today or keypoints is None:
            return False

        nose = keypoints[0]
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        
        if left_wrist[2] < 0.4 and right_wrist[2] < 0.4:
            return False

        dynamic_threshold = bbox_height * self.proximity_ratio
        
        dist_left = self._calculate_l2_norm(nose[:2], left_wrist[:2]) if left_wrist[2] > 0.4 else float('inf')
        dist_right = self._calculate_l2_norm(nose[:2], right_wrist[:2]) if right_wrist[2] > 0.4 else float('inf')
        min_dist = min(dist_left, dist_right)

        is_hand_at_mouth = min_dist < dynamic_threshold
        current_time = time.time()
        
        if self.state == 0: # IDLE
            if is_hand_at_mouth:
                print("[Med Tracker] State 1: Hand reached mouth (Possible Pill).")
                self.state = 1
                self.state_start_time = current_time

        elif self.state == 1: # PILL_MOTION
            dwell_time = current_time - self.state_start_time
            if is_hand_at_mouth:
                if dwell_time > self.pill_max_time:
                    print("[Med Tracker] Reset: Hand stayed too long to be a pill (False positive).")
                    self.state = 0
            else:
                if dwell_time >= self.pill_min_time:
                    print(f"[Med Tracker] State 2: Pill ingested ({dwell_time:.2f}s). Waiting for water...")
                    self.state = 2
                    self.gap_start_time = current_time
                else:
                    self.state = 0 

        elif self.state == 2: # WAITING_FOR_WATER
            gap_time = current_time - self.gap_start_time
            if is_hand_at_mouth:
                print("[Med Tracker] State 3: Hand returned to mouth (Possible Water).")
                self.state = 3
                self.state_start_time = current_time
            elif gap_time > self.max_gap_time:
                print("[Med Tracker] Reset: Patient did not drink water within the time limit.")
                self.state = 0

        elif self.state == 3: # WATER_MOTION
            dwell_time = current_time - self.state_start_time
            if is_hand_at_mouth:
                if dwell_time >= self.water_min_time:
                    print(f"[Med Tracker] SUCCESS: Water consumed ({dwell_time:.2f}s). Medication sequence complete!")
                    self.medication_taken_today = True
                    self.state = 0
                    return True
            else:
                print("[Med Tracker] Hand left before swallowing water. Resuming wait...")
                self.state = 2
                self.gap_start_time = current_time

        return False

    def reset_daily_tracker(self):
        self.medication_taken_today = False
        self.state = 0

    def get_telemetry(self) -> dict:
        """Extracts O(1) state data for the GUI overlay."""
        current_time = time.time()
        elapsed = 0.0
        
        if self.state == 1 or self.state == 3:
            elapsed = current_time - self.state_start_time
        elif self.state == 2:
            elapsed = current_time - self.gap_start_time
            
        return {
            "state": self.state,
            "elapsed": elapsed,
            "completed": self.medication_taken_today
        }