import threading
from dataclasses import dataclass


@dataclass
class SystemState:
    """
    Plain Old Data (POD) container holding volatile system flags.
    Adding new features over the next 9 months only requires adding a field here.
    """
    is_running: bool = True
    is_falling: bool = False
    
    # Biomechanical & Scheduling Flags
    medication_completed: bool = False        # Absolute latch: Did they take it?
    medication_reminder_active: bool = False  # Has the clock reached the target time?
    medication_reminder_spoken: bool = False  # Did the AudioWorker finish speaking the reminder?
    
    is_agent_speaking: bool = False
    is_user_speaking: bool = False

class StateCoordinator:
    def __init__(self):
        # Mutual exclusion primitive to prevent race conditions during concurrent execution
        self._lock = threading.Lock()
        self._state = SystemState()

    def update_state(self, **kwargs) -> None:
        """
        Dynamically update state fields using keyword arguments inside a safe transaction.
        Time Complexity: O(K) where K is the number of fields being updated.
        """
        with self._lock:  # RAII-style lock acquisition via Python context manager
            for key, value in kwargs.items():
                if hasattr(self._state, key):
                    setattr(self._state, key, value)
                else:
                    raise AttributeError(f"SystemState does not possess field descriptor: '{key}'")

    def get_snapshot(self) -> SystemState:
        """
        Return a decoupled copy of the state vector.
        Threads should read from this snapshot to avoid holding the lock during heavy logic.
        """
        with self._lock:
            return SystemState(
                is_running=self._state.is_running,
                is_falling=self._state.is_falling,
                medication_completed=self._state.medication_completed,
                medication_reminder_active=self._state.medication_reminder_active,
                medication_reminder_spoken=self._state.medication_reminder_spoken,
                is_agent_speaking=self._state.is_agent_speaking,
                is_user_speaking=self._state.is_user_speaking
            )