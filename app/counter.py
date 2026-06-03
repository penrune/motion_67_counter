"""
counter.py - State machine that counts valid "67" repetitions.

State flow:
  IDLE
    → SIX_DETECTED    (when classifier returns "SIX")
      → MOVING_TO_SEVEN (as motion transitions)
        → SEVEN_DETECTED  (classifier returns "SEVEN" + sufficient displacement)
          → REP_COUNTED (rep count incremented, cooldown timer starts)
            → IDLE (after cooldown)

The machine also resets to IDLE if landmarks are lost for too long,
or if the user holds a position without completing the motion.
"""

from __future__ import annotations
import time
from enum import Enum, auto

from app.motion_analyzer import MotionFeatures


class MotionState(Enum):
    IDLE = auto()
    SIX_DETECTED = auto()
    MOVING_TO_SEVEN = auto()
    SEVEN_DETECTED = auto()
    REP_COUNTED = auto()


class RepCounter:
    """
    Consumes classified motion positions frame-by-frame and increments
    self.count whenever a valid SIX → SEVEN transition is completed.
    """

    def __init__(
        self,
        min_rep_interval: float = 0.5,
        min_movement_distance: float = 0.08,
        lost_tracking_reset: float = 1.0,
    ):
        self.count: int = 0
        self.state: MotionState = MotionState.IDLE

        self._min_rep_interval = min_rep_interval
        self._min_movement_distance = min_movement_distance
        self._lost_tracking_reset = lost_tracking_reset

        self._last_rep_time: float = 0.0
        self._last_detected_time: float = time.time()

        # Wrist position when SIX was first detected — used to measure displacement
        self._six_wrist_pos: tuple[float, float] = (0.0, 0.0)

    # ── Public API ──────────────────────────────────────────────────────────

    def update(self, position: str, features: MotionFeatures) -> bool:
        """
        Feed the current classified position and motion features.

        Args:
            position : "SIX", "SEVEN", or "NEUTRAL"
            features : MotionFeatures from the analyzer

        Returns:
            True if a new rep was just counted this call, else False.
        """
        now = time.time()

        # ── Handle lost tracking ──────────────────────────────────────────
        if features.detected:
            self._last_detected_time = now
        else:
            if now - self._last_detected_time > self._lost_tracking_reset:
                self._transition_to(MotionState.IDLE)
            return False

        # ── State machine ─────────────────────────────────────────────────
        new_rep = False

        if self.state == MotionState.IDLE:
            if position == "SIX":
                self._six_wrist_pos = features.wrist_position
                self._transition_to(MotionState.SIX_DETECTED)

        elif self.state == MotionState.SIX_DETECTED:
            if position == "SIX":
                # Still in SIX — update reference position
                self._six_wrist_pos = features.wrist_position
            elif position == "SEVEN":
                # Jumped straight from SIX to SEVEN in one classification
                # but only if enough movement happened
                if self._sufficient_movement(features):
                    self._transition_to(MotionState.SEVEN_DETECTED)
                else:
                    self._transition_to(MotionState.IDLE)
            elif position == "NEUTRAL":
                # Started moving toward SEVEN
                self._transition_to(MotionState.MOVING_TO_SEVEN)

        elif self.state == MotionState.MOVING_TO_SEVEN:
            if position == "SEVEN":
                if self._sufficient_movement(features):
                    self._transition_to(MotionState.SEVEN_DETECTED)
                else:
                    # Not enough displacement — go back to idle
                    self._transition_to(MotionState.IDLE)
            elif position == "SIX":
                # Reversed back to SIX — restart
                self._six_wrist_pos = features.wrist_position
                self._transition_to(MotionState.SIX_DETECTED)
            # NEUTRAL: keep waiting

        elif self.state == MotionState.SEVEN_DETECTED:
            # We are in SEVEN; count the rep
            if self._cooldown_elapsed(now):
                self.count += 1
                self._last_rep_time = now
                new_rep = True
                self._transition_to(MotionState.REP_COUNTED)
            else:
                # Cooldown not elapsed; reset silently
                self._transition_to(MotionState.IDLE)

        elif self.state == MotionState.REP_COUNTED:
            # Transition back to IDLE so we can count the next rep
            self._transition_to(MotionState.IDLE)

        return new_rep

    def reset(self):
        """Reset the counter and state machine."""
        self.count = 0
        self.state = MotionState.IDLE
        self._last_rep_time = 0.0
        self._six_wrist_pos = (0.0, 0.0)
        print("[Counter] Reset.")

    # ── Helpers ──────────────────────────────────────────────────────────

    def _transition_to(self, new_state: MotionState):
        self.state = new_state

    def _sufficient_movement(self, features: MotionFeatures) -> bool:
        """Check that the wrist moved far enough from the SIX position."""
        dx = features.wrist_position[0] - self._six_wrist_pos[0]
        dy = features.wrist_position[1] - self._six_wrist_pos[1]
        distance = (dx ** 2 + dy ** 2) ** 0.5
        return distance >= self._min_movement_distance

    def _cooldown_elapsed(self, now: float) -> bool:
        return (now - self._last_rep_time) >= self._min_rep_interval

    @property
    def state_name(self) -> str:
        return self.state.name
