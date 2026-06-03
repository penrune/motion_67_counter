"""
counter.py - Counts valid 67 motion repetitions with cooldown.

The MotionAnalyzer handles all cycle detection (peak/valley tracking).
This module only applies:
  - Cooldown between reps (prevents double-counting from jitter)
  - Lost-tracking reset
"""

from __future__ import annotations
import time

from app.motion_analyzer import MotionFeatures


class RepCounter:
    """
    Consumes MotionFeatures frame-by-frame and maintains a rep count.

    Cycle detection lives in the analyzer; this counter simply gates
    on cooldown timing and resets when tracking is lost.
    """

    def __init__(
        self,
        min_rep_interval: float = 0.3,
        lost_tracking_reset: float = 1.0,
    ):
        self.count: int = 0
        self._min_interval = min_rep_interval
        self._lost_reset = lost_tracking_reset

        self._last_rep_time: float = 0.0
        self._last_detected_time: float = time.time()
        self._tracking: bool = False

    # ── public API ─────────────────────────────────────────────────────────

    def update(self, features: MotionFeatures) -> bool:
        """
        Feed the current frame's motion features.

        Returns True if a new rep was counted this frame.
        """
        now = time.time()

        # ── handle lost tracking ──────────────────────────────────────────
        if features.detected:
            self._last_detected_time = now
            self._tracking = True
        else:
            if now - self._last_detected_time > self._lost_reset:
                self._tracking = False
            return False

        # ── count reps (with cooldown gating) ─────────────────────────────
        if features.rep_completed and (now - self._last_rep_time) >= self._min_interval:
            self.count += 1
            self._last_rep_time = now
            return True

        return False

    def reset(self):
        """Reset the counter and internal state."""
        self.count = 0
        self._last_rep_time = 0.0
        self._tracking = False
        print("[Counter] Reset.")

    # ── display helpers ───────────────────────────────────────────────────

    @property
    def state_name(self) -> str:
        if not self._tracking:
            return "NO_TRACKING"
        elapsed = time.time() - self._last_rep_time
        if self._last_rep_time > 0 and elapsed < 0.5:
            return "REP_COUNTED"
        return "TRACKING"
