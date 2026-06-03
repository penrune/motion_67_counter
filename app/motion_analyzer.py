"""
motion_analyzer.py - Peak/valley cycle detection for the 67 motion counter.

Instead of classifying frames as SIX/SEVEN/NEUTRAL with static angle
thresholds, we track the vertical position (Y-coordinate) of the wrist
over time and count complete up-down oscillation cycles.

Detection pipeline:
  1. Smooth the wrist Y with an exponential moving average (EMA)
  2. Detect direction reversals (UP→DOWN or DOWN→UP) with a noise gate
  3. Measure the amplitude of each half-swing
  4. Two consecutive half-swings with sufficient amplitude = one rep
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.landmark_tracker import LandmarkResult, HandLandmarks


# ── Output data ────────────────────────────────────────────────────────────

@dataclass
class MotionFeatures:
    """Computed motion features for one frame."""
    detected: bool = False
    wrist_y: float = 0.0         # smoothed wrist Y of active hand (0=top, 1=bottom)
    direction: str = "IDLE"      # "UP", "DOWN", or "IDLE"
    amplitude: float = 0.0      # amplitude of last completed half-swing
    hand_count: int = 0          # number of hands/arms detected this frame
    rep_completed: bool = False  # True if a rep just completed this frame


# ── Per-hand / per-arm swing detector ──────────────────────────────────────

class SwingDetector:
    """
    Detects up-down oscillation cycles for a single tracked hand or arm.

    Tracks the smoothed wrist Y position, detects direction reversals,
    and counts full cycles.  Two consecutive half-swings whose amplitude
    exceeds ``min_amplitude`` are counted as one repetition.
    """

    def __init__(
        self,
        smoothing_factor: float = 0.35,
        min_amplitude: float = 0.08,
        reversal_threshold: float = 0.015,
    ):
        self.alpha = smoothing_factor
        self.min_amp = min_amplitude
        self.rev_thresh = reversal_threshold
        self._reset_state()

    # ── internal state ─────────────────────────────────────────────────────

    def _reset_state(self):
        self.smooth_y: Optional[float] = None
        self.prev_smooth_y: Optional[float] = None
        self.direction: str = "IDLE"

        # The most extreme Y reached during the current half-swing
        self.extremum_y: float = 0.0
        # Y at the point the last reversal was confirmed
        self.reversal_y: float = 0.0
        # True when one valid half-swing is "banked", waiting for a second
        self.pending_half: bool = False

        self.amplitude: float = 0.0
        self.activity: float = 0.0        # rolling score of recent movement
        self.frames_since_update: int = 0

    # ── public API ─────────────────────────────────────────────────────────

    def update(self, wrist_y: float) -> tuple[bool, str, float]:
        """
        Feed a new wrist-Y value for this frame.

        Returns
        -------
        rep_completed : bool
            True if a full oscillation cycle just completed.
        direction : str
            Current movement direction ("UP", "DOWN", or "IDLE").
        amplitude : float
            Amplitude of the last completed half-swing.
        """
        self.frames_since_update = 0

        # ── first frame: just initialise ──────────────────────────────────
        if self.smooth_y is None:
            self.smooth_y = wrist_y
            self.prev_smooth_y = wrist_y
            self.extremum_y = wrist_y
            self.reversal_y = wrist_y
            return False, "IDLE", 0.0

        # ── EMA smoothing ─────────────────────────────────────────────────
        self.prev_smooth_y = self.smooth_y
        self.smooth_y = self.alpha * wrist_y + (1.0 - self.alpha) * self.smooth_y

        delta = self.smooth_y - self.prev_smooth_y
        self.activity = 0.3 * abs(delta) + 0.7 * self.activity

        # ── track the extremum in the current half-swing ──────────────────
        if self.direction == "UP":
            # arm going up → Y is decreasing → track minimum
            if self.smooth_y < self.extremum_y:
                self.extremum_y = self.smooth_y
        elif self.direction == "DOWN":
            # arm going down → Y is increasing → track maximum
            if self.smooth_y > self.extremum_y:
                self.extremum_y = self.smooth_y
        else:
            self.extremum_y = self.smooth_y

        # ── below the noise gate? no direction update ─────────────────────
        if abs(delta) < self.rev_thresh:
            return False, self.direction, self.amplitude

        new_dir = "UP" if delta < 0 else "DOWN"

        # ── first direction after IDLE ────────────────────────────────────
        if self.direction == "IDLE":
            self.direction = new_dir
            self.reversal_y = self.smooth_y
            self.extremum_y = self.smooth_y
            return False, self.direction, self.amplitude

        # ── same direction: keep going ────────────────────────────────────
        if new_dir == self.direction:
            return False, self.direction, self.amplitude

        # ── direction reversal detected ───────────────────────────────────
        half_amp = abs(self.extremum_y - self.reversal_y)

        rep = False
        if half_amp >= self.min_amp:
            self.amplitude = half_amp
            if self.pending_half:
                # second valid half-swing → full cycle
                rep = True
                self.pending_half = False
            else:
                # first valid half-swing → bank it
                self.pending_half = True
        else:
            # swing too small → rhythm broken, reset pending
            self.pending_half = False

        # the extremum of the completed half-swing is the start of the next
        self.reversal_y = self.extremum_y
        self.extremum_y = self.smooth_y
        self.direction = new_dir

        return rep, self.direction, self.amplitude

    def tick(self):
        """Call when this detector is NOT updated (hand not visible)."""
        self.frames_since_update += 1
        if self.frames_since_update > 30:      # ~1 s at 30 fps
            self._reset_state()

    def reset(self):
        self._reset_state()


# ── Main analyser ──────────────────────────────────────────────────────────

class MotionAnalyzer:
    """
    Converts landmark data into motion features and detects 67 reps.

    For **hand mode**: maintains two SwingDetectors assigned by horizontal
    position (left/right side of frame).  Each detected hand is routed
    to the nearest detector so hands can be tracked independently.

    For **pose mode**: maintains two SwingDetectors, one per arm (left
    wrist, right wrist).
    """

    def __init__(
        self,
        mode: str = "hand",
        smoothing_factor: float = 0.35,
        min_swing_amplitude: float = 0.08,
        direction_reversal_threshold: float = 0.015,
    ):
        self.mode = mode
        self._det_kw = dict(
            smoothing_factor=smoothing_factor,
            min_amplitude=min_swing_amplitude,
            reversal_threshold=direction_reversal_threshold,
        )
        # index 0 = left side of frame, index 1 = right side
        self._detectors: list[SwingDetector] = [
            SwingDetector(**self._det_kw),
            SwingDetector(**self._det_kw),
        ]
        self._active_idx: int = 0

    # ── public API ─────────────────────────────────────────────────────────

    def analyze(self, result: LandmarkResult) -> MotionFeatures:
        """
        Process one frame of landmarks.

        Returns a MotionFeatures object.  ``rep_completed`` will be True
        if any tracked hand/arm just finished a full oscillation cycle.
        """
        if not result.detected:
            for d in self._detectors:
                d.tick()
            return MotionFeatures(detected=False)

        if self.mode == "hand":
            return self._analyze_hands(result)
        return self._analyze_pose(result)

    def reset(self):
        """Reset all detectors."""
        for d in self._detectors:
            d.reset()
        self._active_idx = 0

    # ── hand mode ──────────────────────────────────────────────────────────

    def _analyze_hands(self, result: LandmarkResult) -> MotionFeatures:
        hands = result.hands
        updated = [False, False]
        any_rep = False
        best_dir = "IDLE"
        best_amp = 0.0
        best_y = 0.0

        for hand in hands:
            # assign to left (0) or right (1) detector by x-position
            idx = 0 if hand.wrist.x < 0.5 else 1
            rep, direction, amplitude = self._detectors[idx].update(hand.wrist.y)
            updated[idx] = True

            if rep:
                any_rep = True

            # choose the more-active detector for display values
            if self._detectors[idx].activity > self._detectors[1 - idx].activity:
                self._active_idx = idx
                best_dir = direction
                best_amp = amplitude
                best_y = self._detectors[idx].smooth_y or 0.0

        # tick detectors that weren't fed this frame
        for i in range(2):
            if not updated[i]:
                self._detectors[i].tick()

        # if no hand was clearly more active, use whichever was updated
        if best_dir == "IDLE":
            for i, upd in enumerate(updated):
                if upd:
                    best_dir = self._detectors[i].direction
                    best_amp = self._detectors[i].amplitude
                    best_y = self._detectors[i].smooth_y or 0.0
                    break

        return MotionFeatures(
            detected=True,
            wrist_y=best_y,
            direction=best_dir,
            amplitude=best_amp,
            hand_count=len(hands),
            rep_completed=any_rep,
        )

    # ── pose mode ──────────────────────────────────────────────────────────

    def _analyze_pose(self, result: LandmarkResult) -> MotionFeatures:
        p = result.pose

        # detector 0 = left arm, detector 1 = right arm
        l_rep, l_dir, l_amp = self._detectors[0].update(p.left_wrist.y)
        r_rep, r_dir, r_amp = self._detectors[1].update(p.right_wrist.y)

        any_rep = l_rep or r_rep

        # pick the more-active arm for display
        if self._detectors[0].activity >= self._detectors[1].activity:
            self._active_idx = 0
            disp_dir, disp_amp = l_dir, l_amp
            disp_y = self._detectors[0].smooth_y or 0.0
        else:
            self._active_idx = 1
            disp_dir, disp_amp = r_dir, r_amp
            disp_y = self._detectors[1].smooth_y or 0.0

        return MotionFeatures(
            detected=True,
            wrist_y=disp_y,
            direction=disp_dir,
            amplitude=disp_amp,
            hand_count=2,           # pose always provides both arms
            rep_completed=any_rep,
        )
