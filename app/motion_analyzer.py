"""
motion_analyzer.py - Calculates geometry (angles, distances, displacement)
from landmarks and classifies them as SIX or SEVEN positions.

The "67" meme involves a rhythmic arm/hand sweep:
  SIX  position  →  arm/hand pointing downward / low angle
  SEVEN position →  arm/hand raised / high angle or outward

For HAND mode  : we measure the angle of the hand axis (wrist → middle MCP).
For POSE mode  : we measure the elbow angle (shoulder-elbow-wrist).

A smoothing window averages the last N frames to reduce jitter.
"""

from __future__ import annotations
import math
from collections import deque
from typing import Optional

import numpy as np

from app.landmark_tracker import LandmarkResult, Point


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Return the angle (degrees) at vertex B in the triangle A-B-C.
    Used for elbow angle: shoulder-elbow-wrist.
    """
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return math.degrees(math.acos(cos_angle))


def _vector_angle_deg(p1: Point, p2: Point) -> float:
    """
    Angle (degrees, 0-180) of the vector p1 → p2 relative to horizontal.
    0° = pointing right, 90° = pointing straight down.
    """
    dx = p2.x - p1.x
    dy = p2.y - p1.y   # y increases downward in image coords
    return math.degrees(math.atan2(abs(dy), abs(dx) + 1e-9))


def _distance(p1: Point, p2: Point) -> float:
    """Euclidean distance in normalized coordinates."""
    return math.hypot(p2.x - p1.x, p2.y - p1.y)


class MotionFeatures:
    """
    Computed motion features for one frame.

    primary_angle  : the key angle for this mode
    wrist_position : normalized (x, y) of the wrist landmark
    detected       : whether landmarks were found this frame
    """
    __slots__ = ("primary_angle", "wrist_position", "detected", "raw")

    def __init__(
        self,
        primary_angle: float,
        wrist_position: tuple[float, float],
        detected: bool,
        raw: dict = None,
    ):
        self.primary_angle = primary_angle
        self.wrist_position = wrist_position
        self.detected = detected
        self.raw = raw or {}


class MotionAnalyzer:
    """
    Converts a LandmarkResult into motion features and classifies the
    current position as SIX, SEVEN, or NEUTRAL.

    Uses a rolling smoothing window on the primary angle.
    """

    def __init__(
        self,
        mode: str = "hand",
        six_angle_threshold: float = 45.0,
        seven_angle_threshold: float = 100.0,
        smoothing_window: int = 5,
    ):
        self.mode = mode
        self.six_angle_threshold = six_angle_threshold
        self.seven_angle_threshold = seven_angle_threshold
        self._angle_history: deque[float] = deque(maxlen=smoothing_window)

    def analyze(self, result: LandmarkResult) -> MotionFeatures:
        """
        Extract features from a LandmarkResult.

        Returns a MotionFeatures object with smoothed angle and wrist position.
        If landmarks are not detected, returns detected=False.
        """
        if not result.detected:
            return MotionFeatures(
                primary_angle=0.0,
                wrist_position=(0.0, 0.0),
                detected=False,
            )

        if self.mode == "hand":
            return self._analyze_hand(result)
        else:
            return self._analyze_pose(result)

    # ── Hand mode ──────────────────────────────────────────────────────────

    def _analyze_hand(self, result: LandmarkResult) -> MotionFeatures:
        h = result.hand
        # Angle of the hand: wrist → middle MCP (knuckle)
        # In the "six" position (low/pointing down) this angle will be larger
        # In the "seven" position (raised/horizontal) it will be smaller
        raw_angle = _vector_angle_deg(h.wrist, h.middle_mcp)

        # Also compute the vertical position of the wrist (y in 0-1 space)
        wrist_y = h.wrist.y   # higher y = lower on screen

        self._angle_history.append(raw_angle)
        smooth_angle = float(np.mean(self._angle_history))

        return MotionFeatures(
            primary_angle=smooth_angle,
            wrist_position=(h.wrist.x, h.wrist.y),
            detected=True,
            raw={
                "raw_angle": raw_angle,
                "wrist_y": wrist_y,
                "index_tip": (h.index_tip.x, h.index_tip.y),
            },
        )

    # ── Pose mode ──────────────────────────────────────────────────────────

    def _analyze_pose(self, result: LandmarkResult) -> MotionFeatures:
        p = result.pose

        # Pick the arm with the higher (more visible) wrist — heuristic
        # Compare wrist y — lower y means higher on screen (more raised)
        if p.right_wrist.y < p.left_wrist.y:
            shoulder, elbow, wrist = p.right_shoulder, p.right_elbow, p.right_wrist
        else:
            shoulder, elbow, wrist = p.left_shoulder, p.left_elbow, p.left_wrist

        raw_angle = _angle_deg(
            shoulder.as_array(), elbow.as_array(), wrist.as_array()
        )

        self._angle_history.append(raw_angle)
            
        smooth_angle = float(np.mean(self._angle_history))

        return MotionFeatures(
            primary_angle=smooth_angle,
            wrist_position=(wrist.x, wrist.y),
            detected=True,
            raw={
                "raw_angle": raw_angle,
                "elbow": (elbow.x, elbow.y),
            },
        )

    # ── Classification ──────────────────────────────────────────────────────

    def classify(self, features: MotionFeatures) -> str:
        """
        Classify features into "SIX", "SEVEN", or "NEUTRAL".

        For HAND mode:
          - SIX   = hand angled downward   → lower angle value (wrist lower than MCP)
          - SEVEN = hand more horizontal   → higher angle value

        For POSE mode:
          - SIX   = elbow more bent        → smaller angle
          - SEVEN = arm more extended/up   → larger angle
        """
        if not features.detected:
            return "NEUTRAL"

        a = features.primary_angle

        if self.mode == "hand":
            # In image coords, a hand pointing downward yields a LARGER atan2 angle
            # So:  low angle → horizontal → SEVEN, high angle → downward → SIX
            if a <= self.six_angle_threshold:
                return "SEVEN"
            elif a >= self.seven_angle_threshold:
                return "SIX"
            else:
                return "NEUTRAL"
        else:
            # Elbow angle: small = bent (six), large = extended (seven)
            if a <= self.six_angle_threshold:
                return "SIX"
            elif a >= self.seven_angle_threshold:
                return "SEVEN"
            else:
                return "NEUTRAL"

    def wrist_displacement(self, prev: MotionFeatures, curr: MotionFeatures) -> float:
        """
        Euclidean displacement of the wrist between two frames (normalized coords).
        Used to reject tiny accidental movements.
        """
        if not prev.detected or not curr.detected:
            return 0.0
        dx = curr.wrist_position[0] - prev.wrist_position[0]
        dy = curr.wrist_position[1] - prev.wrist_position[1]
        return math.hypot(dx, dy)
