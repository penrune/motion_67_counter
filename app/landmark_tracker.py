"""
landmark_tracker.py - MediaPipe Tasks API landmark extraction (mediapipe 0.10+).

Supports two modes:
  "hand"  — HandLandmarker  (wrist, fingers) — up to 2 hands
  "pose"  — PoseLandmarker  (shoulder, elbow, wrist)

Model files must be present in models/ — run setup_models.py first.

Returns clean LandmarkResult dataclasses so the rest of the app
doesn't need to know about MediaPipe internals.
"""

from __future__ import annotations
import pathlib
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

# ── New Tasks API imports ──────────────────────────────────────────────────
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarksConnections,
    PoseLandmark,
)

# ── Hand landmark indices (0-20) ───────────────────────────────────────────
class _H:
    WRIST          = 0
    THUMB_CMC      = 1
    THUMB_TIP      = 4
    INDEX_MCP      = 5
    INDEX_TIP      = 8
    MIDDLE_MCP     = 9
    MIDDLE_TIP     = 12

MODELS_DIR = pathlib.Path(__file__).parent.parent / "models"


# ── Data containers ────────────────────────────────────────────────────────

@dataclass
class Point:
    """Normalized (0-1) 2-D landmark point."""
    x: float
    y: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y])


@dataclass
class HandLandmarks:
    wrist: Point
    index_mcp: Point
    index_tip: Point
    middle_mcp: Point
    middle_tip: Point
    thumb_cmc: Point
    thumb_tip: Point


@dataclass
class PoseLandmarks:
    left_shoulder: Point
    left_elbow: Point
    left_wrist: Point
    right_shoulder: Point
    right_elbow: Point
    right_wrist: Point


@dataclass
class LandmarkResult:
    """Unified result from either tracking mode."""
    mode: str
    hands: list[HandLandmarks] = field(default_factory=list)
    poses: list[PoseLandmarks] = field(default_factory=list)
    annotated_image: Optional[np.ndarray] = None
    detected: bool = False

    @property
    def pose(self) -> Optional[PoseLandmarks]:
        """First detected pose (convenience accessor for backward compatibility)."""
        return self.poses[0] if self.poses else None

    @property
    def hand(self) -> Optional[HandLandmarks]:
        """First detected hand (convenience accessor)."""
        return self.hands[0] if self.hands else None

    @property
    def hand_count(self) -> int:
        return len(self.hands)

    @property
    def pose_count(self) -> int:
        return len(self.poses)


# ── Tracker ────────────────────────────────────────────────────────────────

class LandmarkTracker:
    """
    Wraps MediaPipe HandLandmarker or PoseLandmarker (Tasks API, v0.10+)
    and returns clean LandmarkResult objects.

    Instantiate once; call process() each frame.
    """

    def __init__(self, mode: str = "hand", draw: bool = True, num_hands: int = 2):
        self.mode = mode
        self.draw = draw
        self.num_hands = num_hands
        self._start_time = time.monotonic()
        self._init_mediapipe()

    def _model_path(self, filename: str) -> str:
        p = MODELS_DIR / filename
        if not p.exists():
            raise FileNotFoundError(
                f"Model file not found: {p}\n"
                "Run  python setup_models.py  to download it."
            )
        return str(p)

    def _init_mediapipe(self):
        if self.mode == "hand":
            opts = mp_vision.HandLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=self._model_path("hand_landmarker.task")
                ),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=self.num_hands,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._detector = mp_vision.HandLandmarker.create_from_options(opts)
            print(f"[Tracker] HandLandmarker initialized (num_hands={self.num_hands}).")

        elif self.mode == "pose":
            opts = mp_vision.PoseLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=self._model_path("pose_landmarker_lite.task")
                ),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_poses=self.num_hands,
                min_pose_detection_confidence=0.6,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._detector = mp_vision.PoseLandmarker.create_from_options(opts)
            print(f"[Tracker] PoseLandmarker initialized (num_poses={self.num_hands}).")

        else:
            raise ValueError(f"Unknown tracking mode '{self.mode}'. Use 'hand' or 'pose'.")

    def process(self, bgr_frame: np.ndarray) -> LandmarkResult:
        """
        Run detection on one BGR frame.
        Uses VIDEO running mode (synchronous, no callback needed).
        """
        # Convert BGR → RGB for MediaPipe
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

        # Use real wall-clock timestamps for proper temporal smoothing
        frame_ts_ms = int((time.monotonic() - self._start_time) * 1000)

        if self.mode == "hand":
            return self._process_hand(bgr_frame, mp_image, frame_ts_ms)
        else:
            return self._process_pose(bgr_frame, mp_image, frame_ts_ms)

    # ── Hand mode ────────────────────────────────────────────────────────

    def _process_hand(
        self, bgr: np.ndarray, mp_image: Image, ts_ms: int
    ) -> LandmarkResult:
        result = self._detector.detect_for_video(mp_image, ts_ms)
        annotated = bgr.copy()

        if not result.hand_landmarks:
            return LandmarkResult(mode="hand", detected=False, annotated_image=annotated)

        hands: list[HandLandmarks] = []

        for lm in result.hand_landmarks:
            if self.draw:
                mp_drawing.draw_landmarks(
                    image=annotated,
                    landmark_list=lm,
                    connections=HandLandmarksConnections.HAND_CONNECTIONS,
                )

            def p(idx: int, _lm=lm) -> Point:
                return Point(_lm[idx].x, _lm[idx].y)

            hand = HandLandmarks(
                wrist=p(_H.WRIST),
                index_mcp=p(_H.INDEX_MCP),
                index_tip=p(_H.INDEX_TIP),
                middle_mcp=p(_H.MIDDLE_MCP),
                middle_tip=p(_H.MIDDLE_TIP),
                thumb_cmc=p(_H.THUMB_CMC),
                thumb_tip=p(_H.THUMB_TIP),
            )
            hands.append(hand)

        return LandmarkResult(
            mode="hand", hands=hands, detected=True, annotated_image=annotated
        )

    # ── Pose mode ─────────────────────────────────────────────────────────

    def _process_pose(
        self, bgr: np.ndarray, mp_image: Image, ts_ms: int
    ) -> LandmarkResult:
        result = self._detector.detect_for_video(mp_image, ts_ms)
        annotated = bgr.copy()

        if not result.pose_landmarks:
            return LandmarkResult(mode="pose", detected=False, annotated_image=annotated)

        poses: list[PoseLandmarks] = []
        for lm in result.pose_landmarks:
            if self.draw:
                mp_drawing.draw_landmarks(
                    image=annotated,
                    landmark_list=lm,
                    connections=PoseLandmarksConnections.POSE_LANDMARKS,
                )

            def p(idx, landmarks=lm) -> Point:
                return Point(landmarks[idx].x, landmarks[idx].y)

            P = PoseLandmark
            pose = PoseLandmarks(
                left_shoulder=p(P.LEFT_SHOULDER),
                left_elbow=p(P.LEFT_ELBOW),
                left_wrist=p(P.LEFT_WRIST),
                right_shoulder=p(P.RIGHT_SHOULDER),
                right_elbow=p(P.RIGHT_ELBOW),
                right_wrist=p(P.RIGHT_WRIST),
            )
            poses.append(pose)

        return LandmarkResult(
            mode="pose", poses=poses, detected=True, annotated_image=annotated
        )

    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, "_detector"):
            self._detector.close()
