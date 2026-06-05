"""
motion_analyzer.py - Face recognition-based player tracking and motion cycle detection.

Combines MediaPipe landmark tracking with OpenCV Haar Cascade face detection
and LBPH face recognition to identify players dynamically, maintain their scores,
and resume counting when they leave and return.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import time
from typing import Optional
import numpy as np
import cv2

from app.landmark_tracker import LandmarkResult, HandLandmarks
from app.counter import RepCounter


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
    players: list[TrackedPlayer] = field(default_factory=list)


# ── Per-hand / per-arm swing detector ──────────────────────────────────────

class SwingDetector:
    """
    Detects up-down oscillation cycles for a single tracked hand or arm.

    Tracks the smoothed wrist Y position, detects direction reversals,
    and counts full cycles.
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

    def _reset_state(self):
        self.smooth_y: Optional[float] = None
        self.prev_smooth_y: Optional[float] = None
        self.direction: str = "IDLE"
        self.extremum_y: float = 0.0
        self.reversal_y: float = 0.0
        self.pending_half: bool = False
        self.amplitude: float = 0.0
        self.activity: float = 0.0        # rolling score of recent movement
        self.frames_since_update: int = 0

    def update(self, wrist_y: float) -> tuple[bool, str, float]:
        self.frames_since_update = 0

        if self.smooth_y is None:
            self.smooth_y = wrist_y
            self.prev_smooth_y = wrist_y
            self.extremum_y = wrist_y
            self.reversal_y = wrist_y
            return False, "IDLE", 0.0

        self.prev_smooth_y = self.smooth_y
        self.smooth_y = self.alpha * wrist_y + (1.0 - self.alpha) * self.smooth_y

        delta = self.smooth_y - self.prev_smooth_y
        self.activity = 0.3 * abs(delta) + 0.7 * self.activity

        if self.direction == "UP":
            if self.smooth_y < self.extremum_y:
                self.extremum_y = self.smooth_y
        elif self.direction == "DOWN":
            if self.smooth_y > self.extremum_y:
                self.extremum_y = self.smooth_y
        else:
            self.extremum_y = self.smooth_y

        if abs(delta) < self.rev_thresh:
            return False, self.direction, self.amplitude

        new_dir = "UP" if delta < 0 else "DOWN"

        if self.direction == "IDLE":
            self.direction = new_dir
            self.reversal_y = self.smooth_y
            self.extremum_y = self.smooth_y
            return False, self.direction, self.amplitude

        if new_dir == self.direction:
            return False, self.direction, self.amplitude

        half_amp = abs(self.extremum_y - self.reversal_y)
        rep = False
        if half_amp >= self.min_amp:
            self.amplitude = half_amp
            if self.pending_half:
                rep = True
                self.pending_half = False
            else:
                self.pending_half = True
        else:
            self.pending_half = False

        self.reversal_y = self.extremum_y
        self.extremum_y = self.smooth_y
        self.direction = new_dir

        return rep, self.direction, self.amplitude

    def tick(self):
        self.frames_since_update += 1
        if self.frames_since_update > 30:
            self._reset_state()

    def reset(self):
        self._reset_state()


# ── Tracked Player Class ────────────────────────────────────────────────────

class TrackedPlayer:
    """
    Tracks state, motion metrics, and rep count for a single person.
    """

    def __init__(self, player_id: int, center: np.ndarray, tracking_mode: str, det_kw: dict):
        self.id = player_id
        self.name = f"Player {player_id}"
        self.center = center
        self.last_seen = time.time()
        self.tracking_mode = tracking_mode
        self.det_kw = det_kw

        # Visual attributes
        self.color = self._get_color(player_id)
        self.last_seen_face_center: Optional[np.ndarray] = None

        # Telegram session alert flag and best frame cache
        self.telegram_session_alert_sent = False
        self.high_score_alert_sent_this_run = False
        self.best_frame: Optional[np.ndarray] = None

        # Swing detectors (0 = left hand/arm, 1 = right hand/arm)
        self.detectors = [
            SwingDetector(
                smoothing_factor=det_kw.get("smoothing_factor", 0.45),
                min_amplitude=det_kw.get("min_amplitude", 0.08),
                reversal_threshold=det_kw.get("reversal_threshold", 0.015)
            ),
            SwingDetector(
                smoothing_factor=det_kw.get("smoothing_factor", 0.45),
                min_amplitude=det_kw.get("min_amplitude", 0.08),
                reversal_threshold=det_kw.get("reversal_threshold", 0.015)
            )
        ]

        self.rep_counter = RepCounter(
            min_rep_interval=det_kw.get("min_rep_interval", 0.2),
            lost_tracking_reset=det_kw.get("lost_tracking_reset", 1.0)
        )

        self.wrist_y: float = 0.0
        self.direction: str = "IDLE"
        self.amplitude: float = 0.0
        self.activity: float = 0.0
        self.rep_completed_this_frame: bool = False
        self.last_landmarks = None
        self.last_landmarks_list = []

    def _get_color(self, idx: int) -> tuple[int, int, int]:
        colors = [
            (255, 100, 100),  # Bright Cyan/Blue (BGR)
            (100, 255, 100),  # Bright Green
            (100, 100, 255),  # Bright Red
            (255, 100, 255),  # Bright Purple/Magenta
            (255, 255, 100),  # Bright Yellow
            (100, 255, 255),  # Bright Orange
        ]
        return colors[(idx - 1) % len(colors)]

    def update_motion(self, left_y: Optional[float], right_y: Optional[float], scale: float, adaptive: bool, landmarks_list=None):
        self.last_seen = time.time()
        self.last_landmarks_list = landmarks_list or []
        self.last_landmarks = landmarks_list[0] if landmarks_list else None

        base_amp = self.det_kw.get("min_amplitude", 0.08)
        if adaptive and scale > 0:
            ref_scale = 0.20 if self.tracking_mode == "pose" else 0.12
            ratio = scale / ref_scale
            if self.tracking_mode == "hand":
                ratio = min(1.0, ratio)
            ratio = max(0.4, min(1.5, ratio))
            scaled_amp = base_amp * ratio
        else:
            scaled_amp = base_amp

        for d in self.detectors:
            d.min_amp = scaled_amp

        l_rep, l_dir, l_amp = False, "IDLE", 0.0
        if left_y is not None:
            l_rep, l_dir, l_amp = self.detectors[0].update(left_y)
        else:
            self.detectors[0].tick()

        r_rep, r_dir, r_amp = False, "IDLE", 0.0
        if right_y is not None:
            r_rep, r_dir, r_amp = self.detectors[1].update(right_y)
        else:
            self.detectors[1].tick()

        if self.detectors[0].activity >= self.detectors[1].activity:
            self.direction = l_dir
            self.amplitude = l_amp
            self.wrist_y = self.detectors[0].smooth_y or (left_y if left_y is not None else 0.0)
            self.activity = self.detectors[0].activity
        else:
            self.direction = r_dir
            self.amplitude = r_amp
            self.wrist_y = self.detectors[1].smooth_y or (right_y if right_y is not None else 0.0)
            self.activity = self.detectors[1].activity

        dummy_features = MotionFeatures(
            detected=True,
            wrist_y=self.wrist_y,
            direction=self.direction,
            amplitude=self.amplitude,
            rep_completed=(l_rep or r_rep)
        )
        self.rep_completed_this_frame = self.rep_counter.update(dummy_features)

    def tick_lost(self):
        for d in self.detectors:
            d.tick()
        self.activity *= 0.95
        self.rep_completed_this_frame = False
        self.last_landmarks_list = []
        self.last_landmarks = None
        
        dummy_features = MotionFeatures(detected=False)
        self.rep_counter.update(dummy_features)


# ── Main analyser ──────────────────────────────────────────────────────────

class MotionAnalyzer:
    """
    Tracks multiple players dynamically using Haar Cascade face detection,
    an online LBPH face recognizer, and greedy proximity matching.
    """

    def __init__(
        self,
        mode: str = "hand",
        smoothing_factor: float = 0.45,
        min_swing_amplitude: float = 0.08,
        direction_reversal_threshold: float = 0.015,
        tracking_match_threshold: float = 0.25,
        adaptive_thresholds: bool = True,
        min_rep_interval: float = 0.2,
        lost_tracking_reset: float = 1.0,
        face_recognition_threshold: float = 85.0,
        max_players: int = 2,
    ):
        self.mode = mode
        self.adaptive_thresholds = adaptive_thresholds
        self.face_recognition_threshold = face_recognition_threshold

        self.det_kw = dict(
            smoothing_factor=smoothing_factor,
            min_amplitude=min_swing_amplitude,
            reversal_threshold=direction_reversal_threshold,
            min_rep_interval=min_rep_interval,
            lost_tracking_reset=lost_tracking_reset,
            max_players=max_players,
        )

        # Initialize face classifier
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Initialize face recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer_trained = False
        
        # Dynamic player mapping
        self.players: dict[int, TrackedPlayer] = {}
        self.next_player_id = 1

    def analyze(self, result: LandmarkResult) -> MotionFeatures:
        """
        Processes a LandmarkResult and camera frame to identify and update player counts.
        """
        # Determine the target frame to analyze (use raw image if available)
        frame = result.raw_image if (result.raw_image is not None) else result.annotated_image
        if frame is None:
            for p in self.players.values():
                p.tick_lost()
            return MotionFeatures(detected=False, players=list(self.players.values()))

        # ── 1. Detect and Identify Faces ─────────────────────────────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(45, 45)
        )

        # Apply Non-Maximum Suppression (NMS) to eliminate duplicate overlapping boxes
        faces = []
        for (x, y, w, h) in faces_detected:
            cx = x + w/2
            cy = y + h/2
            is_duplicate = False
            for (ux, uy, uw, uh) in faces:
                ucx = ux + uw/2
                ucy = uy + uh/2
                dist = np.sqrt((cx - ucx)**2 + (cy - ucy)**2)
                if dist < max(w, uw) * 0.5:
                    is_duplicate = True
                    break
            if not is_duplicate:
                faces.append((x, y, w, h))

        detected_faces = []
        face_detections = []

        for idx, (x, y, w, h) in enumerate(faces):
            fcx = (x + w/2) / frame.shape[1]
            fcy = (y + h/2) / frame.shape[0]
            face_center = np.array([fcx, fcy])
            
            roi = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (100, 100))

            if not self.recognizer_trained:
                predicted_id = -1
                confidence = 999.0
            else:
                predicted_id, confidence = self.recognizer.predict(roi_resized)
                
            face_detections.append({
                "idx": idx,
                "rect": (x, y, w, h),
                "center": face_center,
                "predicted_id": predicted_id,
                "confidence": confidence,
                "roi": roi_resized
            })

        # Match face detections to players using Cost-Based Greedy Matching
        # Cost = spatial_distance - 0.4 (if ID matches and confidence <= threshold)
        face_match_options = []
        for f_det in face_detections:
            for p_id, player in self.players.items():
                dist = np.linalg.norm(f_det["center"] - player.center)
                cost = dist
                if f_det["predicted_id"] == p_id and f_det["confidence"] <= self.face_recognition_threshold:
                    cost -= 0.4
                face_match_options.append((f_det["idx"], p_id, cost))

        face_match_options.sort(key=lambda x: x[2])
        matched_face_indices = set()
        matched_player_ids = set()

        for f_idx, p_id, cost in face_match_options:
            if f_idx in matched_face_indices or p_id in matched_player_ids:
                continue

            f_det = face_detections[f_idx]
            is_confident_face = (f_det["predicted_id"] == p_id and f_det["confidence"] <= self.face_recognition_threshold)
            spatial_dist = np.linalg.norm(f_det["center"] - self.players[p_id].center)

            # Accept match if face recognized or spatial distance is very small
            if is_confident_face or spatial_dist <= 0.35:
                matched_face_indices.add(f_idx)
                matched_player_ids.add(p_id)

                player = self.players[p_id]
                player.center = f_det["center"] # Keep center strictly at face center
                player.last_seen_face_center = f_det["center"]
                player.last_seen = time.time()

                detected_faces.append({
                    "player_id": p_id,
                    "center": f_det["center"],
                    "rect": f_det["rect"]
                })

                if f_det["confidence"] < 60.0:
                    self.recognizer.update([f_det["roi"]], np.array([p_id]))

        # Enroll unmatched face detections as new players
        for f_det in face_detections:
            if f_det["idx"] in matched_face_indices:
                continue

            new_id = self.next_player_id
            self.next_player_id += 1

            self.players[new_id] = TrackedPlayer(
                player_id=new_id,
                center=f_det["center"],
                tracking_mode=self.mode,
                det_kw=self.det_kw
            )
            self.players[new_id].last_seen_face_center = f_det["center"]
            self.players[new_id].last_seen = time.time()

            if not self.recognizer_trained:
                self.recognizer.train([f_det["roi"]], np.array([new_id]))
                self.recognizer_trained = True
            else:
                self.recognizer.update([f_det["roi"]], np.array([new_id]))

            detected_faces.append({
                "player_id": new_id,
                "center": f_det["center"],
                "rect": f_det["rect"]
            })
            print(f"[MotionAnalyzer] Enrolled Player {new_id} dynamically (predicted={f_det['predicted_id']}, conf={f_det['confidence']:.1f})")

        # Draw dynamic bounding boxes on the annotated frame
        if result.annotated_image is not None and detected_faces:
            for face in detected_faces:
                p_id = face["player_id"]
                x, y, w, h = face["rect"]
                player = self.players.get(p_id)
                color = player.color if player else (255, 255, 255)
                cv2.rectangle(result.annotated_image, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
                cv2.putText(
                    result.annotated_image,
                    f"Player {p_id}",
                    (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA
                )

        # ── 2. Extract Motion Candidates ─────────────────────────────────────
        candidates = []
        if self.mode == "hand":
            for hand in result.hands:
                pts = [
                    hand.wrist.as_array(),
                    hand.index_mcp.as_array(),
                    hand.middle_mcp.as_array(),
                    hand.thumb_cmc.as_array()
                ]
                center = np.mean(pts, axis=0)
                scale = np.linalg.norm(hand.wrist.as_array() - hand.middle_mcp.as_array())
                candidates.append({
                    "center": center,
                    "scale": scale,
                    "left_y": hand.wrist.y,
                    "right_y": None,
                    "landmarks": hand
                })
        else:  # pose mode
            for pose in result.poses:
                l_sh = pose.left_shoulder.as_array()
                r_sh = pose.right_shoulder.as_array()
                center = (l_sh + r_sh) / 2.0
                scale = np.linalg.norm(l_sh - r_sh)
                candidates.append({
                    "center": center,
                    "scale": scale,
                    "left_y": pose.left_wrist.y if pose.left_wrist else None,
                    "right_y": pose.right_wrist.y if pose.right_wrist else None,
                    "landmarks": pose
                })

        # ── 3. Candidate-to-Player Proximity Mapping ─────────────────────────
        # Hand candidates are matched to the closest active player's face/body center.
        # This allows multiple hand candidates (e.g. left and right hand) to map to a single player.
        player_candidates = {p_id: [] for p_id in self.players}

        for cand in candidates:
            best_p_id = None
            min_dist = float('inf')
            
            for p_id, player in self.players.items():
                # We always match relative to the face/body center (player.center)
                dist = np.linalg.norm(cand["center"] - player.center)
                if dist < min_dist:
                    min_dist = dist
                    best_p_id = p_id

            if best_p_id is not None and min_dist <= 0.45:
                player_candidates[best_p_id].append(cand)

        # ── 4. Update Player Motion States ───────────────────────────────────
        for p_id, player in self.players.items():
            cands = player_candidates[p_id]
            if cands:
                # Do NOT overwrite player.center with hand center to keep face matching stable
                if self.mode == "hand":
                    cands.sort(key=lambda x: x["center"][0])
                    player.update_motion(
                        left_y=cands[0]["left_y"],
                        right_y=cands[1]["left_y"] if len(cands) > 1 else None,
                        scale=cands[0]["scale"],
                        adaptive=self.adaptive_thresholds,
                        landmarks_list=[c["landmarks"] for c in cands]
                    )
                else:  # pose mode
                    cand = cands[0]
                    player.center = cand["center"] # pose center is shoulder center, close to face
                    player.update_motion(
                        left_y=cand["left_y"],
                        right_y=cand["right_y"],
                        scale=cand["scale"],
                        adaptive=self.adaptive_thresholds,
                        landmarks_list=[cand["landmarks"]]
                    )
            else:
                player.tick_lost()
                # If face is still detected in this frame, update center position
                face_match = next((f for f in detected_faces if f["player_id"] == p_id), None)
                if face_match is not None:
                    player.center = face_match["center"]
                    player.last_seen_face_center = face_match["center"]
                    player.last_seen = time.time()

        # ── 5. Assemble Results ──────────────────────────────────────────────
        active_list = list(self.players.values())
        if not active_list:
            return MotionFeatures(detected=False, players=[])

        any_tracked = any(p.rep_counter._tracking for p in active_list)
        best_player = max(active_list, key=lambda x: x.activity)
        any_rep = any(p.rep_completed_this_frame for p in active_list)

        return MotionFeatures(
            detected=any_tracked,
            wrist_y=best_player.wrist_y,
            direction=best_player.direction,
            amplitude=best_player.amplitude,
            hand_count=len(candidates),
            rep_completed=any_rep,
            players=active_list
        )

    def reset(self):
        """Reset all player scores and rebuild the face models for a fresh session."""
        self.players.clear()
        self.next_player_id = 1
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer_trained = False
        print("[MotionAnalyzer] Session score and face models reset.")
