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

from dataclasses import dataclass, field
import time
from typing import Optional
import numpy as np

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
    players: list[TrackedPlayer] = field(default_factory=list)


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


# ── Tracked Player Class ────────────────────────────────────────────────────

class TrackedPlayer:
    """
    Tracks state, motion metrics, and rep count for a single person/hand.
    """

    def __init__(self, player_id: int, center: np.ndarray, tracking_mode: str, det_kw: dict):
        self.id = player_id
        self.name = f"Player {player_id}"
        self.center = center
        self.last_seen = time.time()
        self.tracking_mode = tracking_mode
        self.det_kw = det_kw

        # Colors for visualization (BGR)
        self.color = self._get_color(player_id)

        # Swing detectors (0 = left hand/wrist/arm, 1 = right hand/wrist/arm)
        # Note: in pose mode, we track both left and right wrists. In hand mode, we track wrist Y.
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

    def update_motion(self, left_y: Optional[float], right_y: Optional[float], scale: float, adaptive: bool, landmarks=None):
        self.last_seen = time.time()
        self.last_landmarks = landmarks

        base_amp = self.det_kw.get("min_amplitude", 0.08)
        if adaptive and scale > 0:
            ref_scale = 0.20 if self.tracking_mode == "pose" else 0.12
            ratio = scale / ref_scale
            ratio = max(0.4, min(1.5, ratio))
            scaled_amp = base_amp * ratio
        else:
            scaled_amp = base_amp

        # Update both detectors with the possibly scaled amplitude threshold
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

        # Update player display features (wrist_y, direction, amplitude, activity)
        # using the more active detector/arm
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

        # Feed the rep counter
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


# ── Main analyser ──────────────────────────────────────────────────────────

class MotionAnalyzer:
    """
    Maintains multiple TrackedPlayers and routes detections to them.
    Differentiates players by Horiz/Vert proximity using Euclidean distance.
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
    ):
        self.mode = mode
        self.tracking_match_threshold = tracking_match_threshold
        self.adaptive_thresholds = adaptive_thresholds

        self.det_kw = dict(
            smoothing_factor=smoothing_factor,
            min_amplitude=min_swing_amplitude,
            reversal_threshold=direction_reversal_threshold,
            min_rep_interval=min_rep_interval,
            lost_tracking_reset=lost_tracking_reset,
        )

        self.players: dict[int, TrackedPlayer] = {}
        self.next_player_id = 1

    def analyze(self, result: LandmarkResult) -> MotionFeatures:
        """
        Process a list of hands/poses from the LandmarkResult.
        Matches them to players, tracks coordinates, counts reps, and ages out lost tracked entities.
        """
        if not result.detected:
            # Tick all players
            for p in list(self.players.values()):
                p.tick_lost()
                if time.time() - p.last_seen > self.det_kw["lost_tracking_reset"]:
                    del self.players[p.id]
            return MotionFeatures(detected=False)

        # 1. Get candidates with center position and scale
        candidates = []
        if self.mode == "hand":
            for hand in result.hands:
                # Average hand landmarks (wrist, CMC, index MCP, middle MCP) for stable center
                pts = [
                    hand.wrist.as_array(),
                    hand.index_mcp.as_array(),
                    hand.middle_mcp.as_array(),
                    hand.thumb_cmc.as_array()
                ]
                center = np.mean(pts, axis=0)
                # Hand scale: wrist to middle finger MCP
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
                # Shoulder scale: distance between left and right shoulders
                scale = np.linalg.norm(l_sh - r_sh)

                candidates.append({
                    "center": center,
                    "scale": scale,
                    "left_y": pose.left_wrist.y if pose.left_wrist else None,
                    "right_y": pose.right_wrist.y if pose.right_wrist else None,
                    "landmarks": pose
                })

        # 2. Greedy bipartite matching
        matched_candidates = set()
        matched_players = set()

        pairs = []
        for c_idx, cand in enumerate(candidates):
            for p_id, player in self.players.items():
                dist = np.linalg.norm(cand["center"] - player.center)
                pairs.append((dist, c_idx, p_id))

        # Sort pairs by distance
        pairs.sort(key=lambda x: x[0])

        for dist, c_idx, p_id in pairs:
            if c_idx in matched_candidates or p_id in matched_players:
                continue
            if dist < self.tracking_match_threshold:
                matched_candidates.add(c_idx)
                matched_players.add(p_id)

                # Update matched player
                player = self.players[p_id]
                player.center = candidates[c_idx]["center"]
                player.update_motion(
                    left_y=candidates[c_idx]["left_y"],
                    right_y=candidates[c_idx]["right_y"],
                    scale=candidates[c_idx]["scale"],
                    adaptive=self.adaptive_thresholds,
                    landmarks=candidates[c_idx]["landmarks"]
                )

        # 3. Create new players for unmatched candidates
        for c_idx, cand in enumerate(candidates):
            if c_idx not in matched_candidates:
                new_id = self.next_player_id
                self.next_player_id += 1
                new_player = TrackedPlayer(
                    player_id=new_id,
                    center=cand["center"],
                    tracking_mode=self.mode,
                    det_kw=self.det_kw
                )
                new_player.update_motion(
                    left_y=cand["left_y"],
                    right_y=cand["right_y"],
                    scale=cand["scale"],
                    adaptive=self.adaptive_thresholds,
                    landmarks=cand["landmarks"]
                )
                self.players[new_id] = new_player

        # 4. Tick unmatched players and remove aged out players
        for p_id, player in list(self.players.items()):
            if p_id not in matched_players:
                player.tick_lost()
                if time.time() - player.last_seen > self.det_kw["lost_tracking_reset"]:
                    del self.players[p_id]

        # 5. Return results
        active_list = list(self.players.values())
        if not active_list:
            return MotionFeatures(detected=False)

        # Sort by player ID for consistent UI display
        active_list.sort(key=lambda x: x.id)

        # For backward compatibility, pick the most active player
        best_player = max(active_list, key=lambda x: x.activity)
        any_rep = any(p.rep_completed_this_frame for p in active_list)

        return MotionFeatures(
            detected=True,
            wrist_y=best_player.wrist_y,
            direction=best_player.direction,
            amplitude=best_player.amplitude,
            hand_count=len(candidates),
            rep_completed=any_rep,
            players=active_list
        )

    def reset(self):
        """Reset all tracked players."""
        self.players.clear()
        self.next_player_id = 1
