"""
overlay.py - Draws the HUD (counter, FPS, state, instructions) onto the video frame.

Keeps all OpenCV drawing logic in one place so main.py stays clean.
"""

from __future__ import annotations
import cv2
import numpy as np

# ── Color palette (BGR) ────────────────────────────────────────────────────
_WHITE   = (255, 255, 255)
_BLACK   = (  0,   0,   0)
_GREEN   = ( 50, 220,  50)
_YELLOW  = ( 30, 220, 220)
_RED     = ( 50,  50, 220)
_CYAN    = (220, 200,  30)
_ORANGE  = ( 30, 140, 220)
_DIM     = (140, 140, 140)

# State → color mapping
_STATE_COLORS: dict[str, tuple] = {
    "NO_TRACKING":  _RED,
    "TRACKING":     _WHITE,
    "REP_COUNTED":  _GREEN,
}

# Direction → color mapping
_DIR_COLORS: dict[str, tuple] = {
    "UP":   _CYAN,
    "DOWN": _ORANGE,
    "IDLE": _DIM,
}


class Overlay:
    """Draws all HUD elements onto a copy of the video frame."""

    def __init__(self, tracking_mode: str = "hand"):
        self.tracking_mode = tracking_mode

    def draw(
        self,
        frame: np.ndarray,
        count: int,
        state_name: str,
        fps: float,
        direction: str,
        amplitude: float,
        hand_count: int,
        players: list | None = None,
    ) -> np.ndarray:
        """
        Draw the full HUD onto the frame.

        Args:
            frame       : annotated BGR frame
            count       : current rep count
            state_name  : current state name
            fps         : current frames per second
            direction   : "UP", "DOWN", or "IDLE"
            amplitude   : amplitude of last completed swing
            hand_count  : number of hands/arms currently detected
            players     : list of TrackedPlayers (optional)

        Returns:
            New BGR frame with HUD overlay.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        # ── Draw Player-Specific Skeletons & Floating HUDs ─────────────────
        if players:
            for p in players:
                if not p.rep_counter._tracking:
                    continue
                lm = p.last_landmarks
                if not lm:
                    continue

                # Draw player-specific skeletal connections
                if self.tracking_mode == "pose":
                    sh_left = (int(lm.left_shoulder.x * w), int(lm.left_shoulder.y * h))
                    sh_right = (int(lm.right_shoulder.x * w), int(lm.right_shoulder.y * h))
                    cv2.line(out, sh_left, sh_right, p.color, 3, cv2.LINE_AA)

                    if lm.left_elbow and lm.left_wrist:
                        el_left = (int(lm.left_elbow.x * w), int(lm.left_elbow.y * h))
                        wr_left = (int(lm.left_wrist.x * w), int(lm.left_wrist.y * h))
                        cv2.line(out, sh_left, el_left, p.color, 3, cv2.LINE_AA)
                        cv2.line(out, el_left, wr_left, p.color, 3, cv2.LINE_AA)
                        cv2.circle(out, el_left, 6, p.color, -1)
                        cv2.circle(out, wr_left, 6, p.color, -1)

                    if lm.right_elbow and lm.right_wrist:
                        el_right = (int(lm.right_elbow.x * w), int(lm.right_elbow.y * h))
                        wr_right = (int(lm.right_wrist.x * w), int(lm.right_wrist.y * h))
                        cv2.line(out, sh_right, el_right, p.color, 3, cv2.LINE_AA)
                        cv2.line(out, el_right, wr_right, p.color, 3, cv2.LINE_AA)
                        cv2.circle(out, el_right, 6, p.color, -1)
                        cv2.circle(out, wr_right, 6, p.color, -1)

                    cv2.circle(out, sh_left, 6, p.color, -1)
                    cv2.circle(out, sh_right, 6, p.color, -1)

                    # Floating tag positioning (overhead)
                    tag_x = int(p.center[0] * w)
                    tag_y = int(min(lm.left_shoulder.y, lm.right_shoulder.y) * h) - 35

                else:  # hand mode
                    wr = (int(lm.wrist.x * w), int(lm.wrist.y * h))
                    t_cmc = (int(lm.thumb_cmc.x * w), int(lm.thumb_cmc.y * h))
                    t_tip = (int(lm.thumb_tip.x * w), int(lm.thumb_tip.y * h))
                    i_mcp = (int(lm.index_mcp.x * w), int(lm.index_mcp.y * h))
                    i_tip = (int(lm.index_tip.x * w), int(lm.index_tip.y * h))
                    m_mcp = (int(lm.middle_mcp.x * w), int(lm.middle_mcp.y * h))
                    m_tip = (int(lm.middle_tip.x * w), int(lm.middle_tip.y * h))

                    cv2.line(out, wr, t_cmc, p.color, 2, cv2.LINE_AA)
                    cv2.line(out, t_cmc, t_tip, p.color, 2, cv2.LINE_AA)
                    cv2.line(out, wr, i_mcp, p.color, 2, cv2.LINE_AA)
                    cv2.line(out, i_mcp, i_tip, p.color, 2, cv2.LINE_AA)
                    cv2.line(out, wr, m_mcp, p.color, 2, cv2.LINE_AA)
                    cv2.line(out, m_mcp, m_tip, p.color, 2, cv2.LINE_AA)

                    for pt in [wr, t_cmc, t_tip, i_mcp, i_tip, m_mcp, m_tip]:
                        cv2.circle(out, pt, 5, p.color, -1)

                    # Floating tag positioning (above hand)
                    tag_x = int(p.center[0] * w)
                    tag_y = int(min(lm.wrist.y, lm.index_tip.y, lm.middle_tip.y) * h) - 25

                # Draw floating player score tag
                tag_x = max(50, min(w - 50, tag_x))
                tag_y = max(30, min(h - 20, tag_y))

                tag_text = f"P{p.id}: {p.rep_counter.count}"
                (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

                self._draw_panel(out, tag_x - tw//2 - 6, tag_y - th - 6, tw + 12, th + 12, alpha=0.6)
                cv2.rectangle(out, (tag_x - tw//2 - 6, tag_y - th - 6), (tag_x + tw//2 + 6, tag_y + 6), p.color, 1, cv2.LINE_AA)
                cv2.putText(out, tag_text, (tag_x - tw//2, tag_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, _WHITE, 1, cv2.LINE_AA)

        # ── Draw Top-Right Leaderboard Panel ───────────────────────────────
        if players:
            # Sort players by score, showing top 5
            sorted_players = sorted(players, key=lambda x: x.rep_counter.count, reverse=True)[:5]
            
            leaderboard_w = 200
            leaderboard_h = 35 + len(sorted_players) * 35
            self._draw_panel(out, w - leaderboard_w - 10, 10, leaderboard_w, leaderboard_h, alpha=0.65)

            cv2.putText(out, "LEADERBOARD", (w - leaderboard_w, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, _YELLOW, 2, cv2.LINE_AA)
            for idx, p in enumerate(sorted_players):
                y_pos = 65 + idx * 35
                p_color = p.color if p.rep_counter._tracking else _DIM
                cv2.circle(out, (w - leaderboard_w + 12, y_pos - 5), 6, p_color, -1)

                p_text = f"{p.name}"
                if not p.rep_counter._tracking:
                    p_text += " (Away)"
                score_text = f"{p.rep_counter.count}"
                
                text_color = _WHITE if p.rep_counter._tracking else _DIM
                cv2.putText(out, p_text, (w - leaderboard_w + 26, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, text_color, 1, cv2.LINE_AA)
                cv2.putText(out, score_text, (w - 40, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, _GREEN, 2, cv2.LINE_AA)

        # ── Draw Top-Left HUD (Main/Lead Player Focus) ────────────────────
        panel_w, panel_h = 310, 195
        self._draw_panel(out, 10, 10, panel_w, panel_h, alpha=0.55)

        # If players exist, display the leading score and details
        if players:
            leader = max(players, key=lambda x: x.rep_counter.count)
            display_count = leader.rep_counter.count
            display_state = leader.rep_counter.state_name
            display_dir = leader.direction
            display_amp = leader.amplitude
            display_title = f"LEAD: P{leader.id}"
        else:
            display_count = count
            display_state = state_name
            display_dir = direction
            display_amp = amplitude
            display_title = "67 COUNTER"

        cv2.putText(out, display_title, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, _YELLOW, 2, cv2.LINE_AA)

        # Rep count (big number)
        cv2.putText(out, str(display_count), (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.8, _GREEN, 5, cv2.LINE_AA)

        # State
        state_color = _STATE_COLORS.get(display_state, _WHITE)
        cv2.putText(out, f"State: {display_state}", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 1, cv2.LINE_AA)

        # Direction
        dir_color = _DIR_COLORS.get(display_dir, _WHITE)
        cv2.putText(out, f"Direction: {display_dir}", (20, 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, dir_color, 1, cv2.LINE_AA)

        # Active hands/arms count
        hand_label = "arm" if self.tracking_mode == "pose" else "hand"
        hand_text = f"Tracking: {hand_count} {hand_label}{'s' if hand_count != 1 else ''}"
        cv2.putText(out, hand_text, (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, _WHITE, 1, cv2.LINE_AA)

        # ── FPS & amplitude (bottom-right panel) ─────────────────────────
        self._draw_panel(out, w - 220, h - 75, 210, 65, alpha=0.45)
        cv2.putText(out, f"FPS: {fps:.1f}", (w - 210, h - 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _WHITE, 1, cv2.LINE_AA)
        cv2.putText(out, f"Amplitude: {display_amp:.3f}", (w - 210, h - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _WHITE, 1, cv2.LINE_AA)

        # ── Mode badge ─────────────────────────────────────────────────────
        # Shift mode badge left if leaderboard is showing
        badge_x = w - 190 if not players else w - 380
        mode_text = f"Mode: {self.tracking_mode.upper()}"
        cv2.putText(out, mode_text, (badge_x, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, _YELLOW, 1, cv2.LINE_AA)

        # ── Key hints (bottom-left) ────────────────────────────────────────
        hints = "[R] Reset   [Q] Quit"
        cv2.putText(out, hints, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _WHITE, 1, cv2.LINE_AA)

        return out

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _draw_panel(
        img: np.ndarray,
        x: int, y: int, w: int, h: int,
        alpha: float = 0.5,
    ):
        """Draw a semi-transparent black rectangle."""
        x2, y2 = min(x + w, img.shape[1]), min(y + h, img.shape[0])
        sub = img[y:y2, x:x2]
        black = np.zeros_like(sub)
        cv2.addWeighted(black, alpha, sub, 1 - alpha, 0, sub)
        img[y:y2, x:x2] = sub
