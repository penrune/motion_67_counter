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

# State → color mapping
_STATE_COLORS: dict[str, tuple] = {
    "IDLE":             _WHITE,
    "SIX_DETECTED":     _CYAN,
    "MOVING_TO_SEVEN":  _YELLOW,
    "SEVEN_DETECTED":   _GREEN,
    "REP_COUNTED":      _GREEN,
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
        position_label: str,
        angle: float,
    ) -> np.ndarray:
        """
        Draw the full HUD onto the frame.

        Args:
            frame         : annotated BGR frame (MediaPipe landmarks already drawn)
            count         : current rep count
            state_name    : current MotionState name
            fps           : current frames per second
            position_label: "SIX", "SEVEN", or "NEUTRAL"
            angle         : smoothed primary angle value

        Returns:
            New BGR frame with HUD overlay.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        # ── Semi-transparent dark panel top-left ──────────────────────────
        panel_w, panel_h = 300, 160
        self._draw_panel(out, 10, 10, panel_w, panel_h, alpha=0.55)

        # ── 67 Counter label ──────────────────────────────────────────────
        cv2.putText(out, "67 COUNTER", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, _YELLOW, 2, cv2.LINE_AA)

        # ── Rep count (big number) ─────────────────────────────────────────
        cv2.putText(out, str(count), (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.8, _GREEN, 5, cv2.LINE_AA)

        # ── State ──────────────────────────────────────────────────────────
        state_color = _STATE_COLORS.get(state_name, _WHITE)
        cv2.putText(out, f"State: {state_name}", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 1, cv2.LINE_AA)

        # ── Position label (SIX / SEVEN / NEUTRAL) ─────────────────────────
        pos_color = {
            "SIX":     _CYAN,
            "SEVEN":   _ORANGE,
            "NEUTRAL": _WHITE,
        }.get(position_label, _WHITE)
        cv2.putText(out, f"Position: {position_label}", (20, 162),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, pos_color, 1, cv2.LINE_AA)

        # ── FPS & angle (bottom-right panel) ──────────────────────────────
        self._draw_panel(out, w - 200, h - 75, 190, 65, alpha=0.45)
        cv2.putText(out, f"FPS: {fps:.1f}", (w - 190, h - 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _WHITE, 1, cv2.LINE_AA)
        cv2.putText(out, f"Angle: {angle:.1f} deg", (w - 190, h - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _WHITE, 1, cv2.LINE_AA)

        # ── Mode badge ─────────────────────────────────────────────────────
        mode_text = f"Mode: {self.tracking_mode.upper()}"
        cv2.putText(out, mode_text, (w - 190, 30),
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
