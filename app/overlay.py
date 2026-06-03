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
    ) -> np.ndarray:
        """
        Draw the full HUD onto the frame.

        Args:
            frame       : annotated BGR frame (MediaPipe landmarks already drawn)
            count       : current rep count
            state_name  : current state name (TRACKING / REP_COUNTED / NO_TRACKING)
            fps         : current frames per second
            direction   : "UP", "DOWN", or "IDLE"
            amplitude   : amplitude of last completed swing
            hand_count  : number of hands/arms currently detected

        Returns:
            New BGR frame with HUD overlay.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        # ── Semi-transparent dark panel top-left ──────────────────────────
        panel_w, panel_h = 310, 195
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

        # ── Direction ──────────────────────────────────────────────────────
        dir_color = _DIR_COLORS.get(direction, _WHITE)
        cv2.putText(out, f"Direction: {direction}", (20, 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, dir_color, 1, cv2.LINE_AA)

        # ── Hands detected ─────────────────────────────────────────────────
        hand_label = "arm" if self.tracking_mode == "pose" else "hand"
        hand_text = f"Tracking: {hand_count} {hand_label}{'s' if hand_count != 1 else ''}"
        cv2.putText(out, hand_text, (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, _WHITE, 1, cv2.LINE_AA)

        # ── FPS & amplitude (bottom-right panel) ─────────────────────────
        self._draw_panel(out, w - 220, h - 75, 210, 65, alpha=0.45)
        cv2.putText(out, f"FPS: {fps:.1f}", (w - 210, h - 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _WHITE, 1, cv2.LINE_AA)
        cv2.putText(out, f"Amplitude: {amplitude:.3f}", (w - 210, h - 22),
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
