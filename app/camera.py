"""
camera.py - Safe OpenCV webcam wrapper.
Handles opening, reading, and releasing the camera gracefully.
"""

import cv2


class Camera:
    """Manages a single webcam capture device."""

    def __init__(self, index: int = 0):
        self.index = index
        self.cap = None
        self._open()

    def _open(self):
        """Open the camera. Raises RuntimeError if unavailable."""
        self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {self.index}. "
                "Check that your webcam is connected and not in use by another app."
            )
        # Prefer 640x480 for speed — MediaPipe works well at this resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"[Camera] Opened camera index {self.index} — "
              f"{int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    def read(self):
        """
        Read one frame from the camera.

        Returns:
            frame (np.ndarray | None): BGR image, or None if read failed.
        """
        if self.cap is None or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        # Mirror the frame so it feels like a mirror (natural for selfie use)
        return cv2.flip(frame, 1)

    def release(self):
        """Release the camera resource."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print("[Camera] Released.")

    def __del__(self):
        self.release()
