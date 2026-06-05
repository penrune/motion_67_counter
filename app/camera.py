"""
camera.py - Safe OpenCV webcam wrapper using a background thread for non-blocking reads.
Handles opening, reading, and releasing the camera gracefully.
"""

import threading
import time
import cv2
import numpy as np


class Camera:
    """Manages a single webcam capture device using a background thread."""

    def __init__(self, index: int = 0):
        self.index = index
        self.cap = None
        self.frame = None
        self.ret = False
        self.stopped = False
        self.lock = threading.Lock()

        self._open()

        # Start the background frame grabber thread
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

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

    def _update(self):
        """Background thread loop to continuously read frames."""
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.01)
                continue

            ret, frame = self.cap.read()
            if ret and frame is not None:
                # Mirror the frame here to offload from main thread
                mirrored = cv2.flip(frame, 1)
                with self.lock:
                    self.frame = mirrored
                    self.ret = True
            else:
                with self.lock:
                    self.ret = False

            # Minimal sleep to prevent CPU pinning
            time.sleep(0.005)

    def read(self) -> np.ndarray | None:
        """
        Read the latest mirrored frame from the camera buffer.

        Returns:
            frame (np.ndarray | None): BGR image, or None if no frame is available.
        """
        with self.lock:
            if not self.ret:
                return None
            return self.frame.copy() if self.frame is not None else None

    def release(self):
        """Release the camera resource and stop the background thread."""
        self.stopped = True
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print("[Camera] Released.")

    def __del__(self):
        self.release()

