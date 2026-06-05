"""
main.py - Entry point for the 67 Meme Motion Counter.

Connects:
  Camera → LandmarkTracker → MotionAnalyzer → RepCounter → Overlay → display

Keys:
  R → reset counter
  Q → quit (saves session)
"""

import queue
import sys
import threading
import time

import cv2

from app.config import Config
from app.camera import Camera
from app.landmark_tracker import LandmarkTracker, LandmarkResult
from app.motion_analyzer import MotionAnalyzer
from app.overlay import Overlay
from app.storage import SessionStorage


class ThreadedTracker:
    """Runs LandmarkTracker in a background thread to prevent camera preview lag."""

    def __init__(self, mode: str, draw: bool, num_hands: int):
        self.mode = mode
        self.draw = draw
        self.num_hands = num_hands
        self.input_queue = queue.Queue(maxsize=1)
        self.output_queue = queue.Queue(maxsize=1)
        self.stopped = False

        self.init_error = None
        self.init_ready = threading.Event()

        # Start the background tracking thread
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            # MediaPipe is initialized inside the thread to avoid cross-thread C++ issues
            self.tracker = LandmarkTracker(
                mode=self.mode,
                draw=self.draw,
                num_hands=self.num_hands
            )
        except Exception as e:
            self.init_error = e
        finally:
            self.init_ready.set()

        if self.init_error:
            self.stopped = True
            return

        while not self.stopped:
            try:
                frame = self.input_queue.get(timeout=0.02)
            except queue.Empty:
                continue

            result = self.tracker.process(frame)

            # Push result to output queue, discarding older if full
            if self.output_queue.full():
                try:
                    self.output_queue.get_nowait()
                except queue.Empty:
                    pass
            self.output_queue.put(result)
            self.input_queue.task_done()

    def process_async(self, frame) -> bool:
        """Submit a frame for processing. Returns True if accepted."""
        if self.input_queue.full():
            return False
        self.input_queue.put(frame)
        return True

    def get_result(self) -> LandmarkResult | None:
        """Fetch the latest available tracking result without blocking."""
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None

    def close(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if hasattr(self, "tracker"):
            self.tracker.close()


def main():
    # ── Load configuration ────────────────────────────────────────────────
    cfg = Config()
    mode = cfg.tracking_mode          # "hand" or "pose"
    print(f"[Main] Starting in '{mode}' tracking mode.")

    # ── Initialise components ─────────────────────────────────────────────
    try:
        camera = Camera(index=cfg.camera_index)
    except RuntimeError as e:
        print(f"[Main] FATAL: {e}")
        sys.exit(1)

    # Use max_players configuration for Pose, fallback to num_hands for Hand mode
    capacity = cfg.max_players if mode == "pose" else cfg.num_hands
    print(f"[Main] Initializing background tracker (capacity={capacity})...")

    tracker = ThreadedTracker(
        mode=mode,
        draw=cfg.draw_landmarks,
        num_hands=capacity
    )

    # Wait for tracker thread setup
    tracker.init_ready.wait()
    if tracker.init_error:
        print(f"[Main] FATAL: Tracker initialization failed: {tracker.init_error}")
        if isinstance(tracker.init_error, FileNotFoundError):
            print("[Main] Run 'python setup_models.py' to download the required model files.")
        camera.release()
        sys.exit(1)

    analyzer = MotionAnalyzer(
        mode=mode,
        smoothing_factor=cfg.smoothing_factor,
        min_swing_amplitude=cfg.min_swing_amplitude,
        direction_reversal_threshold=cfg.direction_reversal_threshold,
        tracking_match_threshold=cfg.tracking_match_threshold,
        adaptive_thresholds=cfg.adaptive_thresholds,
        min_rep_interval=cfg.min_rep_interval_seconds,
        lost_tracking_reset=cfg.lost_tracking_reset_seconds,
    )

    overlay = Overlay(tracking_mode=mode)

    storage = SessionStorage()
    if cfg.save_sessions:
        storage.start_session()

    # ── FPS & Session Score Tracking ─────────────────────────────────────
    fps_history: list[float] = []
    prev_time = time.time()
    player_scores: dict[str, int] = {}

    print("[Main] Running. Press R to reset counter, Q to quit.")

    last_result = None
    last_players = []
    last_features = None

    cv2.namedWindow("67 Meme Counter", cv2.WINDOW_AUTOSIZE)

    # ── Main loop ─────────────────────────────────────────────────────────
    while True:
        frame = camera.read()
        if frame is None:
            # Give camera a moment if buffer is empty
            time.sleep(0.005)
            continue

        # ── FPS calculation ───────────────────────────────────────────────
        now = time.time()
        elapsed = now - prev_time
        prev_time = now
        fps = min(1.0 / elapsed, 120.0) if elapsed > 0.001 else 0.0
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        smooth_fps = sum(fps_history) / len(fps_history)

        # ── Submit frame for tracking ─────────────────────────────────────
        tracker.process_async(frame)

        # ── Process new tracking results if ready ─────────────────────────
        new_result = tracker.get_result()
        if new_result is not None:
            last_result = new_result
            last_features = analyzer.analyze(new_result)
            if last_features.detected:
                last_players = last_features.players
                # Update all-time session scores for players
                for p in last_players:
                    player_scores[p.name] = max(player_scores.get(p.name, 0), p.rep_counter.count)
            else:
                last_players = []

        # ── Draw HUD ──────────────────────────────────────────────────────
        # Render the custom overlay HUD
        display_frame = last_result.annotated_image if (last_result and last_result.annotated_image is not None) else frame

        total_hand_count = last_features.hand_count if last_features else 0
        best_direction = last_features.direction if last_features else "IDLE"
        best_amplitude = last_features.amplitude if last_features else 0.0

        display_frame = overlay.draw(
            frame=display_frame,
            count=0,
            state_name="TRACKING",
            fps=smooth_fps,
            direction=best_direction,
            amplitude=best_amplitude,
            hand_count=total_hand_count,
            players=last_players
        )

        cv2.imshow("67 Meme Counter", display_frame)

        # ── Keyboard input ────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            analyzer.reset()
            player_scores.clear()
            print("[Main] All counts reset.")

    # ── Cleanup ───────────────────────────────────────────────────────────
    print("[Main] Shutting down...")
    tracker.close()
    camera.release()
    cv2.destroyAllWindows()

    if cfg.save_sessions:
        avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0.0
        max_reps = max(player_scores.values()) if player_scores else 0
        storage.save_session(
            rep_count=max_reps,
            tracking_mode=mode,
            avg_fps=avg_fps,
            player_reps=player_scores,
        )

    print(f"[Main] Session ended. Max reps: {max(player_scores.values()) if player_scores else 0}")


if __name__ == "__main__":
    main()

