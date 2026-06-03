"""
main.py - Entry point for the 67 Meme Motion Counter.

Connects:
  Camera → LandmarkTracker → MotionAnalyzer → RepCounter → Overlay → display

Keys:
  R → reset counter
  Q → quit (saves session)
"""

import sys
import time

import cv2

from app.config import Config
from app.camera import Camera
from app.landmark_tracker import LandmarkTracker
from app.motion_analyzer import MotionAnalyzer
from app.counter import RepCounter
from app.overlay import Overlay
from app.storage import SessionStorage


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

    try:
        tracker = LandmarkTracker(
            mode=mode,
            draw=cfg.draw_landmarks,
            num_hands=cfg.num_hands,
        )
    except FileNotFoundError as e:
        print(f"[Main] FATAL: {e}")
        print("[Main] Run 'python setup_models.py' to download the required model files.")
        camera.release()
        sys.exit(1)

    analyzer = MotionAnalyzer(
        mode=mode,
        smoothing_factor=cfg.smoothing_factor,
        min_swing_amplitude=cfg.min_swing_amplitude,
        direction_reversal_threshold=cfg.direction_reversal_threshold,
    )

    counter = RepCounter(
        min_rep_interval=cfg.min_rep_interval_seconds,
        lost_tracking_reset=cfg.lost_tracking_reset_seconds,
    )

    overlay = Overlay(tracking_mode=mode)

    storage = SessionStorage()
    if cfg.save_sessions:
        storage.start_session()

    # ── FPS tracking ──────────────────────────────────────────────────────
    fps_history: list[float] = []
    prev_time = time.time()

    print("[Main] Running. Press R to reset counter, Q to quit.")

    # ── Main loop ─────────────────────────────────────────────────────────
    while True:
        frame = camera.read()
        if frame is None:
            print("[Main] Warning: empty frame. Retrying...")
            time.sleep(0.05)
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

        # ── Landmark detection ────────────────────────────────────────────
        result = tracker.process(frame)

        # ── Motion analysis (cycle detection happens here) ────────────────
        features = analyzer.analyze(result)

        # ── Rep counting (cooldown gating) ────────────────────────────────
        new_rep = counter.update(features)
        if new_rep:
            print(f"[Counter] Rep counted! Total: {counter.count}")

        # ── Draw HUD ──────────────────────────────────────────────────────
        display_frame = result.annotated_image if result.annotated_image is not None else frame
        display_frame = overlay.draw(
            frame=display_frame,
            count=counter.count,
            state_name=counter.state_name,
            fps=smooth_fps,
            direction=features.direction,
            amplitude=features.amplitude,
            hand_count=features.hand_count,
        )

        cv2.imshow("67 Meme Counter", display_frame)

        # ── Keyboard input ────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            counter.reset()
            analyzer.reset()

    # ── Cleanup ───────────────────────────────────────────────────────────
    print("[Main] Shutting down...")
    tracker.close()
    camera.release()
    cv2.destroyAllWindows()

    if cfg.save_sessions:
        avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0.0
        storage.save_session(
            rep_count=counter.count,
            tracking_mode=mode,
            avg_fps=avg_fps,
        )

    print(f"[Main] Session ended. Total reps: {counter.count}")


if __name__ == "__main__":
    main()
