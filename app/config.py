"""
config.py - Loads settings from config/settings.json.
Provides safe defaults for all values so the app works even if the file is missing or incomplete.
"""

import json
import os
from pathlib import Path

# Absolute path to the config file, relative to project root
CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.json"

DEFAULTS = {
    "camera_index": 0,
    "tracking_mode": "hand",           # "hand" or "pose"
    "min_rep_interval_seconds": 0.5,   # Minimum seconds between counted reps
    "movement_smoothing_window": 5,    # Number of frames to smooth over
    "six_position_angle_threshold": 45,   # Angle (deg) below this = SIX position
    "seven_position_angle_threshold": 100, # Angle (deg) above this = SEVEN position
    "min_movement_distance": 0.08,     # Minimum normalized wrist displacement to count
    "lost_tracking_reset_seconds": 1.0, # Seconds without landmarks before state reset
    "draw_landmarks": True,
    "save_sessions": True,
}


class Config:
    """Loads and exposes application settings."""

    def __init__(self):
        self.settings = dict(DEFAULTS)
        self._load()

    def _load(self):
        """Load settings from JSON, merging with defaults."""
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, "r") as f:
                    loaded = json.load(f)
                self.settings.update(loaded)
            except (json.JSONDecodeError, OSError) as e:
                print(f"[Config] Warning: could not load settings.json ({e}). Using defaults.")
        else:
            print(f"[Config] settings.json not found at {CONFIG_PATH}. Using defaults.")

    def get(self, key, fallback=None):
        return self.settings.get(key, fallback)

    def __getattr__(self, key):
        if key.startswith("_") or key == "settings":
            raise AttributeError(key)
        if key in self.settings:
            return self.settings[key]
        raise AttributeError(f"Config has no setting '{key}'")

    def save(self):
        """Persist current settings back to JSON."""
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump(self.settings, f, indent=2)
