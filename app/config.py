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
    "tracking_mode": "hand",              # "hand" or "pose"
    "num_hands": 2,                       # fallback for tracking_mode hand
    "max_players": 4,                     # maximum number of players/poses to track
    "min_rep_interval_seconds": 0.2,      # lower cooldown (cooldown between counted reps) for faster motion
    "smoothing_factor": 0.45,             # higher EMA alpha (less lag, faster counting)
    "min_swing_amplitude": 0.08,          # base minimum wrist-Y travel to count a swing
    "direction_reversal_threshold": 0.015,# minimum Y delta to confirm a reversal
    "lost_tracking_reset_seconds": 1.0,   # seconds without landmarks before reset
    "tracking_match_threshold": 0.25,     # distance threshold to match players across frames
    "adaptive_thresholds": True,          # auto-scale swing amplitude based on player distance
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
