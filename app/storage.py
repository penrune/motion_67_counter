"""
storage.py - Saves session data locally to data/sessions.json.

Each session record stores:
  - start/end datetime strings
  - total rep count
  - duration in seconds
  - tracking mode used
  - average FPS
"""

from __future__ import annotations
import json
import os
from datetime import datetime
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "sessions.json"


class SessionStorage:
    """Append-only session log stored as a JSON array."""

    def __init__(self):
        self._start_time: datetime | None = None
        self._sessions: list[dict] = self._load()

    def start_session(self):
        self._start_time = datetime.now()

    def save_session(
        self,
        rep_count: int,
        tracking_mode: str,
        avg_fps: float,
        player_reps: dict[str, int] | None = None,
    ):
        """Append a completed session record and write to disk."""
        if self._start_time is None:
            print("[Storage] Warning: save_session called before start_session.")
            self._start_time = datetime.now()

        end_time = datetime.now()
        duration = (end_time - self._start_time).total_seconds()

        record = {
            "start": self._start_time.isoformat(timespec="seconds"),
            "end": end_time.isoformat(timespec="seconds"),
            "duration_seconds": round(duration, 1),
            "rep_count": rep_count,
            "tracking_mode": tracking_mode,
            "avg_fps": round(avg_fps, 1),
        }
        if player_reps:
            record["player_reps"] = player_reps

        self._sessions.append(record)
        self._write()
        print(f"[Storage] Session saved: {rep_count} reps in {duration:.1f}s @ {avg_fps:.1f} FPS")

    def _load(self) -> list[dict]:
        if DATA_PATH.exists():
            try:
                with open(DATA_PATH, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            except (json.JSONDecodeError, OSError):
                pass
        return []

    def _write(self):
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DATA_PATH, "w") as f:
            json.dump(self._sessions, f, indent=2)

    def all_sessions(self) -> list[dict]:
        return list(self._sessions)
