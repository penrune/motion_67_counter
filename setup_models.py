"""
setup_models.py - Downloads required MediaPipe model files on first run.
Run once before main.py:  python setup_models.py

Models are saved to the models/ folder (~20-30 MB total).
No API key needed — these are free public model files from Google.
"""

import urllib.request
import pathlib
import sys

MODELS_DIR = pathlib.Path(__file__).parent / "models"

MODELS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    ),
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    ),
}


def download(name: str, url: str):
    dest = MODELS_DIR / name
    if dest.exists():
        print(f"  ✓ {name} already present ({dest.stat().st_size // 1024} KB)")
        return
    print(f"  ↓ Downloading {name} ...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"done ({dest.stat().st_size // 1024} KB)")
    except Exception as e:
        print(f"\n  ERROR: {e}")
        print(f"  Try downloading manually:\n    {url}\n  → save to: {dest}")
        sys.exit(1)


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading MediaPipe model files...")
    for name, url in MODELS.items():
        download(name, url)
    print("\nAll models ready. You can now run:  python main.py")


if __name__ == "__main__":
    main()
