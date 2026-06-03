# 67 Meme Motion Counter 🤙

A local Python app that uses your webcam to detect and count the "six-seven" arm/hand meme motion — **no internet, no API keys, runs entirely on your machine.**

---

## What It Does

The app watches your webcam, detects your hand or arm landmarks using MediaPipe, and counts each time you complete one full "six → seven" motion cycle. A live OpenCV window shows:

- **67 Counter** — your rep count in large green text
- **State** — current state-machine position (IDLE, SIX_DETECTED, …)
- **Position** — what the classifier currently sees (SIX / SEVEN / NEUTRAL)
- **Angle** — the raw geometric measurement used for classification
- **FPS** — frames per second

---

## 1. Installation

### Prerequisites
- Python 3.10 or newer
- A working webcam

### Steps

```bash
# 1. Clone or download the project folder
cd motion_67_counter

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

That's it. No API keys, no cloud setup.

---

## 2. Running the App

```bash
python main.py
```

A window called **"67 Meme Counter"** will open. Stand or sit in front of your webcam and perform the "67" meme motion.

**Keyboard controls:**

| Key | Action |
|-----|--------|
| `R` | Reset the counter to 0 |
| `Q` | Quit and save the session |

---

## 3. How the Motion Counter Works

### The Motion

The "67" meme involves a rhythmic arm sweep:

1. **SIX position** — your arm/hand points downward or is in a low/bent position
2. **SEVEN position** — your arm/hand swings upward or outward to a higher/more extended position

Each complete **SIX → SEVEN** transition counts as **one rep**.

### Detection Pipeline

```
Webcam frame
   ↓
LandmarkTracker (MediaPipe)
   ↓  hand or pose landmarks
MotionAnalyzer
   ↓  smoothed angle + wrist position
Classifier → "SIX" / "SEVEN" / "NEUTRAL"
   ↓
RepCounter (state machine)
   ↓  counts valid transitions
Overlay + display
```

### State Machine

```
IDLE
  ↓  "SIX" detected
SIX_DETECTED
  ↓  position changes to NEUTRAL or SEVEN
MOVING_TO_SEVEN
  ↓  "SEVEN" detected + sufficient displacement
SEVEN_DETECTED
  ↓  cooldown elapsed
REP_COUNTED  →  IDLE
```

Anti-double-counting rules:
- A **minimum movement distance** (normalized, default 0.08) must be covered between SIX and SEVEN positions.
- A **cooldown timer** (default 0.5 s) prevents the same motion from being counted twice rapidly.
- If landmarks are lost for > 1 second, the state machine resets.

---

## 4. Adjusting Thresholds

Open `config/settings.json` and edit these values. **No code changes needed.**

```json
{
  "camera_index": 0,
  "tracking_mode": "hand",
  "min_rep_interval_seconds": 0.5,
  "movement_smoothing_window": 5,
  "six_position_angle_threshold": 45,
  "seven_position_angle_threshold": 100,
  "min_movement_distance": 0.08,
  "lost_tracking_reset_seconds": 1.0,
  "draw_landmarks": true,
  "save_sessions": true
}
```

| Setting | Effect |
|---------|--------|
| `six_position_angle_threshold` | Lower = stricter "SIX" detection (must be more downward) |
| `seven_position_angle_threshold` | Higher = stricter "SEVEN" detection (must be more raised) |
| `min_movement_distance` | Higher = requires a bigger physical movement per rep |
| `min_rep_interval_seconds` | Higher = longer cooldown between reps (prevents spam) |
| `movement_smoothing_window` | Higher = smoother but more lag |

**Tuning tip:** Watch the **Angle** value in the bottom-right corner while performing the motion. Note the angle at your lowest position (SIX) and highest position (SEVEN). Set your thresholds a few degrees inside those values.

---

## 5. Switching Between Hand and Pose Tracking

In `config/settings.json`, change `"tracking_mode"`:

```json
"tracking_mode": "hand"    ← MediaPipe Hands (default, more precise for finger/wrist)
"tracking_mode": "pose"    ← MediaPipe Pose (tracks full arm: shoulder-elbow-wrist)
```

**When to use each:**

- **Hand mode** (`"hand"`) — best when your hand is clearly visible and the meme motion is mostly in the wrist/hand.
- **Pose mode** (`"pose"`) — best when the motion is a full arm swing and you want to track the elbow angle.

---

## 6. Common Problems and Fixes

| Problem | Fix |
|---------|-----|
| Camera won't open | Try `"camera_index": 1` (or 2) in settings.json |
| Counter never increments | Watch the **Angle** value; adjust thresholds to match your motion range |
| Counter jumps by 2 | Increase `"min_rep_interval_seconds"` to 0.8 or 1.0 |
| Landmarks lost constantly | Improve lighting; ensure your hand/body is fully in frame |
| Very low FPS | Set `"draw_landmarks": false` or reduce window size |
| App crashes on start | Confirm `pip install -r requirements.txt` ran without errors |

---

## 7. Session Data

Each time you quit (press `Q`), a session is saved to `data/sessions.json`:

```json
[
  {
    "start": "2025-01-15T14:30:00",
    "end": "2025-01-15T14:32:45",
    "duration_seconds": 165.0,
    "rep_count": 42,
    "tracking_mode": "hand",
    "avg_fps": 28.4
  }
]
```

Set `"save_sessions": false` to disable this.

---

## 8. Project Structure

```
motion_67_counter/
├── main.py              Entry point
├── requirements.txt
├── README.md
├── app/
│   ├── config.py        Loads settings.json with defaults
│   ├── camera.py        OpenCV webcam wrapper
│   ├── landmark_tracker.py  MediaPipe Hands + Pose
│   ├── motion_analyzer.py   Angle/distance calculations + SIX/SEVEN classifier
│   ├── counter.py       State machine rep counter
│   ├── overlay.py       HUD drawing
│   └── storage.py       Session saving (JSON)
├── config/
│   └── settings.json    All tuneable thresholds
└── data/
    └── sessions.json    Auto-created on first save
```

---

## 9. Future Improvements

After the MVP is stable, here are ways to make detection smarter:

### Calibration Mode
Add a guided calibration phase at startup: ask the user to hold the SIX position for 2 seconds, then the SEVEN position for 2 seconds. Automatically compute and save thresholds from the measured angles. No manual tuning needed.

### ML Classifier (scikit-learn)
Record landmark sequences to CSV. Train a simple RandomForest or SVM on labelled SIX/SEVEN/NEUTRAL frames. Replace the angle threshold classifier with the trained model.

### Sequence Model (PyTorch LSTM/GRU)
Capture N-frame windows of landmark coordinates as sequences. Train an LSTM to classify the full motion pattern rather than single frames. Handles more complex motion variations.

### CSV Export
Add a button to export session history as a CSV for spreadsheet analysis.

### Streamlit Dashboard
Build a `dashboard.py` with Streamlit to visualise past sessions, plot rep counts over time, and adjust settings through a web UI.

### PyQt Desktop GUI
Replace the OpenCV window with a proper PyQt6 desktop app: larger counter display, settings panel, session history table.
