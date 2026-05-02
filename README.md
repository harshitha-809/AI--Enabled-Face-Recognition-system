# AI-Enabled Face Recognition System

Real-time face recognition using **OpenCV LBPH algorithm** and **Flask** — register faces via webcam and identify them live.

**Stack:** Python · OpenCV · Flask · LBPH Algorithm · HTML/CSS/JS

---

## Problem Statement

Manual identity verification is slow and error-prone. This system automates face registration and real-time recognition using a laptop webcam — no cloud API, no external service, fully local.

---

## How It Works
Webcam input → Face detection (Haar Cascade) → LBPH training → Real-time recognition → Flask UI
1. **Register** — captures 30 face samples per user, trains LBPH model, saves to `datasets/`
2. **Recognize** — streams webcam feed, detects faces, matches against trained model, overlays name

---

## Project Structure
1. **Register** — captures 30 face samples per user, trains LBPH model, saves to `datasets/`
2. **Recognize** — streams webcam feed, detects faces, matches against trained model, overlays name

---

AI--Enabled-Face-Recognition-system/
│
├── app.py                               # Flask app — routes, webcam stream, API endpoints
├── face_recognize.py                    # Core logic — registration, LBPH training, recognition
├── create_data.py                       # Manual dataset creation (optional utility)
├── check_opencv.py                      # OpenCV installation verifier
├── haarcascade_frontalface_default.xml  # Pre-trained Haar Cascade face detector
│
├── templates/
│   └── index.html                       # Web UI — register, start/stop recognition
│
├── static/
│   ├── styles.css                       # UI styling
│   └── script.js                        # Frontend JS — API calls, video stream handling
│
├── datasets/                            # Auto-created — stores face samples per user
│   └── [username]/                      # One folder per registered user
│       ├── 1.jpg
│       ├── 2.jpg
│       └── ...
│
├── Live face recognition/               # Standalone live recognition module
│
├── .gitignore                           # Excludes __pycache__, venv, face_model.xml
├── requirements.txt                     # (recommended) pip dependencies
└── README.md

---

## Setup

```bash
git clone https://github.com/harshitha-809/AI--Enabled-Face-Recognition-system.git
cd AI--Enabled-Face-Recognition-system

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install opencv-python opencv-contrib-python numpy flask
python app.py
```

Open `http://localhost:5000` in your browser.

---

## Usage

**Register a user**
1. Enter a name in the input box
2. Click **Register User** — webcam opens and captures face samples
3. Model retrains automatically

**Run recognition**
1. Click **Start Face Recognition**
2. Live webcam feed shows bounding boxes with names
3. Click **Stop Face Recognition** to end

---

## Key Technical Details

| Component | Detail |
|---|---|
| Face Detection | Haar Cascade (`haarcascade_frontalface_default.xml`) |
| Recognition Algorithm | LBPH (Local Binary Patterns Histograms) |
| Training | Per-user, incremental — retrains on each new registration |
| Backend | Flask (Python) |
| Frontend | Vanilla HTML/CSS/JS with live video stream |

---

## License

MIT
