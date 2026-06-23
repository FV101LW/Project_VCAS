# Emotion + Face Recognition (Webcam)

Real-time webcam app that detects a face, identifies who it is, and predicts their emotion. It is the live-demo module of the VCAS (Virtual Classroom Attendance System) emotion pipeline.

- **Face detection** — Haar cascade (frontal face)
- **Face recognition** — LBPH, trained on your own `known_faces/` images
- **Emotion recognition** — a weighted ensemble of two Keras CNNs (ResNet50V2 + EfficientNetB4), optionally blended with saved Vision Transformer predictions
- **Auto-save** — once a face stays in frame for ~5 seconds, a labelled snapshot is written to `known_faces/auto_saved/`

Emotions recognised: **Surprise, Fear, Disgust, Happy, Sad, Angry, Neutral**.

---

## Requirements

**Hardware**
- A working webcam

**Software**
- Python **3.11** recommended (TensorFlow 2.19 only ships wheels for roughly Python 3.9–3.12, so avoid the newest Python releases)
- PyCharm (any recent version)

**Model files** — these must sit in the same folder as the script:
- `best_resnet50v2.keras`
- `best_efficientnetb4.keras`
- `vit_test_preds.npy` *(optional — if missing, the script automatically falls back to a ResNet + EfficientNet only ensemble)*

---

## Expected folder layout

```
your_project/
├── emotion_face_recognition.py     # this script
├── best_resnet50v2.keras
├── best_efficientnetb4.keras
├── vit_test_preds.npy              # optional
└── known_faces/                    # you create this
    ├── Alice/
    │   ├── 1.jpg
    │   └── 2.jpg
    ├── Bob/
    │   ├── 1.jpg
    │   └── 2.jpg
    └── auto_saved/                 # created automatically
```

Each person gets their own subfolder named exactly as you want the label to appear on screen. Put a handful of clear, front-facing photos in each. The `auto_saved` folder is ignored during training and is created for you on first run.

---

## Step-by-step setup

All commands run in the **PyCharm terminal** (`Alt+F12`, or the **Terminal** tab at the bottom of the window).

### Step 1 — Confirm your Python version

```bash
python --version
```

If this is newer than 3.12, create a virtual environment on Python 3.11 instead (in PyCharm: **Settings → Project → Python Interpreter → Add Interpreter → Virtualenv**, and pick a 3.11 base). TensorFlow won't install on a Python version it doesn't have a build for.

### Step 2 — Check that the terminal is in your project environment

The prompt should show your virtual environment name in parentheses, e.g. `(venv)`. That tells you the terminal is pointed at the same interpreter PyCharm uses to run the script. If it isn't, you'll install packages into the wrong place and the script will fail with "no module named ...".

### Step 3 — Check pip

```bash
python -m pip --version
```

Using `python -m pip` (rather than a bare `pip`) guarantees the install goes to *this* interpreter, not some other pip earlier on your PATH. This is the single most common cause of "I installed it but it still says missing" in PyCharm.

### Step 4 — Install the dependencies

```bash
python -m pip install opencv-contrib-python numpy tensorflow==2.19.0
```

**Important — opencv:** the script uses `cv2.face.LBPHFaceRecognizer_create()`, and the `cv2.face` module ships **only** in `opencv-contrib-python`, not the plain `opencv-python` package. Install the contrib build (it already contains everything the base package has).

Do **not** keep both `opencv-python` and `opencv-contrib-python` installed at once — they clash. If you previously installed the plain version, clear both out first:

```bash
python -m pip uninstall opencv-python opencv-contrib-python
python -m pip install opencv-contrib-python
```

### Step 5 — Verify everything installed correctly

```bash
python -c "import cv2, numpy, tensorflow as tf; print('OpenCV', cv2.__version__); print('TF', tf.__version__); print('face module:', cv2.face)"
```

You want to see your OpenCV version, `TF 2.19.0`, and a line like `face module: <module 'cv2.face'>`. If that last line prints a module instead of an error, the contrib package is working.

### Step 6 — Prepare your known faces

Create the `known_faces/` folder and a subfolder per person, as shown in the layout above. The first time you run the script with no `lbph_model.yml` present, it trains the recogniser on these images and saves the trained model to `lbph_model.yml`. Anyone not in `known_faces/` shows up as **Unknown**.

### Step 7 — Run it

```bash
python emotion_face_recognition.py
```

A window opens showing your webcam with a coloured box around the detected face and a `Name: Emotion (confidence)` label. **Press `q` to quit.**

---

## Checking what you already have installed

To see the version of a single package:

```bash
python -m pip show tensorflow
```

To list everything in the current environment:

```bash
python -m pip list
```

Or in `requirements.txt` format (handy for copying into a dependencies file):

```bash
python -m pip freeze
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'tensorflow'`** (or any other package)
The package landed in a different environment than the one running the script — the classic PyCharm split between the terminal's environment and the Run button's "Project Interpreter." Two checks:
1. In the terminal, run `python -m pip show tensorflow`. If it prints nothing, reinstall with `python -m pip install tensorflow==2.19.0`.
2. If the import works in the terminal but the script still fails on Run, look at the interpreter shown in the bottom-right corner of PyCharm and repoint it to the venv where the import succeeded.

**`Could not find a version that satisfies the requirement tensorflow`**
Your Python is outside TensorFlow's supported range. Check `python --version` and, if you're ahead of ~3.12, make a venv on Python 3.11 (Step 1) and install there.

**Model fails to load** — error about file format or an unknown layer
This is almost always a TensorFlow version mismatch between where the `.keras` files were trained (Kaggle) and your machine. Match the major.minor version: check `tf.__version__` in your Kaggle notebook and pin to the same here.

**`Haar cascade not found`**
Reinstall opencv: `python -m pip install --force-reinstall opencv-contrib-python`.

**Face recognition disabled / everyone is "Unknown"**
The `known_faces/` folder is missing, empty, or its photos have no detectable frontal face. Add clear front-facing images and delete `lbph_model.yml` so the script retrains on the next run.

**Webcam won't open**
Another app may be holding the camera, or index `0` isn't your webcam. Close other camera apps; if needed, change `cv2.VideoCapture(0)` to `1`.