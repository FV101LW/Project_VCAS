# Emotion + Face Recognition System

Real‑time emotion recognition and face identification using a webcam, powered by an ensemble of ResNet50 and EfficientNet models (trained on RAF‑DB).  
The system detects faces, recognises known people (via LBPH), predicts emotions, and automatically saves face images after stable presence (5 seconds) with emotion confidence above 0.5.

---

## 📦 Large Files – Must Download Separately

The following files are **too large for GitHub** (>100 MB) or are binary wheels.  
They are **excluded** from this repository via `.gitignore`.  
**You must download them manually** from the links below and place them in the correct folders.

| File | Purpose | Download Link | Where to Place |
|------|---------|---------------|----------------|
| `resnet.weights.h5` | ResNet50 model weights | [Download](https://drive.google.com/your-resnet-link) | `models/resnet.weights.h5` |
| `efficientnet.weights.h5` | EfficientNet model weights | [Download](https://drive.google.com/your-efficientnet-link) | `models/efficientnet.weights.h5` |
| `vit_test_preds.npy` | (Optional) NumPy predictions – not needed to run the app | [Download](https://drive.google.com/your-vit-preds-link) | `vit_test_preds.npy` (root) |
| `dlib-19.22.99-cp310-cp310-win_amd64.whl` | Dlib wheel for Windows (Python 3.10) | [Download](https://drive.google.com/your-dlib-whl-link) | (Place in root for local installation) |

> 🔁 **Replace** all placeholder links with your actual Google Drive shareable links (set to **“Anyone with the link”**).

---

## 🔒 `.gitignore` – What’s Already Ignored

To prevent large files from being accidentally pushed, the repository already includes a `.gitignore` with:

```gitignore
# Virtual environments
.venv/
venv/
env/

# IDE / OS files
.idea/
.vscode/
__pycache__/
*.pyc
.DS_Store

# Large model weight files (these are >200 MB)
*.h5
*.keras
*.pb

# Optional: ignore predictions .npy
vit_test_preds.npy

# Known faces images – often large and private
known_faces/

# Dlib wheel – users can install via pip
*.whl

# Output images
output_*.jpg
output_*.png
```

---

## 🚀 Step‑by‑Step Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Create and activate a virtual environment (recommended)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

If you encounter issues with `dlib` (especially on Windows), you can install it using the provided `.whl` file:
```bash
pip install dlib-19.22.99-cp310-cp310-win_amd64.whl
```
> **Note:** This wheel is for Python 3.10 64‑bit. If you use a different Python version, download the appropriate wheel from [this link](https://github.com/z-mahmud22/Dlib_Windows_Python3.x/).

### 4. Download and place the large weight files
- Click the links in the table above.
- Download the two `.h5` files (`resnet.weights.h5` and `efficientnet.weights.h5`).
- Create a folder named `models` in the project root.
- Place both `.h5` files inside `models/`.

- (Optional) If you want the `.npy` predictions file, place `vit_test_preds.npy` in the project root.

### 5. (Optional) Set up face recognition (LBPH)
- The system includes a pre‑trained LBPH model (`lbph_model.yml` and `lbph_labels.json`) – you can use it as is.
- To recognise specific people, create a `known_faces/` folder and inside it, **one subfolder per person** (e.g., `known_faces/Alice/`, `known_faces/Bob/`).
- Place several face images (JPG/PNG) of each person inside their folder.
- The system will automatically train (or update) the LBPH model when you run `run.py`.
- If you don't have known faces, everyone will be labelled `"Unknown"`.

### 6. Run the webcam application
```bash
python src/run.py
```
- The camera window will open.
- Press `q` to quit.
- Faces will be detected, recognised, emotions predicted, and face crops will be saved after 5 seconds of stable presence (if emotion confidence > 0.5).
- Saved images are stored in `known_faces/auto_saved/`.

---

## 🧪 Test on a Single Image
```bash
python src/test_single_image.py
```
(Adjust the image path inside the script if needed.)

---

## ⚙️ Configuration

- **Emotion ensemble weights**: Adjust in `predict_emotion()` inside `run.py`.
- **Auto‑save parameters**: `SAVE_THRESHOLD_SEC = 5`, `CONFIDENCE_THRESHOLD = 0.5` – modify as desired.
- **Frame skipping**: `FRAME_SKIP_DETECTION = 2`, `FRAME_SKIP_EMOTION = 2` – controls performance.

---

## 📌 Notes

- The emotion models expect **224×224 RGB** input with CLAHE preprocessing.
- GPU support is **not** enabled by default (CPU version). For GPU, install TensorFlow‑GPU and set up CUDA.
- All large files are ignored by Git – you will not accidentally push them.

---

## 📄 License

[MIT](https://choosealicense.com/licenses/mit/)
