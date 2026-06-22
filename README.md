# Emotion + Face Recognition System

Real‑time emotion recognition and face identification using a webcam, powered by an ensemble of ResNet50 and EfficientNet models (trained on RAF‑DB).  
The system detects faces, recognises known people (via LBPH), predicts emotions, and automatically saves face images after stable presence (5 seconds) with emotion confidence above 0.5.

---

## 📦 Large Files – Must Download Separately

The pre‑trained weight files for ResNet50 and EfficientNet are **too large for GitHub** (each >200 MB).  
They are **excluded** from this repository via `.gitignore`.  
**You must download them manually** from the links below and place them in the `models/` folder.

| Model | File name | Download link |
|-------|-----------|---------------|
| ResNet50 | `resnet.weights.h5` | [Download from Google Drive](https://drive.google.com/your-resnet-link) |
| EfficientNet | `efficientnet.weights.h5` | [Download from Google Drive](https://drive.google.com/your-efficientnet-link) |

> 🔁 **Replace** the placeholder links with your actual Google Drive shareable links (set to **“Anyone with the link”**).

---

## 🔒 `.gitignore` – What’s Already Ignored

This project already ignores large files so they are **not** accidentally pushed to GitHub.  
If you are setting up the repository from scratch, create a `.gitignore` file with:

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

If you encounter issues with `dlib`, install it manually:
```bash
pip install dlib
```
(or use the provided `.whl` file if present in the repository).

### 4. Download the large weight files
- Click the links in the table above.
- Download both `.h5` files.
- Create a folder named `models` in the project root.
- Place both `.h5` files inside `models/`.

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
