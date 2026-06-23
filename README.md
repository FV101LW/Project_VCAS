# Emotion + Face Recognition Webcam

## Overview

This project runs a webcam-based emotion detection and face recognition demo using TensorFlow, OpenCV, and local LBPH face recognition.

The main script is `run.py`, which loads:
- `best_resnet50v2.keras`
- `best_efficientnetb4.keras`
- `lbph_model.yml` (optional; created automatically if missing)
- `vit_test_preds.npy` (optional logits file)
- `known_faces/` folder for named person training images

## Requirements

- Python 3.10 or 3.11 (If you don't have Python 3.10.11, install it from https://python.org/)
- `opencv-contrib-python`
- `tensorflow`
- `numpy`

## Setup Instructions

1. Open a terminal in the project folder:
   - `c:\Users\Weili_lin\Downloads\AI Final Project\AI Final Project`

2. Create a virtual environment with Python:
   - Windows: `python -m venv .venv`
   - macOS/Linux: `python3 -m venv .venv`

3. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`

4. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

5. Confirm the required model and data files are present:
   - `best_resnet50v2.keras`
   - `best_efficientnetb4.keras`
   - `vit_test_preds.npy` (optional; ignored if shape mismatches)
   - `known_faces/` directory with person subfolders (skip `auto_saved`)

   If `lbph_model.yml` is missing, the script will create it automatically from `known_faces/` images.

6. Run the webcam application:
   ```bash
   python run.py
   ```

7. Press `q` in the webcam window to exit.

## Notes

- If `lbph_model.yml` is missing, the script trains a face recognizer from images in `known_faces/`.
- `vit_test_preds.npy` is optional; if missing or invalid, the app falls back to ResNet + EfficientNet only.
- The Haar cascade file is loaded from OpenCV, so no extra download is needed.
- Use a stable camera input and a well-lit environment for best face detection performance.
