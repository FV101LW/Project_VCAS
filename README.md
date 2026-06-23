# VCAS: Virtual Classroom Attendance System

Real-time emotion recognition and face identification using a webcam, powered by an ensemble of EfficientNetB4 and ResNet50V2 trained on RAF-DB.

## Large Files — Download Separately

Model weights exceed GitHub 100MB limit. Download from Google Drive:
https://drive.google.com/drive/folders/1Z2tZWLJUWWdhH0xUXT6pNmPHc4q95XiF?usp=sharing

Place these in the AI Final Project folder after downloading:
- best_efficientnetb4.keras
- best_resnet50v2.keras

## Setup

1. Clone: git clone https://github.com/FV101LW/Project_VCAS.git
2. Create venv: python -m venv .venv
3. Activate: .venv\Scripts\activate
4. Install: pip install -r requirements.txt
5. Download weights from Drive link above
6. Run: python run.py
7. Press q to quit

## Model Performance

| Model | Test Accuracy (TTA) |
|---|---|
| EfficientNetB4 | 85.14% |
| ResNet50V2 | 84.65% |
| ViT-Base | 84.29% |
| 3-Model Ensemble | 87.26% |

Dataset: RAF-DB (15,000 real-world images, 7 emotion categories)

## Team
- Filippo Jason Budiyanto (s1123541)
- Wei-li Lin (s1123533)
- Darren Nicholas Suwito (s1123583)

Yuan Ze University — AI and Applications / Intro to Image Processing, 2026
