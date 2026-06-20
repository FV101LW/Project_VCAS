# Project_VCAS
VCAS: Virtual Class Attendance System finals project for AI Applications and Image Processing courses. A digital class attendance assistant program using facial and emotional attendance with a student database, tested using a standard webcam.

Language: Python

Platform/s: Google Colab, Visual Studio Code, Kaggle

Models: EfficientNetB4, ResNet50V2, ViT

Techniques: Transfer Learning, Ensemble Learning

Devices: Laptop computer, webcam

# 🎭 Emotion + Face Recognition System (RAF-DB Based)

This project is a real-time system that detects **facial emotions** and recognizes **known faces** using a webcam.

---

# 📊 Dataset

## Emotion Dataset (RAF-DB)

We use the **RAF-DB dataset** from Kaggle:  
https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset

### Emotion Classes:
- Angry  
- Disgust  
- Fear  
- Happy  
- Sad  
- Surprise  
- Neutral  

---

# 🧠 Models Used

The system uses an ensemble of deep learning models:

- ResNet50  
- EfficientNet  
- (Optional) Vision Transformer (ViT)

### Final Prediction Logic:

```python
if ViT exists:
    final = 0.4 * ResNet + 0.4 * EfficientNet + 0.2 * ViT
```
### 📌 Description:
RAF-DB contains real-world facial images collected from the internet with variations in:
- Lighting conditions  
- Facial expressions  
- Head poses  
- Age, gender, and ethnicity  

It is widely used for training deep learning models for emotion recognition.

---

# 👤 Face Recognition Dataset (Known Faces)

For identity recognition, this project uses a **custom folder-based dataset**:

```md
known_faces/
│
├── darren/
├── weili/
└── auto_saved/
```
## 🧪 How the System Works

1. Capture video from the webcam.
2. Detect faces using Haar Cascade (OpenCV).
3. Recognize identities using the LBPH face recognition model.
4. Predict emotions using deep learning models (ResNet50, EfficientNet, and optional Vision Transformer).
5. Display the detected face with the person's name, emotion label, confidence score, and bounding box.

---

## 💾 Auto Save Feature

The system automatically saves detected face images when:

- The face remains stable for at least 5 seconds.
- The emotion confidence score is greater than 0.5.

Saved images are stored in:

```text
known_faces/auto_saved/
```

---

## 📦 Output Example

For each detected face, the system displays:

- **Name** (Darren, Weili, or Unknown)
- **Emotion** (Happy, Sad, Angry, Neutral, etc.)
- **Confidence Score**
- **Color-Coded Bounding Box**

---

## 📌 Summary

- **Emotion Dataset:** RAF-DB (Kaggle)
- **Emotion Classes:** Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **Face Recognition Model:** LBPH (OpenCV)
- **Identity Dataset:** `known_faces/` containing images of Darren and Weili
- **Emotion Models:** ResNet50, EfficientNet, and optional Vision Transformer (ViT)
- **Input:** Real-time webcam feed
- **Output:** Face identity recognition and emotion detection
