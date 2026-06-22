import os
import cv2
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax

# ------------------------------------------------------------
# 0. LOAD .NPY FILE (ViT logits) – silently ignore if shape mismatch
# ------------------------------------------------------------
vit_logits = None
try:
    vit_data = np.load('vit_test_preds.npy')
    if vit_data.shape == (7,):
        vit_logits = vit_data
        # Optional: uncomment for confirmation
        # print("✅ ViT logits loaded (shape 7).")
    # else silently ignore
except:
    pass  # no error messages

# ------------------------------------------------------------
# 1. LOAD EMOTION MODELS (ResNet + EfficientNet)
# ------------------------------------------------------------
print("📦 Loading ResNet50 from best_resnet50v2.keras...")
model_resnet = load_model("best_resnet50v2.keras")

print("📦 Loading EfficientNet from best_efficientnetb4.keras...")
model_eff = load_model("best_efficientnetb4.keras")

emotion_labels = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Neutral"]
print(f"✅ Emotion models loaded. Using labels: {emotion_labels}")

# ------------------------------------------------------------
# 2. PREPROCESSING + PREDICTION
# ------------------------------------------------------------
def sharpen_image(img):
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)

def preprocess_face_enhanced(face_img, model):
    h, w = model.input_shape[1], model.input_shape[2]
    if len(face_img.shape) == 2:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_rgb = sharpen_image(face_rgb)
    lab = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enh = clahe.apply(l)
    lab_enh = cv2.merge((l_enh, a, b))
    face_enh = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2RGB)
    face_resized = cv2.resize(face_enh, (w, h))
    face_norm = face_resized.astype('float32') / 255.0
    return np.expand_dims(face_norm, axis=0)

def predict_emotion(face_crop):
    inp_eff = preprocess_face_enhanced(face_crop, model_eff)
    p_eff = model_eff.predict(inp_eff, verbose=0)[0]
    inp_res = preprocess_face_enhanced(face_crop, model_resnet)
    p_res = model_resnet.predict(inp_res, verbose=0)[0]

    if vit_logits is not None:
        p_vit = softmax(vit_logits).numpy() if hasattr(softmax, 'numpy') else softmax(vit_logits)
        final = 0.4 * p_res + 0.4 * p_eff + 0.2 * p_vit
    else:
        final = 0.5 * p_eff + 0.5 * p_res

    class_id = np.argmax(final)
    confidence = final[class_id]
    emotion = emotion_labels[class_id] if class_id < len(emotion_labels) else str(class_id)
    return emotion, confidence

# ------------------------------------------------------------
# 3. HELPER: Bounding box colour by emotion
# ------------------------------------------------------------
def get_emotion_color(emotion):
    colors = {
        "Angry": (0, 0, 255),        # Red
        "Fear": (0, 165, 255),       # Orange
        "Disgust": (0, 128, 128),    # Olive
        "Happy": (0, 255, 0),        # Green
        "Sad": (255, 0, 0),          # Blue
        "Surprise": (255, 255, 0),   # Cyan
        "Neutral": (255, 255, 255)   # White
    }
    return colors.get(emotion, (0, 255, 0))

# ------------------------------------------------------------
# 4. MAIN WEBCAM LOOP – optimised for speed & clarity
# ------------------------------------------------------------
def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise RuntimeError("❌ Haar cascade not found. Reinstall opencv-python.")

    # ---- LBPH face recognition setup ----
    KNOWN_FACES_DIR = "known_faces"
    MODEL_FILE = "lbph_model.yml"

    def train_lbph():
        if not os.path.exists(KNOWN_FACES_DIR):
            print(f"❌ Folder '{KNOWN_FACES_DIR}' not found. Face recognition disabled.")
            return None, []
        faces, labels, label_names = [], [], []
        current_id = 0
        for person in os.listdir(KNOWN_FACES_DIR):
            person_dir = os.path.join(KNOWN_FACES_DIR, person)
            if not os.path.isdir(person_dir) or person == "auto_saved":
                continue
            label_names.append(person)
            for img_file in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                rects = face_cascade.detectMultiScale(img, 1.1, 5, minSize=(40, 40))
                if len(rects) == 0:
                    continue
                (x, y, w, h) = max(rects, key=lambda r: r[2] * r[3])
                face_roi = img[y:y + h, x:x + w]
                faces.append(face_roi)
                labels.append(current_id)
            current_id += 1
        if not faces:
            print("❌ No faces detected in training images.")
            return None, []
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(labels))
        recognizer.save(MODEL_FILE)
        print(f"✅ Trained LBPH model on {len(faces)} faces. Labels: {label_names}")
        return recognizer, label_names

    if os.path.exists(MODEL_FILE):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(MODEL_FILE)
        label_names = sorted([p for p in os.listdir(KNOWN_FACES_DIR)
                              if os.path.isdir(os.path.join(KNOWN_FACES_DIR, p)) and p != "auto_saved"])
        print(f"✅ Loaded existing LBPH model. Labels: {label_names}")
    else:
        recognizer, label_names = train_lbph()
        if recognizer is None:
            print("⚠️ Face recognition disabled.")
            label_names = []

    def recognize_face_lbph(face_gray):
        if recognizer is None or not label_names:
            return "Unknown"
        label, confidence = recognizer.predict(face_gray)
        return label_names[label] if confidence < 80 else "Unknown"

    # ---- Auto-save feature ----
    SAVE_DIR = os.path.join(KNOWN_FACES_DIR, "auto_saved")
    os.makedirs(SAVE_DIR, exist_ok=True)
    SAVE_THRESHOLD_SEC = 5
    CONFIDENCE_THRESHOLD = 0.5
    DISAPPEAR_TOLERANCE = 2.0

    face_timer = {
        "active": False,
        "start_time": 0,
        "last_seen": 0,
        "saved": False,
        "centroid": None,
    }
    FACE_MATCH_TOLERANCE = 40

    def get_centroid(x, y, w, h):
        return (x + w // 2, y + h // 2)

    def distance(c1, c2):
        return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

    def save_face_image(face_crop, name, emotion, conf_emotion, timestamp):
        if conf_emotion < CONFIDENCE_THRESHOLD:
            print(f"⏭️ Skipped save: emotion confidence {conf_emotion:.2f} < {CONFIDENCE_THRESHOLD}")
            return
        safe_name = name.replace(" ", "_")
        safe_emotion = emotion.replace(" ", "_")
        filename = f"{safe_name}_{safe_emotion}_{timestamp}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(filepath, face_crop)
        print(f"💾 Saved: {filepath} (emotion confidence: {conf_emotion:.2f})")

    # ---- Webcam initialisation ----
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        exit()

    # Optional: reduce resolution for faster processing (uncomment if needed)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ---- Detection parameters ----
    SCALE_FACTOR = 1.1
    MIN_NEIGHBORS = 7
    MIN_FACE_SIZE = (120, 120)          # ignore small faces
    FRAME_SKIP_DETECTION = 2
    FRAME_SKIP_EMOTION = 2

    detection_counter = 0
    emotion_counter = 0
    last_faces = []
    last_emotion = "Neutral"
    last_conf = 0.0

    window_name = "Emotion + Face Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("✅ Webcam started. Press 'q' to quit.")
    print(f"⚡ Optimised: detection every {FRAME_SKIP_DETECTION} frames, emotion every {FRAME_SKIP_EMOTION} frames")
    print(f"📸 Auto-save after {SAVE_THRESHOLD_SEC} seconds of stable presence (saved to '{SAVE_DIR}')")
    print(f"   Emotion confidence threshold for saving: {CONFIDENCE_THRESHOLD}\n")

    current_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detection_counter += 1
        if detection_counter % FRAME_SKIP_DETECTION == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            all_faces = face_cascade.detectMultiScale(gray, scaleFactor=SCALE_FACTOR,
                                                      minNeighbors=MIN_NEIGHBORS,
                                                      minSize=MIN_FACE_SIZE)
            if len(all_faces) > 0:
                # Pick only the largest face
                largest = max(all_faces, key=lambda r: r[2] * r[3])
                faces = [largest]
            else:
                faces = []
            last_faces = faces
        else:
            faces = last_faces

        # ---- Process the single face ----
        if len(faces) > 0:
            x, y, w, h = faces[0]
            cx, cy = get_centroid(x, y, w, h)
            face_gray = gray[y:y + h, x:x + w]
            face_color = frame[y:y + h, x:x + w]

            # Name from LBPH
            name = recognize_face_lbph(face_gray)

            # Emotion prediction (every FRAME_SKIP_EMOTION frames)
            emotion_counter += 1
            if emotion_counter % FRAME_SKIP_EMOTION == 0:
                emotion, conf = predict_emotion(face_color)
                last_emotion = emotion
                last_conf = conf
            else:
                emotion, conf = last_emotion, last_conf

            # ---- Tracking for auto-save ----
            if face_timer["active"] and face_timer["centroid"]:
                dist = distance((cx, cy), face_timer["centroid"])
                if dist <= FACE_MATCH_TOLERANCE:
                    face_timer["last_seen"] = current_time
                    elapsed = current_time - face_timer["start_time"]
                    if elapsed >= SAVE_THRESHOLD_SEC and not face_timer["saved"]:
                        save_face_image(face_color, name, emotion, conf, int(current_time))
                        face_timer["saved"] = True
                else:
                    face_timer = {
                        "active": True,
                        "start_time": current_time,
                        "last_seen": current_time,
                        "saved": False,
                        "centroid": (cx, cy)
                    }
            else:
                face_timer = {
                    "active": True,
                    "start_time": current_time,
                    "last_seen": current_time,
                    "saved": False,
                    "centroid": (cx, cy)
                }

            # ---- Draw bounding box and BIG text ----
            # Ensure coordinates are within frame
            x, y = max(0, x), max(0, y)
            w, h = min(frame.shape[1] - x, w), min(frame.shape[0] - y, h)
            if w > 0 and h > 0:
                # 1. Thick bounding box with emotion color
                box_color = get_emotion_color(emotion)
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, thickness=4)

                # 2. Label text: name, emotion, confidence
                label_text = f"{name}: {emotion} ({conf:.2f})"

                # 3. Large, clear font
                font = cv2.FONT_HERSHEY_DUPLEX   # cleaner than SIMPLEX
                font_scale = 0.9                 # big and readable
                thickness = 3                    # bold
                (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

                # 4. Position the text – try above the face, else below
                text_x = x
                text_y = y - 10 if y - 10 > text_h else y + h + 10
                if text_y + text_h > frame.shape[0]:
                    text_y = y - 10 if y - 10 > text_h else y + h - 5

                # 5. Black background rectangle for readability
                cv2.rectangle(frame,
                              (text_x, text_y - text_h - 5),
                              (text_x + text_w, text_y + 5),
                              (0, 0, 0), cv2.FILLED)

                # 6. White text
                cv2.putText(frame, label_text, (text_x, text_y),
                            font, font_scale, (255, 255, 255), thickness)

        # ---- Show the frame ----
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        current_time = time.time()

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Webcam closed.")

if __name__ == "__main__":
    main()