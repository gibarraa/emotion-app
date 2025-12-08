import cv2
import mediapipe as mp
import numpy as np
import os
import joblib
import time
from openai import OpenAI

client = OpenAI()

mp_pose = mp.solutions.pose
drawer = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

models_base = os.path.join(os.path.dirname(__file__), "models")
model = joblib.load(os.path.join(models_base, "emotion_model.pkl"))
scaler = joblib.load(os.path.join(models_base, "scaler.pkl"))
encoder = joblib.load(os.path.join(models_base, "label_encoder.pkl"))
numeric_cols = joblib.load(os.path.join(models_base, "feature_cols.pkl"))

def extract_live_features(landmarks):
    xs = np.array([lm.x for lm in landmarks])
    ys = np.array([lm.y for lm in landmarks])

    dx = np.diff(xs)
    dy = np.diff(ys)
    vel = float(np.sqrt(dx**2 + dy**2).mean())

    idx = [11, 12, 23, 24, 15, 16, 27, 28]
    xs_sel = xs[idx]
    ys_sel = ys[idx]
    width = float(xs_sel.max() - xs_sel.min())
    height = float(ys_sel.max() - ys_sel.min())
    exp_val = max(width, 0.0) * max(height, 0.0)

    stab = float(np.abs(dx).mean() + np.abs(dy).mean())

    return np.array([vel, exp_val, stab])

def predict_emotion(features):
    X = scaler.transform([features])
    y = model.predict(X)
    return encoder.inverse_transform(y)[0]

def get_suggestion(emotion):
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Eres un asistente empático y breve."},
            {"role": "user", "content": f"La emoción detectada es {emotion}. Da una recomendación corta y útil."}
        ]
    )
    return r.choices[0].message.content.strip()

def run_inference():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    print("Cámara abierta. Presiona 'q' para salir.")

    last_emotion = ""
    last_suggestion = ""
    last_suggestion_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            drawer.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            feats = extract_live_features(results.pose_landmarks.landmark)
            emotion = predict_emotion(feats)

            now = time.time()
            if emotion != last_emotion or now - last_suggestion_time > 4:
                last_emotion = emotion
                last_suggestion = get_suggestion(emotion)
                last_suggestion_time = now

            cv2.putText(frame, f"Emoción: {last_emotion}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            y = 90
            for line in last_suggestion.split("."):
                line = line.strip()
                if line:
                    cv2.putText(frame, line, (30, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    y += 30

        cv2.imshow("EMOTION AI - Tiempo Real", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference()
