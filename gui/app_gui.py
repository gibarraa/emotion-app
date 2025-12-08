import cv2
import time
import numpy as np
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
from openai import OpenAI

import joblib
import os

from dotenv import load_dotenv

# Cargar .env correctamente desde Analisis_E
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))


# ===========================
#  CARGA DE MODELOS Y CONFIG
# ===========================

model_path = os.path.join(BASE_DIR, "ml/models/emotion_model.pkl")
scaler_path = os.path.join(BASE_DIR, "ml/models/scaler.pkl")
encoder_path = os.path.join(BASE_DIR, "ml/models/label_encoder.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
encoder = joblib.load(encoder_path)

# Cliente OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ===========================
#   FUNCIONES DEL LLM
# ===========================
def get_suggestion(emotion):
    """Genera una sugerencia usando ChatGPT basado en la emoción detectada."""
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Eres un asistente que da recomendaciones breves, cálidas y motivadoras según la emoción."
                },
                {
                    "role": "user",
                    "content": f"La emoción detectada es: {emotion}. Da una recomendación muy breve."
                }
            ],
            max_tokens=40,
            temperature=0.6
        )

        return r.choices[0].message.content.strip()

    except Exception as e:
        return f"(Error LLM: {e})"


# ===========================
#   INTERFAZ GRÁFICA (GUI)
# ===========================
class EmotionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Detección de Emociones con Pose + IA")

        # Cámara
        self.cap = cv2.VideoCapture(0)

        # Mediapipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

        # GUI
        self.label_video = tk.Label(self.root)
        self.label_video.pack()

        self.label_emotion = tk.Label(self.root, text="Emoción detectada: ---", font=("Arial", 16))
        self.label_emotion.pack(pady=5)

        self.label_suggestion = tk.Label(self.root, text="Sugerencia: ---", font=("Arial", 14))
        self.label_suggestion.pack(pady=5)

        # Rate limit para el LLM (una sugerencia cada 5s)
        self.last_suggestion_time = 0
        self.last_suggestion = "---"

        self.update_frame()

    # ===========================
    #        LOOP DE VIDEO
    # ===========================
    def update_frame(self):
        ret, frame = self.cap.read()

        if not ret:
            self.root.after(10, self.update_frame)
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pose
        results = self.pose.process(frame_rgb)

        if results.pose_landmarks:
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

            X = np.array(keypoints).reshape(1, -1)
            X_scaled = scaler.transform(X)

            pred = model.predict(X_scaled)
            emotion = encoder.inverse_transform(pred)[0]

            self.label_emotion.config(text=f"Emoción detectada: {emotion}")

            # Sugerencias cada 5s
            now = time.time()
            if now - self.last_suggestion_time > 5:
                self.last_suggestion_time = now
                self.last_suggestion = get_suggestion(emotion)

            self.label_suggestion.config(text=f"Sugerencia: {self.last_suggestion}")

        # Convertir imagen a Tkinter
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.label_video.imgtk = imgtk
        self.label_video.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    # ===========================
    #       MAIN LOOP
    # ===========================
    def run(self):
        self.root.mainloop()


# ===========================
#        EJECUCIÓN
# ===========================
if __name__ == "__main__":
    EmotionGUI().run()
