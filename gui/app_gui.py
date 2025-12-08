import cv2
import time
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from openai import OpenAI
import joblib
import os
from dotenv import load_dotenv

# ===========================
#         CONFIG
# ===========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Colors por emoción
EMO_COLORS = {
    "alegria": (0, 255, 255),
    "tristeza": (255, 0, 0),
    "ansiedad": (0, 140, 255),
    "neutral": (200, 200, 200)
}

EMOJI = {
    "alegria": "😊",
    "tristeza": "😢",
    "ansiedad": "😰",
    "neutral": "😐"
}

# Mediapipe
mp_pose = mp.solutions.pose
drawer = mp.solutions.drawing_utils


# ===========================
#   CARGA MODELOS
# ===========================

model_path = os.path.join(BASE_DIR, "ml/models/emotion_model.pkl")
scaler_path = os.path.join(BASE_DIR, "ml/models/scaler.pkl")
encoder_path = os.path.join(BASE_DIR, "ml/models/label_encoder.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
encoder = joblib.load(encoder_path)


# ===========================
#   FEATURE EXTRACTION
# ===========================

def extract_features(landmarks):
    xs = np.array([lm.x for lm in landmarks])
    ys = np.array([lm.y for lm in landmarks])
    dx = np.diff(xs)
    dy = np.diff(ys)
    vel = float(np.sqrt(dx**2 + dy**2).mean())
    stab = float(np.abs(dx).mean() + np.abs(dy).mean())

    idx = [11, 12, 23, 24]
    width = float(xs[idx].max() - xs[idx].min())
    height = float(ys[idx].max() - ys[idx].min())
    exp_val = width * height
    return np.array([vel, exp_val, stab])


def get_suggestion(emotion):
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres empático, motivador y muy breve."},
                {"role": "user", "content": f"La emoción detectada es {emotion}. Dame un consejo corto."}
            ]
        )
        return r.choices[0].message.content.strip()
    except:
        return "(No disponible)"


# ===========================
#            GUI
# ===========================

class EmotionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Emotion AI Suite (v2.0)")
        self.root.configure(bg="#1e1e1e")
        self.root.geometry("1600x900")

        self.mode = "infer"

        # cámara
        self.cap = cv2.VideoCapture(0)

        # mediapipe
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            smooth_landmarks=True,
            min_detection_confidence=0.5
        )

        # historial sesión
        self.history = []

        # entrenamiento
        self.is_recording = False
        self.record_data = []
        self.record_start_time = 0
        self.countdown = 3

        self.build_ui()
        self.update_frame()

    # ===========================
    #       CONSTRUCCIÓN UI
    # ===========================

    def build_ui(self):
        # Panel principal
        self.left_frame = tk.Frame(self.root, bg="#1e1e1e")
        self.left_frame.pack(side="left", fill="both", expand=True)

        self.right_frame = tk.Frame(self.root, width=450, bg="#1e1e1e")
        self.right_frame.pack(side="right", fill="y")

        # ------- Video ------
        self.label_video = tk.Label(self.left_frame, bg="#1e1e1e")
        self.label_video.pack(expand=True)

        # ------- Panel derecho ------
        title = tk.Label(self.right_frame, text="EMOTION AI SUITE", font=("Arial", 22, "bold"),
                         fg="white", bg="#1e1e1e")
        title.pack(pady=10)

        self.label_emoji = tk.Label(self.right_frame, text="😐", font=("Arial", 60),
                                    fg="white", bg="#1e1e1e")
        self.label_emoji.pack()

        self.label_emotion = tk.Label(self.right_frame, text="Emoción: ---", font=("Arial", 16),
                                      fg="white", bg="#1e1e1e")
        self.label_emotion.pack(pady=5)

        self.label_suggestion = tk.Label(self.right_frame, text="---", wraplength=350,
                                         justify="center", font=("Arial", 12), fg="#cccccc", bg="#1e1e1e")
        self.label_suggestion.pack(pady=5)

        # Botón pantalla completa
        tk.Button(self.right_frame, text="Pantalla completa (F11)",
                  command=self.toggle_fullscreen, bg="#444", fg="white").pack(pady=5)

        # ----- gráfica -----
        fig = plt.Figure(figsize=(4, 2.5), facecolor="#1e1e1e")
        self.ax = fig.add_subplot(111)
        self.ax.set_facecolor("#1e1e1e")
        self.ax.tick_params(colors="white")
        self.ax.spines["bottom"].set_color("white")
        self.ax.spines["left"].set_color("white")

        self.canvas = FigureCanvasTkAgg(fig, master=self.right_frame)
        self.canvas.get_tk_widget().pack()

        # ----- historial -----
        tk.Label(self.right_frame, text="Historial de Sesión", font=("Arial", 14, "bold"),
                 fg="white", bg="#1e1e1e").pack(pady=5)

        self.tree = ttk.Treeview(self.right_frame, columns=("time", "emo"), show="headings", height=8)
        self.tree.heading("time", text="Tiempo")
        self.tree.heading("emo", text="Emoción")
        self.tree.pack(fill="x", padx=10)

        # ---- PANEL DE ENTRENAMIENTO ----
        self.train_panel = tk.Frame(self.right_frame, bg="#1e1e1e")

        tk.Label(self.train_panel, text="Entrenamiento", font=("Arial", 16, "bold"),
                 fg="white", bg="#1e1e1e").pack(pady=5)

        self.combo_emo = ttk.Combobox(self.train_panel,
                                      values=["alegria", "tristeza", "ansiedad", "neutral"],
                                      state="readonly")
        self.combo_emo.pack(pady=5)

        self.bt_record = tk.Button(self.train_panel, text="Grabar", command=self.start_record,
                                   bg="#555", fg="white")
        self.bt_record.pack(pady=5)

        self.timer_label = tk.Label(self.train_panel, text="", font=("Arial", 14),
                                    fg="white", bg="#1e1e1e")
        self.timer_label.pack()

        # botón modo entrenamiento
        tk.Button(self.right_frame, text="Modo Entrenamiento", command=lambda: self.switch_mode("train"),
                  bg="#444", fg="white").pack(pady=5)



    # ===========================
    #        MODO
    # ===========================

    def switch_mode(self, mode):
        self.mode = mode

        if mode == "train":
            # ocultar inferencia
            self.label_emoji.pack_forget()
            self.label_emotion.pack_forget()
            self.label_suggestion.pack_forget()
            self.canvas.get_tk_widget().pack_forget()
            self.tree.pack_forget()

            # mostrar entrenamiento
            self.train_panel.pack(pady=10)

        else:
            # mostrar inferencia
            self.label_emoji.pack()
            self.label_emotion.pack(pady=5)
            self.label_suggestion.pack(pady=5)
            self.canvas.get_tk_widget().pack()
            self.tree.pack(fill="x", padx=10)

            # ocultar entrenamiento
            self.train_panel.pack_forget()


    # ===========================
    #   INICIO DE GRABACIÓN
    # ===========================

    def start_record(self):
        if not self.combo_emo.get():
            self.timer_label.config(text="Selecciona una emoción")
            return

        self.record_data = []
        self.is_recording = False
        self.countdown = 3
        self.timer_label.config(text=f"Comenzando en {self.countdown}...")
        self.root.after(1000, self.countdown_record)


    def countdown_record(self):
        self.countdown -= 1
        if self.countdown > 0:
            self.timer_label.config(text=f"Comenzando en {self.countdown}...")
            self.root.after(1000, self.countdown_record)
        else:
            self.timer_label.config(text="Grabando...")
            self.is_recording = True
            self.record_start_time = time.time()


    def stop_record(self):
        self.is_recording = False

        emotion = self.combo_emo.get()
        out_path = os.path.join(BASE_DIR, "data/raw_sessions",
                                f"{emotion}_{int(time.time())}.csv")

        import pandas as pd
        df = pd.DataFrame(self.record_data)
        df.to_csv(out_path, index=False)

        self.timer_label.config(text=f"Guardado: {out_path}")


    # ===========================
    #     LOOP VIDEO
    # ===========================

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_frame)
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        # ------------------- ENTRENAMIENTO -------------------
        if self.mode == "train":
            if results.pose_landmarks:
                drawer.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                if self.is_recording:
                    row = []
                    for lm in results.pose_landmarks.landmark:
                        row += [lm.x, lm.y, lm.z]
                    self.record_data.append(row)

                    elapsed = time.time() - self.record_start_time
                    self.timer_label.config(text=f"Grabando... {elapsed:.1f}s")

                    # detener automáticamente después de 6 segundos
                    if elapsed >= 6:
                        self.stop_record()

        # ------------------ INFERENCIA ------------------------
        elif self.mode == "infer" and results.pose_landmarks:
            feats = extract_features(results.pose_landmarks.landmark)
            Xs = scaler.transform([feats])
            emotion = encoder.inverse_transform(model.predict(Xs))[0]

            # UI
            self.label_emoji.config(text=EMOJI.get(emotion, "😐"))
            self.label_emotion.config(text=f"Emoción: {emotion}")

            suggestion = get_suggestion(emotion)
            self.label_suggestion.config(text=suggestion)

            # historial
            t = time.strftime("%H:%M:%S")
            self.tree.insert("", "end", values=(t, emotion))

            # gráfica
            self.history.append(emotion)
            self.update_plot()

            # contorno dinámico
            color = EMO_COLORS.get(emotion, (200, 200, 200))
            frame = cv2.copyMakeBorder(frame, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=color)

        # ------------------- RENDER VIDEO -------------------
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.label_video.imgtk = imgtk
        self.label_video.configure(image=imgtk)

        self.root.after(10, self.update_frame)


    # ===========================
    #    GRÁFICA TIEMPO REAL
    # ===========================

    def update_plot(self):
        emomap = {"alegria": 3, "neutral": 2, "ansiedad": 1, "tristeza": 0}
        ys = [emomap[e] for e in self.history[-30:]]

        self.ax.clear()
        self.ax.plot(ys, color="#00ffff")
        self.ax.set_ylim(-1, 4)
        self.ax.set_facecolor("#1e1e1e")
        self.canvas.draw()


    # ===========================
    #   PANTALLA COMPLETA
    # ===========================

    def toggle_fullscreen(self):
        self.root.attributes("-fullscreen",
                             not self.root.attributes("-fullscreen"))


    # ===========================
    #         MAIN LOOP
    # ===========================

    def run(self):
        self.root.mainloop()


# ===========================
#        EJECUCIÓN
# ===========================

if __name__ == "__main__":
    EmotionGUI().run()
