import cv2
import time
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from openai import OpenAI
import joblib
import os
from dotenv import load_dotenv
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ===========================
# LOAD ENV + MODELS
# ===========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

model = joblib.load(os.path.join(BASE_DIR, "ml/models/emotion_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "ml/models/scaler.pkl"))
encoder = joblib.load(os.path.join(BASE_DIR, "ml/models/label_encoder.pkl"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
drawer = mp.solutions.drawing_utils


# ===========================
# EMOJIS
# ===========================

EMOJI = {
    "alegria": "😄",
    "tristeza": "😢",
    "ansiedad": "😰",
    "neutral": "😐",
}


# ===========================
# GENERAL COLORS – DARK MODE
# ===========================

COLORS = {
    "bg": "#0E0E0F",        # fondo principal
    "panel": "#151515",     # panel derecho
    "text": "#FFFFFF",
    "accent": "#5AC8FA",
}


# ===========================
# SUGGESTER
# ===========================

def get_suggestion(emotion):
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Responde cálido y breve."},
                {"role": "user", "content": f"Recomienda algo breve para la emoción '{emotion}'."}
            ]
        )
        return r.choices[0].message.content.strip()
    except:
        return "…"


# ===========================
# MAIN CLASS
# ===========================

class EmotionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Emotion AI – Dark UI")
        self.root.geometry("1280x720")
        self.root.configure(bg=COLORS["bg"])

        # Capture
        self.cap = cv2.VideoCapture(0)

        # State
        self.mode = "dashboard"
        self.emotion_history = deque(maxlen=50)

        # Layout containers
        self.build_layout()

        # First frame update
        self.update_frame()


    # ======================
    # LAYOUT: CAMERA + PANEL
    # ======================

    def build_layout(self):
        for w in self.root.winfo_children():
            w.destroy()

        # LEFT SIDE → CAMERA
        self.camera_frame = tk.Frame(self.root, bg="black")
        self.camera_frame.pack(side="left", fill="both", expand=True)

        self.video_label = tk.Label(self.camera_frame, bg="black")
        self.video_label.pack(fill="both", expand=True)

        # RIGHT SIDE → PANEL
        self.ui_panel = tk.Frame(self.root, bg=COLORS["panel"], width=380)
        self.ui_panel.pack(side="right", fill="y")
        self.ui_panel.pack_propagate(False)

        self.build_dashboard_panel()


    # ======================
    # DASHBOARD PANEL UI
    # ======================

    def build_dashboard_panel(self):

        # EMOJI
        self.emoji_label = tk.Label(
            self.ui_panel, text="😐",
            font=("SF Pro Display", 70),
            bg=COLORS["panel"], fg="white"
        )
        self.emoji_label.pack(pady=10)

        # Emotion text
        self.emotion_label = tk.Label(
            self.ui_panel, text="Emoción: ---",
            font=("SF Pro Display", 24, "bold"),
            bg=COLORS["panel"], fg="white"
        )
        self.emotion_label.pack(pady=5)

        # Suggestion
        self.suggestion_label = tk.Label(
            self.ui_panel,
            text="Sugerencia: ---",
            wraplength=300,
            justify="center",
            font=("SF Pro Display", 14),
            bg=COLORS["panel"], fg="#CCCCCC"
        )
        self.suggestion_label.pack(pady=10)

        # TRAINING MODE BUTTON
        train_btn = tk.Button(
            self.ui_panel,
            text="Modo Entrenamiento",
            command=self.switch_training,
            bg="#222222", fg="white",
            relief="flat", padx=15, pady=10
        )
        train_btn.pack(pady=10)

        # ======== HISTORICAL GRAPH ========
        fig = plt.Figure(figsize=(3.5, 2.2), dpi=80)
        fig.patch.set_facecolor(COLORS["panel"])
        self.ax = fig.add_subplot(111)
        self.ax.set_facecolor(COLORS["panel"])
        self.ax.tick_params(colors="white")
        self.ax.title.set_color("white")
        self.ax.set_title("Historial de emociones")

        self.canvas = FigureCanvasTkAgg(fig, master=self.ui_panel)
        self.canvas.get_tk_widget().pack(pady=20)


    # ======================
    # TRAINING MODE PANEL UI
    # ======================

    def switch_training(self):
        self.mode = "training"
        for w in self.ui_panel.winfo_children():
            w.destroy()

        # TITLE
        tk.Label(
            self.ui_panel,
            text="Modo Entrenamiento",
            font=("SF Pro Display", 22, "bold"),
            bg=COLORS["panel"], fg="white"
        ).pack(pady=10)

        # Select emotion
        self.train_emotion = tk.StringVar(value="alegria")
        combo = ttk.Combobox(
            self.ui_panel,
            values=["alegria", "tristeza", "ansiedad", "neutral"],
            textvariable=self.train_emotion,
            width=15
        )
        combo.pack(pady=5)

        # Timer
        self.timer_label = tk.Label(
            self.ui_panel, text="00:00",
            font=("SF Pro Display", 30),
            bg=COLORS["panel"], fg="white"
        )
        self.timer_label.pack(pady=10)

        # Record button
        btn = tk.Button(
            self.ui_panel, text="Grabar muestra",
            command=self.start_recording,
            bg="#333333", fg="white",
            relief="flat", padx=15, pady=10
        )
        btn.pack(pady=10)

        # Back button
        back = tk.Button(
            self.ui_panel, text="Volver al Dashboard",
            command=self.back_dashboard,
            bg="#222222", fg="white",
            relief="flat"
        )
        back.pack(pady=20)

        self.is_recording = False
        self.record_start = 0
        self.record_data = []


    # ======================
    def back_dashboard(self):
        self.mode = "dashboard"
        self.build_layout()


    # ======================
    # RECORDING LOGIC
    # ======================

    def start_recording(self):
        self.is_recording = True
        self.record_start = time.time()
        self.record_data = []


    # ======================
    # FRAME LOOP
    # ======================

    def update_frame(self):

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(20, self.update_frame)
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        # ======================
        # TRAINING MODE
        # ======================

        if self.mode == "training" and results.pose_landmarks:

            drawer.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            pts = []
            for lm in results.pose_landmarks.landmark:
                pts.extend([lm.x, lm.y, lm.z])

            if self.is_recording:
                self.record_data.append(pts)
                elapsed = int(time.time() - self.record_start)
                self.timer_label.config(text=f"{elapsed//60:02d}:{elapsed%60:02d}")

                if elapsed >= 5:
                    self.is_recording = False
                    emotion = self.train_emotion.get()
                    filename = f"data/raw_sessions/{emotion}_{int(time.time())}.csv"
                    np.savetxt(filename, np.array(self.record_data), delimiter=",")
                    print("Guardado:", filename)


        # ======================
        # DASHBOARD MODE
        # ======================

        if self.mode == "dashboard" and results.pose_landmarks:

            pts = []
            for lm in results.pose_landmarks.landmark:
                pts.extend([lm.x, lm.y, lm.z])

            X = np.array(pts).reshape(1, -1)
            emotion = encoder.inverse_transform(model.predict(scaler.transform(X)))[0]

            # UI updates
            self.emoji_label.config(text=EMOJI.get(emotion, "😐"))
            self.emotion_label.config(text=f"Emoción: {emotion}")

            suggestion = get_suggestion(emotion)
            self.suggestion_label.config(text=suggestion)

            # History graph data (store emotion index)
            idx = ["alegria", "tristeza", "ansiedad", "neutral"].index(emotion)
            self.emotion_history.append(idx)

            self.update_graph()


        # ======================
        # Show camera (left pane)
        # ======================

        frame = cv2.resize(frame, (900, 720))
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(20, self.update_frame)


    # ======================
    # GRAPH UPDATE
    # ======================

    def update_graph(self):
        self.ax.clear()
        self.ax.set_facecolor(COLORS["panel"])
        self.ax.plot(list(self.emotion_history), color=COLORS["accent"], linewidth=2)
        self.ax.set_title("Historial de emociones", color="white")
        self.ax.tick_params(colors="white")
        self.canvas.draw()


    # ======================
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    EmotionGUI().run()
