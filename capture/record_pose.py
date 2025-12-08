import cv2
import mediapipe as mp
import pandas as pd
import time
import os

SAVE_DIR = "data/raw_sessions"
os.makedirs(SAVE_DIR, exist_ok=True)

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

def record_session(emotion, seconds=8):
    cap = cv2.VideoCapture(0)
    keypoints_list = []

    print(f"Grabando emoción '{emotion}' por {seconds} segundos...")
    start = time.time()

    while time.time() - start < seconds:
        ret, frame = cap.read()
        if not ret:
            print("Error al leer cámara.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            kp = []
            for lm in results.pose_landmarks.landmark:
                kp.extend([lm.x, lm.y, lm.z])
            keypoints_list.append(kp)

            # DIBUJAR ESQUELETO EN LA IMAGEN
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                connection_drawing_spec=mp_draw.DrawingSpec(color=(255,0,0), thickness=2)
            )

        cv2.imshow("Grabando...", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{emotion}_{ts}.csv"
    path = os.path.join(SAVE_DIR, filename)

    df = pd.DataFrame(keypoints_list)
    df.to_csv(path, index=False)

    print(f"Guardado → {path}")


if __name__ == "__main__":
    emo = input("Emoción a grabar (alegria, ansiedad, tristeza, neutral): ")
    record_session(emo.strip())
