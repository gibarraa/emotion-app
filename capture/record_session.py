import cv2
import mediapipe as mp
import pandas as pd
import time
import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

emotions = ["alegria", "tristeza", "ansiedad", "neutral"]

def create_directories():
    base_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw_sessions")
    os.makedirs(base_path, exist_ok=True)
    return base_path

def record_session(emotion, duration=20):
    base_path = create_directories()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    clip_id = f"{emotion}_{timestamp}"
    csv_path = os.path.join(base_path, f"{clip_id}.csv")

    cap = cv2.VideoCapture(0)
    pose = mp_pose.Pose()
    columns = ["frame", "timestamp"]
    for i in range(33):
        columns += [f"x_{i}", f"y_{i}", f"z_{i}", f"v_{i}"]
    df = pd.DataFrame(columns=columns)

    start_time = time.time()
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        row = [frame_id, time.time() - start_time]
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                row += [lm.x, lm.y, lm.z, lm.visibility]
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        else:
            row += [None] * (33 * 4)

        df.loc[len(df)] = row
        frame_id += 1

        cv2.putText(frame, f"Grabando: {emotion.upper()}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elapsed = int(time.time() - start_time)
        cv2.putText(frame, f"Tiempo: {elapsed}s / {duration}s", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.imshow("Grabando sesion", frame)

        if elapsed >= duration or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    df.to_csv(csv_path, index=False)
    print(f"Sesion guardada: {csv_path}")

if __name__ == "__main__":
    print("Selecciona la emocion a grabar:")
    for i, emo in enumerate(emotions):
        print(f"{i + 1}) {emo}")
    choice = int(input("Numero: ")) - 1
    emotion = emotions[choice]
    record_session(emotion)
