import pandas as pd
import numpy as np
import os

RAW_DIR = "data/raw_sessions"
OUT_CSV = "data/features/dataset_features.csv"

def compute_features(df):
    """
    df: DataFrame con columnas '0'...'98'
    Asumimos: cada 3 columnas = (x, y, z) de un landmark.
    """

    # Convertir a numpy
    arr = df.values  # shape: (frames, 99)

    # Calcular velocidad promedio
    diffs = np.diff(arr, axis=0)   # diferencia frame a frame → (N-1, 99)
    vels = np.linalg.norm(diffs, axis=1)  # norma de cada cambio → (N-1,)
    mean_vel = float(np.mean(vels))

    # Expansion = varianza promedio de los puntos
    expansion = float(np.mean(np.var(arr, axis=0)))

    # Stability = inverso del cambio promedio
    stability = float(1.0 / (np.mean(vels) + 1e-6))

    return mean_vel, expansion, stability


def build_dataset():
    rows = []

    for filename in os.listdir(RAW_DIR):
        if not filename.endswith(".csv"):
            continue

        emotion = filename.split("_")[0]  # alegria_....csv → alegria
        csv_path = os.path.join(RAW_DIR, filename)

        df = pd.read_csv(csv_path)

        mean_vel, expansion, stability = compute_features(df)

        rows.append({
            "clip_id": filename.replace(".csv", ""),
            "emotion": emotion,
            "mean_velocity": mean_vel,
            "expansion": expansion,
            "stability": stability
        })

    # Guardar dataset
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    print("Dataset generado en:", OUT_CSV)
    print(out_df)


if __name__ == "__main__":
    build_dataset()
