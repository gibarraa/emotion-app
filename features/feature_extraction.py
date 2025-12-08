import os
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull

def calculate_velocity(df):
    joints = [col for col in df.columns if col.startswith("x_")]
    n_joints = len(joints)
    velocities = []

    for i in range(0, n_joints):
        x = df[f"x_{i}"].values
        y = df[f"y_{i}"].values
        dx = np.diff(x)
        dy = np.diff(y)
        v = np.sqrt(dx**2 + dy**2)
        velocities.append(np.nanmean(v))
    return np.nanmean(velocities)

def calculate_body_expansion(df):
    points = []
    for i in [11, 12, 23, 24, 15, 16, 27, 28]:  # hombros, caderas, muñecas, rodillas
        if f"x_{i}" in df.columns and f"y_{i}" in df.columns:
            points.append([df[f"x_{i}"].mean(), df[f"y_{i}"].mean()])
    points = np.array(points)
    if len(points) >= 3:
        hull = ConvexHull(points)
        return hull.volume
    return np.nan

def calculate_stability(df):
    positions = df[[c for c in df.columns if c.startswith("x_") or c.startswith("y_")]]
    diffs = positions.diff().abs().mean().mean()
    return float(diffs)

def extract_features_from_csv(csv_path, reports_dir):
    df = pd.read_csv(csv_path)
    clip_id = os.path.splitext(os.path.basename(csv_path))[0]
    emotion = clip_id.split("_")[0]
    report_path = os.path.join(reports_dir, f"{clip_id}.md")

    mean_velocity = calculate_velocity(df)
    expansion = calculate_body_expansion(df)
    stability = calculate_stability(df)

    summary_text = ""
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            summary_text = f.read().strip()

    features = {
        "clip_id": clip_id,
        "emotion": emotion,
        "mean_velocity": mean_velocity,
        "expansion": expansion,
        "stability": stability,
        "summary": summary_text
    }

    return features

def process_all_sessions():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    raw_dir = os.path.join(base_dir, "raw_sessions")
    reports_dir = os.path.join(base_dir, "reports")
    features_dir = os.path.join(base_dir, "features")
    os.makedirs(features_dir, exist_ok=True)

    csv_files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
    all_features = []

    for csv in csv_files:
        features = extract_features_from_csv(os.path.join(raw_dir, csv), reports_dir)
        all_features.append(features)

    df_features = pd.DataFrame(all_features)
    out_path = os.path.join(features_dir, "dataset_features.csv")
    df_features.to_csv(out_path, index=False)
    print(f"✅ Dataset de características generado: {out_path}")

if __name__ == "__main__":
    process_all_sessions()
