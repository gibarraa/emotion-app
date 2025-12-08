import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

df = pd.read_csv("data/features/dataset_features.csv")

X = df[["mean_velocity", "expansion", "stability"]].values
y = df["emotion"].values

encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train_scaled, y_train)

models_dir = "ml/models"
os.makedirs(models_dir, exist_ok=True)

joblib.dump(model, os.path.join(models_dir, "emotion_model.pkl"))
joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
joblib.dump(encoder, os.path.join(models_dir, "label_encoder.pkl"))
joblib.dump(["mean_velocity", "expansion", "stability"], os.path.join(models_dir, "feature_cols.pkl"))

print("Modelo entrenado y guardado correctamente.")
