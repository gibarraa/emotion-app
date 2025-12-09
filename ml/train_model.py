import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw_sessions")
MODEL_DIR = os.path.join(BASE_DIR, "ml", "models")

print("== ENTRENANDO MODELO CON 99 FEATURES ==")

files = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]

X = []
y = []

for filename in files:
    emotion = filename.split("_")[0]  # alegria_xxx.csv -> alegria
    path = os.path.join(DATA_PATH, filename)

    # Cargar CSV como DataFrame (soporta encabezados)
    df = pd.read_csv(path)

    # Detectar si sobran columnas no numéricas
    df = df.select_dtypes(include=[float, int])

    if df.shape[1] != 99:
        print("❌ Archivo ignorado (columnas inválidas):", filename)
        print("   Columnas encontradas:", df.shape[1])
        continue

    # Vector promedio para esta muestra
    features = df.mean(axis=0).values
    X.append(features)
    y.append(emotion)

print("Muestras válidas:", len(X))

X = np.array(X)
y = np.array(y)

# Codificar etiquetas
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entrenar modelo
model = RandomForestClassifier(n_estimators=200)
model.fit(X_scaled, y_encoded)

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, os.path.join(MODEL_DIR, "emotion_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

print("== MODELO ENTRENADO Y GUARDADO ==")
