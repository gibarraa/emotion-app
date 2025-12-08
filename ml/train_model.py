import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Ruta del dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data/features/dataset_features.csv")

df = pd.read_csv(DATA_PATH)

print("Columnas encontradas:", df.columns.tolist())

# ==========================
#   DEFINIR FEATURES
# ==========================
FEATURE_COLS = ["mean_velocity", "expansion", "stability"]
TARGET_COL = "emotion"

X = df[FEATURE_COLS]
y = df[TARGET_COL]

# ==========================
#   ENCODING Y ESCALADO
# ==========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# ==========================
#   ENTRENAMIENTO
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=42
)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# ==========================
#   EVALUACIÓN
# ==========================
preds = clf.predict(X_test)
print("\n==== REPORT ====\n")
print(classification_report(y_test, preds, target_names=encoder.classes_))

# ==========================
#   GUARDAR MODELOS
# ==========================
MODELS_DIR = os.path.join(BASE_DIR, "ml/models")

joblib.dump(clf, os.path.join(MODELS_DIR, "emotion_model.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
joblib.dump(encoder, os.path.join(MODELS_DIR, "label_encoder.pkl"))

# También guardamos la lista de columnas
joblib.dump(FEATURE_COLS, os.path.join(MODELS_DIR, "feature_cols.pkl"))

print("\nModelos guardados en:", MODELS_DIR)
