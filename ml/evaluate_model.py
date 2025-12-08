import os
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics import classification_report, confusion_matrix
import joblib

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_text(texts):
    embeddings = client.embeddings.create(
        model="text-embedding-3-small",
        input=list(texts)
    )
    return np.array([e.embedding for e in embeddings.data])

def load_artifacts():
    base = os.path.join(os.path.dirname(__file__), "models")
    model = joblib.load(os.path.join(base, "emotion_model.pkl"))
    scaler = joblib.load(os.path.join(base, "scaler.pkl"))
    encoder = joblib.load(os.path.join(base, "label_encoder.pkl"))
    numeric_cols = joblib.load(os.path.join(base, "feature_cols.pkl"))
    return model, scaler, encoder, numeric_cols

def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "features", "dataset_features.csv")
    return pd.read_csv(path)

def evaluate():
    df = load_dataset()
    model, scaler, encoder, numeric_cols = load_artifacts()
    df = df.dropna(subset=["summary"])
    emb = embed_text(df["summary"])
    X_num = df[numeric_cols].values
    X = np.hstack([X_num, emb])
    X_scaled = scaler.transform(X)
    y_true = encoder.transform(df["emotion"])
    y_pred = model.predict(X_scaled)
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    evaluate()
