import os
import pandas as pd
from openai import OpenAI

client = OpenAI()

def load_csv(path):
    return pd.read_csv(path)

def clean_for_prompt(df):
    cols = ["mean", "std", "min", "max"]
    return df.describe().transpose()[cols].to_string()

def generate_summary(csv_path, reports_dir):
    df = load_csv(csv_path)
    clip_id = os.path.splitext(os.path.basename(csv_path))[0]
    os.makedirs(reports_dir, exist_ok=True)
    out_path = os.path.join(reports_dir, f"{clip_id}.md")

    stats = clean_for_prompt(df)

    prompt = f"""
    Resume el movimiento humano detectado en este clip.
    Describe ritmo, estabilidad, amplitud y energía, sin lenguaje médico.
    Estadísticas:
    {stats}
    """

    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Eres un analista de movimiento humano."},
            {"role": "user", "content": prompt}
        ]
    )

    text = r.choices[0].message.content.strip()
    with open(out_path, "w") as f:
        f.write(text)

    print(f"Resumen generado: {out_path}")

def run():
    base = os.path.join(os.path.dirname(__file__), "..", "data")
    raw = os.path.join(base, "raw_sessions")
    reports = os.path.join(base, "reports")

    csvs = [f for f in os.listdir(raw) if f.endswith(".csv")]
    for c in csvs:
        generate_summary(os.path.join(raw, c), reports)

if __name__ == "__main__":
    run()
