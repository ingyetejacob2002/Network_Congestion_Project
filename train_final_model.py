
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "network_congestion_labeled.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    "packet_count",
    "byte_count",
    "duration_total_sec",
    "log_packet_count",
    "log_byte_count",
    "packet_rate",
    "byte_rate",
    "bytes_per_packet",
]


def prepare_features(df):
    packets = pd.to_numeric(
        df["packet_count"], errors="coerce"
    ).clip(lower=0)

    bytes_count = pd.to_numeric(
        df["byte_count"], errors="coerce"
    ).clip(lower=0)

    seconds = pd.to_numeric(
        df["duration_sec"], errors="coerce"
    ).fillna(0)

    nanoseconds = pd.to_numeric(
        df["duration_nsec"], errors="coerce"
    ).fillna(0)

    duration = (seconds + nanoseconds / 1e9).clip(lower=1e-9)

    X = pd.DataFrame({
        "packet_count": packets,
        "byte_count": bytes_count,
        "duration_total_sec": duration,
        "log_packet_count": np.log1p(packets),
        "log_byte_count": np.log1p(bytes_count),
        "packet_rate": packets / duration,
        "byte_rate": bytes_count / duration,
        "bytes_per_packet": bytes_count / packets.clip(lower=1),
    })

    return X[FEATURES].replace([np.inf, -np.inf], np.nan)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required = [
        "packet_count",
        "byte_count",
        "duration_sec",
        "duration_nsec",
        "congestion_level",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=[
        "packet_count", "byte_count", "congestion_level"
    ]).copy()

    X = prepare_features(df)
    y = df["congestion_level"].astype(str).str.strip()

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    model.fit(X, y)

    model_path = MODEL_DIR / "final_congestion_model.joblib"
    joblib.dump(model, model_path)

    metadata = {
        "features": FEATURES,
        "classes": sorted(y.unique().tolist()),
        "training_rows": int(len(X)),
        "label_warning": (
            "Labels are scenario-derived. Predictions are not independently "
            "validated real-world congestion measurements."
        ),
    }

    with open(
        MODEL_DIR / "model_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    print("MODEL TRAINING COMPLETED")
    print("-" * 45)
    print(f"Training records: {len(X)}")
    print(f"Classes: {sorted(y.unique())}")
    print(f"Model saved to: {model_path}")
    print("This model predicts the existing assigned scenario labels.")


if __name__ == "__main__":
    main()
