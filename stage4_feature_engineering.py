
"""
STAGE 4: FEATURE ENGINEERING AND NUMERICAL PREPARATION

Project:
Machine Learning-Based Prediction of Network Traffic
Congestion in Computer Networks

Input:
    prepared_data/train_data.csv
    prepared_data/test_data.csv

Output:
    stage4_results/
        engineered_train.csv
        engineered_test.csv
        scaled_train.csv
        scaled_test.csv
        feature_scaler.joblib
        feature_names.csv
        stage4_summary.txt
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. CONFIGURATION
# ============================================================

TRAIN_FILE = Path("prepared_data/train_data.csv")
TEST_FILE = Path("prepared_data/test_data.csv")

OUTPUT_DIR = Path("stage4_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "congestion_level"

REQUIRED_FEATURES = [
    "packet_count",
    "byte_count",
    "duration_sec",
    "duration_nsec",
]

ENGINEERED_FEATURES = [
    "log_packet_count",
    "log_byte_count",
    "duration_total_sec",
]


# ============================================================
# 2. LOAD THE PREPARED DATA
# ============================================================

print("=" * 70)
print("STAGE 4: FEATURE ENGINEERING")
print("=" * 70)

for file_path in [TRAIN_FILE, TEST_FILE]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}\n"
            "Run Stage 2 successfully before Stage 4."
        )

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("\n[1] DATA LOADED")
print("-" * 70)
print(f"Training records: {len(train_df):,}")
print(f"Testing records : {len(test_df):,}")


# ============================================================
# 3. VALIDATE THE REQUIRED COLUMNS
# ============================================================

for split_name, data in [
    ("Training", train_df),
    ("Testing", test_df),
]:
    missing = [
        column for column in REQUIRED_FEATURES + [TARGET]
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"{split_name} dataset is missing columns: {missing}"
        )

    if data[TARGET].isna().any():
        raise ValueError(
            f"{split_name} target contains missing values."
        )

    for column in REQUIRED_FEATURES:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    if data[REQUIRED_FEATURES].isna().any().any():
        raise ValueError(
            f"{split_name} features contain missing or "
            "non-numeric values."
        )

    if not np.isfinite(
        data[REQUIRED_FEATURES].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            f"{split_name} features contain infinite values."
        )

    if (data[["packet_count", "byte_count"]] < 0).any().any():
        raise ValueError(
            f"{split_name} contains negative packet or byte counts. "
            "Investigate these values before applying log1p."
        )


# ============================================================
# 4. CREATE ENGINEERED FEATURES
# ============================================================

def engineer_features(data):
    """
    Create the same three model features for any dataset.

    log1p(x) = log(1 + x)
    duration_total_sec = duration_sec + duration_nsec / 1e9
    """

    engineered = pd.DataFrame(index=data.index)

    engineered["log_packet_count"] = np.log1p(
        data["packet_count"].astype(float)
    )

    engineered["log_byte_count"] = np.log1p(
        data["byte_count"].astype(float)
    )

    engineered["duration_total_sec"] = (
        data["duration_sec"].astype(float)
        + data["duration_nsec"].astype(float) / 1_000_000_000
    )

    return engineered[ENGINEERED_FEATURES]


X_train = engineer_features(train_df)
X_test = engineer_features(test_df)

y_train = train_df[TARGET].copy()
y_test = test_df[TARGET].copy()

print("\n[2] FEATURE ENGINEERING COMPLETED")
print("-" * 70)

print("Original features:")
for feature in REQUIRED_FEATURES:
    print(f"- {feature}")

print("\nEngineered features:")
for feature in ENGINEERED_FEATURES:
    print(f"- {feature}")

print("\nTraining feature preview:")
print(X_train.head().round(4).to_string())


# ============================================================
# 5. VALIDATE TARGET CLASS CONSISTENCY
# ============================================================

train_classes = set(y_train.astype(str).unique())
test_classes = set(y_test.astype(str).unique())

if train_classes != test_classes:
    raise ValueError(
        "Training and testing data do not contain the same "
        f"target classes. Train={train_classes}; Test={test_classes}"
    )

print("\n[3] TARGET VALIDATION")
print("-" * 70)
print("Training class counts:")
print(y_train.value_counts().sort_index().to_string())

print("\nTesting class counts:")
print(y_test.value_counts().sort_index().to_string())


# ============================================================
# 6. SAVE UN-SCALED ENGINEERED DATA
# ============================================================

engineered_train = X_train.copy()
engineered_train[TARGET] = y_train.to_numpy()

engineered_test = X_test.copy()
engineered_test[TARGET] = y_test.to_numpy()

engineered_train.to_csv(
    OUTPUT_DIR / "engineered_train.csv",
    index=False
)

engineered_test.to_csv(
    OUTPUT_DIR / "engineered_test.csv",
    index=False
)

print("\n[4] ENGINEERED DATA SAVED")
print("-" * 70)
print("engineered_train.csv")
print("engineered_test.csv")


# ============================================================
# 7. SCALE THE FEATURES WITHOUT TEST-DATA LEAKAGE
# ============================================================

# Fit the scaler using training data only.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

# Apply the already-fitted scaler to the test data.
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrames.
X_train_scaled_df = pd.DataFrame(
    X_train_scaled,
    columns=ENGINEERED_FEATURES,
    index=X_train.index
)

X_test_scaled_df = pd.DataFrame(
    X_test_scaled,
    columns=ENGINEERED_FEATURES,
    index=X_test.index
)

scaled_train = X_train_scaled_df.copy()
scaled_train[TARGET] = y_train.to_numpy()

scaled_test = X_test_scaled_df.copy()
scaled_test[TARGET] = y_test.to_numpy()

scaled_train.to_csv(
    OUTPUT_DIR / "scaled_train.csv",
    index=False
)

scaled_test.to_csv(
    OUTPUT_DIR / "scaled_test.csv",
    index=False
)

joblib.dump(
    scaler,
    OUTPUT_DIR / "feature_scaler.joblib"
)

pd.DataFrame({
    "feature_name": ENGINEERED_FEATURES,
    "training_mean_before_scaling": scaler.mean_,
    "training_standard_deviation_before_scaling": scaler.scale_,
}).to_csv(
    OUTPUT_DIR / "feature_names.csv",
    index=False
)

print("\n[5] SCALING COMPLETED")
print("-" * 70)
print("Scaler fitted on training data only.")
print("Test data transformed using the training scaler.")
print("\nScaled training feature preview:")
print(X_train_scaled_df.head().round(4).to_string())

print("\nScaled training feature means (approximately zero):")
print(X_train_scaled_df.mean().round(6).to_string())

print("\nSaved:")
print("- scaled_train.csv")
print("- scaled_test.csv")
print("- feature_scaler.joblib")
print("- feature_names.csv")


# ============================================================
# 8. WRITE A SUMMARY REPORT
# ============================================================

summary = [
    "STAGE 4: FEATURE ENGINEERING SUMMARY",
    "=" * 55,
    f"Training records: {len(X_train):,}",
    f"Testing records: {len(X_test):,}",
    f"Number of engineered features: {len(ENGINEERED_FEATURES)}",
    "",
    "ENGINEERED FEATURES:",
    "- log_packet_count = log(1 + packet_count)",
    "- log_byte_count = log(1 + byte_count)",
    "- duration_total_sec = duration_sec + duration_nsec / 1e9",
    "",
    "SCALING:",
    "- StandardScaler was fitted using training data only.",
    "- The fitted scaler was applied to the testing data.",
    "- The target column was not scaled.",
    "",
    "TARGET CLASS MAPPING:",
    "- The original Stage 2 numeric class encoding is preserved.",
    "- Refer to prepared_data/class_mapping.csv for class names.",
    "",
    "IMPORTANT LIMITATION:",
    "The labels were assigned from separate traffic scenarios.",
    "Successful model performance will not by itself prove that",
    "the model generalises to independently observed network congestion.",
    "",
    "FILES:",
    "- engineered_train.csv",
    "- engineered_test.csv",
    "- scaled_train.csv",
    "- scaled_test.csv",
    "- feature_scaler.joblib",
    "- feature_names.csv",
]

(OUTPUT_DIR / "stage4_summary.txt").write_text(
    "\n".join(summary),
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("STAGE 4 COMPLETED SUCCESSFULLY")
print("=" * 70)
print(f"Results directory: {OUTPUT_DIR.resolve()}")
print(f"Summary: {(OUTPUT_DIR / 'stage4_summary.txt').resolve()}")
print("\nNext: Stage 5 - Baseline Machine Learning Models.")
