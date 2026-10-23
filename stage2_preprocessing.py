
# ============================================================
# PROJECT: MACHINE LEARNING-BASED PREDICTION OF NETWORK
#          TRAFFIC CONGESTION IN COMPUTER NETWORKS
#
# STAGE 2: DATA PREPROCESSING AND FEATURE SELECTION
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

DATASET_PATH = "network_congestion_labeled.csv"
OUTPUT_DIR = "prepared_data"
RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET = "congestion_level"
CLASSES = ["Low", "Medium", "High"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("NETWORK CONGESTION PREDICTION")
print("STAGE 2: DATA PREPROCESSING AND FEATURE SELECTION")
print("=" * 70)

# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Cannot find {DATASET_PATH}. "
        "Place it in the project folder."
    )

df = pd.read_csv(DATASET_PATH)

print("\n[1] ORIGINAL DATASET")
print("-" * 70)
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(df.head())

# ------------------------------------------------------------
# 3. VALIDATE REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = [
    "timestamp",
    "dpid",
    "in_port",
    "eth_src",
    "eth_dst",
    "out_port",
    "packet_count",
    "byte_count",
    "duration_sec",
    "duration_nsec",
    TARGET
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# ------------------------------------------------------------
# 4. VALIDATE TARGET LABELS
# ------------------------------------------------------------

print("\n[2] VALIDATING TARGET LABELS")
print("-" * 70)

if df[TARGET].isna().any():
    raise ValueError("The target contains missing labels.")

df[TARGET] = df[TARGET].astype(str).str.strip()

unexpected_classes = set(df[TARGET].unique()) - set(CLASSES)

if unexpected_classes:
    raise ValueError(
        f"Unexpected congestion classes: {unexpected_classes}"
    )

print(df[TARGET].value_counts())

# ------------------------------------------------------------
# 5. REMOVE EXACT DUPLICATES
# ------------------------------------------------------------

print("\n[3] DUPLICATE HANDLING")
print("-" * 70)

duplicates_before = df.duplicated().sum()

print(f"Exact duplicates found: {duplicates_before}")

df = df.drop_duplicates().reset_index(drop=True)

print(f"Rows after duplicate removal: {len(df)}")

# ------------------------------------------------------------
# 6. CHECK MISSING AND INFINITE VALUES
# ------------------------------------------------------------

print("\n[4] DATA QUALITY VALIDATION")
print("-" * 70)

print("Missing values per column:")
print(df.isna().sum())

numeric_columns = df.select_dtypes(
    include=["number"]
).columns

if len(numeric_columns):
    numeric_array = df[numeric_columns].to_numpy(dtype=float)

    if not np.isfinite(numeric_array).all():
        raise ValueError(
            "Infinite numeric values detected. "
            "Investigate them before training."
        )

# Do not silently invent values for missing network measurements.
if df.isna().any().any():
    raise ValueError(
        "Missing values detected. Investigate and handle them "
        "before continuing."
    )

# ------------------------------------------------------------
# 7. PARSE TIMESTAMPS FOR SPLITTING ONLY
# ------------------------------------------------------------

print("\n[5] TIMESTAMP VALIDATION")
print("-" * 70)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

if df["timestamp"].isna().any():
    raise ValueError(
        "Some timestamps could not be parsed."
    )

print("Unique timestamps:", df["timestamp"].nunique())

# IMPORTANT:
# Timestamp is not used as a model input.
# It is used only to keep records from the same time window
# together during the train/test split.

# ------------------------------------------------------------
# 8. SELECT INITIAL PREDICTOR FEATURES
# ------------------------------------------------------------

print("\n[6] FEATURE SELECTION")
print("-" * 70)

# Exclude:
# - timestamp: can reveal which scenario produced a row
# - MAC addresses: identifiers rather than congestion measurements
# - dpid/in_port/out_port: topology identifiers, not direct
#   measures of congestion
#
# Start with numeric flow measurements only.

FEATURES = [
    "packet_count",
    "byte_count",
    "duration_sec",
    "duration_nsec"
]

for feature in FEATURES:
    if not pd.api.types.is_numeric_dtype(df[feature]):
        raise TypeError(
            f"Feature {feature} is not numeric."
        )

X = df[FEATURES].copy()
y = df[TARGET].copy()

print("Selected features:")
for feature in FEATURES:
    print(f"- {feature}")

print("\nExcluded from model inputs:")
print("- timestamp")
print("- dpid")
print("- in_port")
print("- out_port")
print("- eth_src")
print("- eth_dst")

# ------------------------------------------------------------
# 9. ENCODE TARGET LABELS
# ------------------------------------------------------------

print("\n[7] TARGET ENCODING")
print("-" * 70)

label_encoder = LabelEncoder()
label_encoder.fit(CLASSES)

y_encoded = pd.Series(
    label_encoder.transform(y),
    index=y.index,
    name=TARGET
)

for label, encoded in zip(
    label_encoder.classes_,
    label_encoder.transform(label_encoder.classes_)
):
    print(f"{label} -> {encoded}")

# ------------------------------------------------------------
# 10. GROUP-AWARE TRAIN/TEST SPLIT
# ------------------------------------------------------------

print("\n[8] GROUP-AWARE TRAIN/TEST SPLIT")
print("-" * 70)

# Records sharing a timestamp are kept in the same split.
# Each class is split by its unique timestamps separately,
# so all three classes should appear in both sets.

train_timestamp_groups = []
test_timestamp_groups = []

for class_name in CLASSES:

    class_rows = df[df[TARGET] == class_name]

    unique_times = class_rows["timestamp"].drop_duplicates()

    if len(unique_times) < 2:
        raise ValueError(
            f"Not enough unique timestamps for class {class_name}."
        )

    train_times, test_times = train_test_split(
        unique_times.to_numpy(),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_timestamp_groups.extend(train_times)
    test_timestamp_groups.extend(test_times)

train_mask = df["timestamp"].isin(train_timestamp_groups)
test_mask = df["timestamp"].isin(test_timestamp_groups)

X_train = X.loc[train_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y_encoded.loc[train_mask].copy()
y_test = y_encoded.loc[test_mask].copy()

print(f"Training records: {len(X_train)}")
print(f"Testing records : {len(X_test)}")

print("\nTraining class counts:")
print(
    label_encoder.inverse_transform(
        y_train.to_numpy(dtype=int)
    ).tolist().count("Low"),
    "Low"
)
print(
    pd.Series(
        label_encoder.inverse_transform(
            y_train.to_numpy(dtype=int)
        )
    ).value_counts()
)

print("\nTesting class counts:")
print(
    pd.Series(
        label_encoder.inverse_transform(
            y_test.to_numpy(dtype=int)
        )
    ).value_counts()
)

# ------------------------------------------------------------
# 11. VERIFY NO TIMESTAMP OVERLAP
# ------------------------------------------------------------

print("\n[9] TRAIN/TEST LEAKAGE CHECK")
print("-" * 70)

train_times_actual = set(df.loc[train_mask, "timestamp"])
test_times_actual = set(df.loc[test_mask, "timestamp"])

overlap = train_times_actual.intersection(test_times_actual)

print("Shared timestamps:", len(overlap))

if overlap:
    raise RuntimeError(
        "Timestamp leakage detected. Train and test groups overlap."
    )

print("Timestamp groups are separated successfully.")

# ------------------------------------------------------------
# 12. SAVE PREPARED DATASETS
# ------------------------------------------------------------

print("\n[10] SAVING PREPARED DATA")
print("-" * 70)

train_output = X_train.copy()
train_output[TARGET] = y_train

test_output = X_test.copy()
test_output[TARGET] = y_test

train_output.to_csv(
    os.path.join(OUTPUT_DIR, "train_data.csv"),
    index=False
)

test_output.to_csv(
    os.path.join(OUTPUT_DIR, "test_data.csv"),
    index=False
)

# Save class mapping for interpreting model predictions.
mapping = pd.DataFrame({
    "congestion_level": label_encoder.classes_,
    "encoded_value": label_encoder.transform(
        label_encoder.classes_
    )
})

mapping.to_csv(
    os.path.join(OUTPUT_DIR, "class_mapping.csv"),
    index=False
)

# Save feature names for reproducibility.
pd.Series(
    FEATURES,
    name="feature"
).to_csv(
    os.path.join(OUTPUT_DIR, "selected_features.csv"),
    index=False
)

# ------------------------------------------------------------
# 13. SAVE PREPROCESSING SUMMARY
# ------------------------------------------------------------

summary = {
    "original_dataset_rows": len(pd.read_csv(DATASET_PATH)),
    "rows_after_duplicate_removal": len(df),
    "number_of_features": len(FEATURES),
    "training_records": len(X_train),
    "testing_records": len(X_test),
    "unique_timestamps": df["timestamp"].nunique(),
    "shared_train_test_timestamps": len(overlap)
}

pd.Series(summary, name="value").to_csv(
    os.path.join(OUTPUT_DIR, "preprocessing_summary.csv")
)

# ------------------------------------------------------------
# 14. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STAGE 2 COMPLETED")
print("=" * 70)

for key, value in summary.items():
    print(f"{key}: {value}")

print("\nFiles created in:", OUTPUT_DIR)
print("- train_data.csv")
print("- test_data.csv")
print("- class_mapping.csv")
print("- selected_features.csv")
print("- preprocessing_summary.csv")

print(
    "\nNEXT: Inspect the split and feature distributions, "
    "then train baseline models in Stage 3."
)
