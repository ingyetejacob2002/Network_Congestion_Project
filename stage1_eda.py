
# ============================================================
# PROJECT: MACHINE LEARNING-BASED PREDICTION OF NETWORK
#          TRAFFIC CONGESTION IN COMPUTER NETWORKS
#
# STAGE 1: DATASET INSPECTION AND EXPLORATORY DATA ANALYSIS
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("default")

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

DATASET_PATH = "network_congestion_labeled.csv"
OUTPUT_DIR = "eda_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

print("=" * 70)
print("NETWORK TRAFFIC CONGESTION PREDICTION")
print("STAGE 1: DATASET INSPECTION AND EXPLORATORY DATA ANALYSIS")
print("=" * 70)

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}\n"
        "Place the CSV file in the same folder as this script."
    )

df = pd.read_csv(DATASET_PATH)

print("\n[1] DATASET LOADED SUCCESSFULLY")
print("-" * 70)
print(f"Number of records : {df.shape[0]}")
print(f"Number of columns : {df.shape[1]}")

# ------------------------------------------------------------
# 3. DISPLAY FIRST AND LAST RECORDS
# ------------------------------------------------------------

print("\n[2] FIRST FIVE RECORDS")
print("-" * 70)
print(df.head().to_string(index=False))

print("\n[3] LAST FIVE RECORDS")
print("-" * 70)
print(df.tail().to_string(index=False))

# ------------------------------------------------------------
# 4. COLUMN INFORMATION AND DATA TYPES
# ------------------------------------------------------------

print("\n[4] COLUMN NAMES")
print("-" * 70)

for column in df.columns:
    print(f"- {column}")

print("\n[5] DATA TYPES AND NON-NULL COUNTS")
print("-" * 70)
df.info()

# ------------------------------------------------------------
# 5. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\n[6] MISSING VALUES")
print("-" * 70)

missing = df.isnull().sum()

missing_report = pd.DataFrame({
    "Column": missing.index,
    "Missing_Count": missing.values,
    "Missing_Percentage": (
        missing.values / len(df) * 100
    ).round(2)
})

print(missing_report.to_string(index=False))

missing_report.to_csv(
    os.path.join(OUTPUT_DIR, "missing_values_report.csv"),
    index=False
)

# ------------------------------------------------------------
# 6. CHECK DUPLICATE RECORDS
# ------------------------------------------------------------

print("\n[7] DUPLICATE RECORDS")
print("-" * 70)

duplicate_count = df.duplicated().sum()

print(f"Number of completely duplicated rows: {duplicate_count}")

# Do not automatically delete duplicates at this stage.
# Repeated measurements can be legitimate in network telemetry.

# ------------------------------------------------------------
# 7. TARGET VARIABLE INSPECTION
# ------------------------------------------------------------

TARGET = "congestion_level"

print("\n[8] TARGET VARIABLE ANALYSIS")
print("-" * 70)

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' is missing from the dataset."
    )

print("\nUnique congestion classes:")
print(df[TARGET].value_counts(dropna=False))

print("\nClass distribution (%):")
class_percentages = (
    df[TARGET].value_counts(normalize=True, dropna=False) * 100
).round(2)

print(class_percentages)

# Save target distribution
class_counts = df[TARGET].value_counts(dropna=False)

class_report = pd.DataFrame({
    "Congestion_Level": class_counts.index.astype(str),
    "Record_Count": class_counts.values,
    "Percentage": class_percentages.reindex(
        class_counts.index
    ).values
})

class_report.to_csv(
    os.path.join(OUTPUT_DIR, "class_distribution.csv"),
    index=False
)

# ------------------------------------------------------------
# 8. VISUALIZE CLASS DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x=TARGET,
    order=["Low", "Medium", "High"]
)

plt.title("Distribution of Network Congestion Classes")
plt.xlabel("Congestion Level")
plt.ylabel("Number of Records")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "class_distribution.png"),
    dpi=300
)

plt.show()

# ------------------------------------------------------------
# 9. NUMERIC FEATURE SUMMARY
# ------------------------------------------------------------

print("\n[9] NUMERIC FEATURE SUMMARY")
print("-" * 70)

numeric_df = df.select_dtypes(include=np.number)

if numeric_df.empty:
    print("No numeric columns were detected.")
else:
    summary = numeric_df.describe().T
    print(summary.to_string())

    summary.to_csv(
        os.path.join(OUTPUT_DIR, "numeric_summary.csv")
    )

# ------------------------------------------------------------
# 10. CATEGORICAL FEATURE SUMMARY
# ------------------------------------------------------------

print("\n[10] CATEGORICAL FEATURE SUMMARY")
print("-" * 70)

categorical_df = df.select_dtypes(
    include=["object", "category"]
)

for column in categorical_df.columns:
    print(f"\nColumn: {column}")
    print(f"Unique values: {df[column].nunique(dropna=False)}")

    # Limit displayed values to avoid excessively long output.
    print(df[column].value_counts(dropna=False).head(10))

# ------------------------------------------------------------
# 11. NUMERIC FEATURE DISTRIBUTIONS
# ------------------------------------------------------------

print("\n[11] GENERATING NUMERIC FEATURE DISTRIBUTIONS")
print("-" * 70)

if not numeric_df.empty:
    numeric_df.hist(
        figsize=(14, 10),
        bins=30,
        edgecolor="black"
    )

    plt.suptitle(
        "Distributions of Numeric Network Traffic Features",
        fontsize=15
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "numeric_distributions.png"),
        dpi=300
    )

    plt.show()

# ------------------------------------------------------------
# 12. CORRELATION ANALYSIS
# ------------------------------------------------------------

print("\n[12] NUMERIC FEATURE CORRELATION")
print("-" * 70)

if numeric_df.shape[1] >= 2:
    correlation = numeric_df.corr()

    print(correlation.round(3).to_string())

    correlation.to_csv(
        os.path.join(OUTPUT_DIR, "correlation_matrix.csv")
    )

    plt.figure(figsize=(10, 7))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )

    plt.title("Correlation Between Numeric Network Features")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "correlation_heatmap.png"),
        dpi=300
    )

    plt.show()
else:
    print("Not enough numeric columns for correlation analysis.")

# ------------------------------------------------------------
# 13. FEATURE COMPARISON BY CONGESTION CLASS
# ------------------------------------------------------------

print("\n[13] FEATURE COMPARISON BY CONGESTION CLASS")
print("-" * 70)

# These are the numeric columns in the original network dataset.
candidate_features = [
    "packet_count",
    "byte_count",
    "duration_sec",
    "duration_nsec",
    "in_port",
    "out_port",
    "dpid"
]

available_features = [
    col for col in candidate_features
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
]

if available_features:
    grouped_summary = df.groupby(TARGET)[
        available_features
    ].agg(["mean", "median", "std"])

    print(grouped_summary.to_string())

    grouped_summary.to_csv(
        os.path.join(OUTPUT_DIR, "features_by_class.csv")
    )

    # Compare packet and byte counts by class.
    for feature in ["packet_count", "byte_count"]:
        if feature in available_features:
            plt.figure(figsize=(8, 5))

            sns.boxplot(
                data=df,
                x=TARGET,
                y=feature,
                order=["Low", "Medium", "High"]
            )

            plt.title(
                f"{feature.replace('_', ' ').title()} by Congestion Class"
            )
            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    OUTPUT_DIR,
                    f"{feature}_by_class.png"
                ),
                dpi=300
            )

            plt.show()
else:
    print("No candidate network features are available.")

# ------------------------------------------------------------
# 14. CHECK CLASS AND TIMESTAMP OVERLAP
# ------------------------------------------------------------

print("\n[14] TIMESTAMP ANALYSIS")
print("-" * 70)

if "timestamp" in df.columns:
    timestamps = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    print(f"Valid timestamps: {timestamps.notna().sum()}")
    print(f"Invalid timestamps: {timestamps.isna().sum()}")

    print("\nRecords per class and timestamp:")

    timestamp_class_counts = pd.crosstab(
        timestamps,
        df[TARGET]
    )

    print(timestamp_class_counts.head(10).to_string())

    timestamp_class_counts.to_csv(
        os.path.join(OUTPUT_DIR, "timestamp_class_counts.csv")
    )

    # Check whether timestamps appear in more than one class.
    if timestamps.notna().any():
        class_overlap = pd.DataFrame({
            "Congestion_Level": df[TARGET].values,
            "timestamp": timestamps.values
        }).groupby("timestamp")["Congestion_Level"].nunique()

        print(
            "\nTimestamps shared by multiple classes:",
            int((class_overlap > 1).sum())
        )

# ------------------------------------------------------------
# 15. BASIC DATA QUALITY CHECKS
# ------------------------------------------------------------

print("\n[15] DATA QUALITY CHECKS")
print("-" * 70)

print("Infinite numeric values:")

if not numeric_df.empty:
    infinite_counts = np.isinf(
        numeric_df.to_numpy(dtype=float)
    ).sum()

    print(int(infinite_counts))
else:
    print(0)

print("\nRecords per congestion class:")
print(df[TARGET].value_counts().to_string())

print("\nPotentially constant columns:")

constant_columns = [
    col for col in df.columns
    if df[col].nunique(dropna=False) <= 1
]

if constant_columns:
    print(constant_columns)
else:
    print("None detected.")

# ------------------------------------------------------------
# 16. SAVE CLEAN INSPECTION COPY
# ------------------------------------------------------------

df.to_csv(
    os.path.join(OUTPUT_DIR, "dataset_inspection_copy.csv"),
    index=False
)

# ------------------------------------------------------------
# 17. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STAGE 1 COMPLETED")
print("=" * 70)

print(f"Dataset shape: {df.shape}")
print(f"Target column: {TARGET}")
print(f"Output folder: {OUTPUT_DIR}")

print("\nGenerated reports and plots:")
for filename in sorted(os.listdir(OUTPUT_DIR)):
    print(f"- {filename}")

print(
    "\nNEXT STEP: Review the generated reports, especially "
    "class_distribution.csv, features_by_class.csv, and "
    "timestamp_class_counts.csv, before preprocessing."
)
