
"""
STAGE 3: EXPLORATORY DATA ANALYSIS (EDA)
Project: Machine Learning-Based Prediction of Network Traffic
         Congestion in Computer Networks

Input:
    network_congestion_labeled.csv
    prepared_data/train_data.csv
    prepared_data/test_data.csv

Output:
    stage3_results/
        class_distribution.png
        feature_distributions.png
        feature_boxplots.png
        feature_correlations.png
        feature_means_by_class.png
        timestamp_class_audit.png
        descriptive_statistics.csv
        class_feature_summary.csv
        feature_correlations.csv
        timestamp_audit.csv
        stage3_summary.txt
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_FILE = Path("network_congestion_labeled.csv")
TRAIN_FILE = Path("prepared_data/train_data.csv")
TEST_FILE = Path("prepared_data/test_data.csv")

OUTPUT_DIR = Path("stage3_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "congestion_level"

# These are the initial numeric traffic features selected
# in Stage 2. Identifiers and timestamps are not model features.
FEATURES = [
    "packet_count",
    "byte_count",
    "duration_sec",
    "duration_nsec",
]

CLASS_ORDER = ["Low", "Medium", "High"]

sns.set_theme(style="whitegrid", context="notebook")


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

print("=" * 70)
print("STAGE 3: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find {DATA_FILE}. Run this script from the "
        "project folder containing your dataset."
    )

df = pd.read_csv(DATA_FILE)

required_columns = FEATURES + [TARGET, "timestamp"]
missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing from dataset: {missing_columns}"
    )

# Convert traffic measurements to numeric values.
for column in FEATURES:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Convert target labels to consistent strings.
df[TARGET] = df[TARGET].astype("string").str.strip()

# Parse timestamps for the time-based audit.
df["timestamp"] = pd.to_datetime(
    df["timestamp"], errors="coerce", utc=True
)

# Retain rows with a valid target and numeric feature values.
initial_rows = len(df)

df = df.dropna(subset=FEATURES + [TARGET]).copy()

# Replace infinite numeric values with missing values and remove
# rows that cannot be used for numeric feature analysis.
df[FEATURES] = df[FEATURES].replace(
    [np.inf, -np.inf], np.nan
)
df = df.dropna(subset=FEATURES).copy()

if df.empty:
    raise ValueError("No valid rows remain after data validation.")

unexpected_classes = sorted(
    set(df[TARGET].dropna().unique()) - set(CLASS_ORDER)
)

if unexpected_classes:
    raise ValueError(
        f"Unexpected congestion labels found: {unexpected_classes}. "
        f"Expected labels: {CLASS_ORDER}"
    )

print(f"\nOriginal records: {initial_rows:,}")
print(f"Records retained: {len(df):,}")
print(f"Records excluded: {initial_rows - len(df):,}")
print(f"Number of features analysed: {len(FEATURES)}")
print(f"Output directory: {OUTPUT_DIR.resolve()}")


# ============================================================
# 3. OVERALL DATASET AND CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("SECTION A: CLASS DISTRIBUTION")
print("=" * 70)

class_counts = (
    df[TARGET]
    .value_counts()
    .reindex(CLASS_ORDER, fill_value=0)
)

class_percentages = (
    class_counts / class_counts.sum() * 100
).round(2)

class_distribution = pd.DataFrame({
    "congestion_level": class_counts.index,
    "record_count": class_counts.values,
    "percentage": class_percentages.values,
})

print(class_distribution.to_string(index=False))

class_distribution.to_csv(
    OUTPUT_DIR / "class_distribution.csv",
    index=False
)

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    class_distribution["congestion_level"],
    class_distribution["record_count"]
)

ax.set_title("Distribution of Network Congestion Classes")
ax.set_xlabel("Congestion Level")
ax.set_ylabel("Number of Records")

for bar, percentage in zip(
    bars, class_distribution["percentage"]
):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{int(bar.get_height()):,}\n({percentage:.1f}%)",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "class_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# 4. DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("SECTION B: DESCRIPTIVE STATISTICS")
print("=" * 70)

descriptive_stats = df[FEATURES].describe().T
descriptive_stats["missing_count"] = (
    df[FEATURES].isna().sum()
)
descriptive_stats["zero_count"] = (
    df[FEATURES].eq(0).sum()
)

print(descriptive_stats.round(3).to_string())

descriptive_stats.to_csv(
    OUTPUT_DIR / "descriptive_statistics.csv"
)


# ============================================================
# 5. FEATURE STATISTICS FOR EACH CONGESTION CLASS
# ============================================================

print("\n" + "=" * 70)
print("SECTION C: FEATURE VALUES BY CONGESTION CLASS")
print("=" * 70)

class_feature_summary = (
    df.groupby(TARGET, observed=False)[FEATURES]
    .agg(["mean", "median", "std", "min", "max"])
)

print(class_feature_summary.round(3).to_string())

class_feature_summary.to_csv(
    OUTPUT_DIR / "class_feature_summary.csv"
)

# Create a simpler table for comparing class means.
class_means = (
    df.groupby(TARGET, observed=False)[FEATURES]
    .mean()
    .reindex(CLASS_ORDER)
)

print("\nMean feature values by class:")
print(class_means.round(3).to_string())

class_means.to_csv(
    OUTPUT_DIR / "class_feature_means.csv"
)

ax = class_means.plot(
    kind="bar",
    figsize=(11, 6)
)

ax.set_title("Mean Network Traffic Measurements by Class")
ax.set_xlabel("Congestion Level")
ax.set_ylabel("Mean Feature Value")
ax.legend(title="Feature", bbox_to_anchor=(1.02, 1))
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "feature_means_by_class.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# 6. FEATURE DISTRIBUTIONS
# ============================================================

print("\n" + "=" * 70)
print("SECTION D: FEATURE DISTRIBUTIONS")
print("=" * 70)

fig, axes = plt.subplots(
    2, 2, figsize=(13, 9)
)

axes = axes.flatten()

for index, feature in enumerate(FEATURES):
    ax = axes[index]

    sns.histplot(
        data=df,
        x=feature,
        hue=TARGET,
        hue_order=CLASS_ORDER,
        bins=40,
        element="step",
        stat="density",
        common_norm=False,
        ax=ax
    )

    ax.set_title(f"Distribution of {feature}")
    ax.set_ylabel("Density")

plt.suptitle(
    "Distribution of Numeric Network Traffic Features",
    fontsize=15
)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "feature_distributions.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# 7. BOX PLOTS TO IDENTIFY OUTLIERS
# ============================================================

print("\n" + "=" * 70)
print("SECTION E: OUTLIER ANALYSIS")
print("=" * 70)

fig, axes = plt.subplots(
    2, 2, figsize=(13, 9)
)

axes = axes.flatten()

outlier_rows = []

for index, feature in enumerate(FEATURES):
    ax = axes[index]

    sns.boxplot(
        data=df,
        x=TARGET,
        y=feature,
        order=CLASS_ORDER,
        ax=ax
    )

    ax.set_title(f"{feature} by Congestion Class")
    ax.set_xlabel("Congestion Level")

    # IQR rule: count values outside 1.5 times the
    # interquartile range. This is a diagnostic, not proof
    # that a value is erroneous.
    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_count = (
        (df[feature] < lower_bound) |
        (df[feature] > upper_bound)
    ).sum()

    outlier_rows.append({
        "feature": feature,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": int(outlier_count),
        "outlier_percentage": (
            outlier_count / len(df) * 100
        ),
    })

plt.suptitle(
    "Box Plots of Network Traffic Features",
    fontsize=15
)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "feature_boxplots.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

outlier_summary = pd.DataFrame(outlier_rows)

print(outlier_summary.round(3).to_string(index=False))

outlier_summary.to_csv(
    OUTPUT_DIR / "outlier_summary.csv",
    index=False
)


# ============================================================
# 8. FEATURE CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("SECTION F: FEATURE CORRELATIONS")
print("=" * 70)

correlation_matrix = df[FEATURES].corr(method="pearson")

print(correlation_matrix.round(3).to_string())

correlation_matrix.to_csv(
    OUTPUT_DIR / "feature_correlations.csv"
)

fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    ax=ax
)

ax.set_title("Correlation Between Numeric Network Features")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "feature_correlations.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# 9. TIMESTAMP AND SCENARIO SEPARATION AUDIT
# ============================================================

print("\n" + "=" * 70)
print("SECTION G: TIMESTAMP / SCENARIO AUDIT")
print("=" * 70)

timestamp_valid = df.dropna(subset=["timestamp"]).copy()

if timestamp_valid.empty:
    print("No valid timestamps available for this audit.")
    timestamp_audit = pd.DataFrame()
else:
    # Determine how many different classes occur at each time.
    classes_per_timestamp = (
        timestamp_valid.groupby("timestamp")[TARGET]
        .nunique()
    )

    timestamps_shared_across_classes = (
        classes_per_timestamp > 1
    ).sum()

    timestamp_class_table = pd.crosstab(
        timestamp_valid["timestamp"],
        timestamp_valid[TARGET]
    ).reindex(columns=CLASS_ORDER, fill_value=0)

    timestamps_by_class = (
        timestamp_valid.groupby(TARGET)["timestamp"]
        .agg(["min", "max", "nunique"])
        .reindex(CLASS_ORDER)
    )

    timestamp_audit = timestamps_by_class.reset_index()
    timestamp_audit = timestamp_audit.rename(
        columns={
            TARGET: "congestion_level",
            "min": "first_timestamp",
            "max": "last_timestamp",
            "nunique": "unique_timestamps",
        }
    )

    print("\nTimestamp coverage by class:")
    print(timestamp_audit.to_string(index=False))

    print(
        "\nTimestamps shared by more than one class:",
        int(timestamps_shared_across_classes)
    )

    timestamp_audit.to_csv(
        OUTPUT_DIR / "timestamp_audit.csv",
        index=False
    )

    # Plot the number of records at each timestamp and class.
    records_over_time = (
        timestamp_valid.groupby(
            ["timestamp", TARGET]
        )
        .size()
        .reset_index(name="record_count")
    )

    fig, ax = plt.subplots(figsize=(13, 5))

    sns.scatterplot(
        data=records_over_time,
        x="timestamp",
        y="record_count",
        hue=TARGET,
        hue_order=CLASS_ORDER,
        alpha=0.7,
        ax=ax
    )

    ax.set_title(
        "Records Over Time by Assigned Congestion Class"
    )
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Records at Timestamp")
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "timestamp_class_audit.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ============================================================
# 10. CHECK PREPARED TRAIN AND TEST DATA
# ============================================================

print("\n" + "=" * 70)
print("SECTION H: STAGE 2 TRAIN/TEST FILE CHECK")
print("=" * 70)

split_audit = []

for split_name, file_path in [
    ("Train", TRAIN_FILE),
    ("Test", TEST_FILE),
]:
    if not file_path.exists():
        print(f"{split_name} file not found: {file_path}")
        continue

    split_df = pd.read_csv(file_path)

    if TARGET not in split_df.columns:
        print(
            f"{split_name} file does not contain "
            f"'{TARGET}'. Skipping class check."
        )
        continue

    counts = split_df[TARGET].value_counts()

    print(f"\n{split_name} records: {len(split_df):,}")
    print(counts.to_string())

    for label in CLASS_ORDER:
        split_audit.append({
            "split": split_name,
            "congestion_level": label,
            "record_count": int(counts.get(label, 0)),
        })

pd.DataFrame(split_audit).to_csv(
    OUTPUT_DIR / "train_test_class_counts.csv",
    index=False
)


# ============================================================
# 11. AUTOMATIC FINDINGS SUMMARY
# ============================================================

summary_lines = []

summary_lines.append("STAGE 3: EXPLORATORY DATA ANALYSIS SUMMARY")
summary_lines.append("=" * 55)
summary_lines.append(f"Valid records analysed: {len(df):,}")
summary_lines.append(f"Numeric features: {', '.join(FEATURES)}")
summary_lines.append("")
summary_lines.append("CLASS DISTRIBUTION:")

for _, row in class_distribution.iterrows():
    summary_lines.append(
        f"- {row['congestion_level']}: "
        f"{int(row['record_count']):,} records "
        f"({row['percentage']:.2f}%)"
    )

summary_lines.append("")
summary_lines.append("MEAN FEATURE VALUES BY CLASS:")

for label in CLASS_ORDER:
    if label in class_means.index:
        values = class_means.loc[label]
        summary_lines.append(
            f"- {label}: " +
            ", ".join(
                f"{feature}={values[feature]:.3f}"
                for feature in FEATURES
            )
        )

summary_lines.append("")
summary_lines.append("FEATURE CORRELATIONS:")

for i, feature_a in enumerate(FEATURES):
    for feature_b in FEATURES[i + 1:]:
        value = correlation_matrix.loc[
            feature_a, feature_b
        ]
        summary_lines.append(
            f"- {feature_a} vs {feature_b}: {value:.3f}"
        )

summary_lines.append("")
summary_lines.append("OUTLIER COUNTS (IQR RULE):")

for _, row in outlier_summary.iterrows():
    summary_lines.append(
        f"- {row['feature']}: "
        f"{int(row['outlier_count']):,} values "
        f"({row['outlier_percentage']:.2f}%) outside "
        "the 1.5*IQR limits"
    )

summary_lines.append("")
summary_lines.append("INTERPRETATION CAUTION:")
summary_lines.append(
    "The congestion labels were assigned from scenario filenames. "
    "Differences between classes may therefore reflect the original "
    "scenario setup rather than independently measured congestion."
)
summary_lines.append(
    "High correlations and outliers should be investigated before "
    "training. Correlation does not establish causation."
)
summary_lines.append(
    "Do not use timestamps or scenario identifiers as predictive "
    "features merely because they correlate with the target."
)
summary_lines.append(
    "A high model score alone will not establish real-world "
    "generalisation; later evaluation should use a defensible split "
    "and independent congestion ground truth where possible."
)

summary_path = OUTPUT_DIR / "stage3_summary.txt"
summary_path.write_text(
    "\n".join(summary_lines),
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("STAGE 3 COMPLETED SUCCESSFULLY")
print("=" * 70)
print(f"Results saved in: {OUTPUT_DIR.resolve()}")
print(f"Summary report: {summary_path.resolve()}")
print("\nGenerated plots:")
for filename in [
    "class_distribution.png",
    "feature_distributions.png",
    "feature_boxplots.png",
    "feature_correlations.png",
    "feature_means_by_class.png",
    "timestamp_class_audit.png",
]:
    path = OUTPUT_DIR / filename
    if path.exists():
        print(f"- {path}")

print("\nPlease inspect the charts and send me:")
print("1. The terminal output.")
print("2. The contents of stage3_results/stage3_summary.txt.")
print("3. The feature_correlations.csv values, especially")
print("   the correlation between packet_count and byte_count.")
