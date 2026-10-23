
"""
STAGE 7: DATASET DISTRIBUTION AND FEATURE RELIABILITY AUDIT

Purpose:
1. Compare training and testing feature distributions.
2. Compare feature distributions within each class.
3. Investigate repeated observations.
4. Check whether numeric features differ across classes.
5. Audit timestamps if they remain available in Stage 2 files.
6. Save a diagnostic report for research interpretation.

This script does not retrain or tune any model.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ks_2samp, kruskal

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ORIGINAL_FILE = BASE_DIR / "network_congestion_labeled.csv"

STAGE2_DIR = BASE_DIR / "prepared_data"
STAGE4_DIR = BASE_DIR / "stage4_results"

TRAIN_FILE = STAGE4_DIR / "engineered_train.csv"
TEST_FILE = STAGE4_DIR / "engineered_test.csv"

STAGE2_TRAIN_FILE = STAGE2_DIR / "train_data.csv"
STAGE2_TEST_FILE = STAGE2_DIR / "test_data.csv"

OUTPUT_DIR = BASE_DIR / "stage7_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "congestion_level"

FEATURES = [
    "log_packet_count",
    "log_byte_count",
    "duration_total_sec",
]

LABEL_NAMES = {
    0: "High",
    1: "Low",
    2: "Medium",
}

CLASS_ORDER = [0, 1, 2]

sns.set_theme(style="whitegrid")


# ============================================================
# 2. LOAD THE ENGINEERED DATA
# ============================================================

print("=" * 70)
print("STAGE 7: DATASET DISTRIBUTION AUDIT")
print("=" * 70)

for path in [TRAIN_FILE, TEST_FILE]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Check the Stage 4 output folder and filenames."
        )

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

for name, data in [
    ("Training", train_df),
    ("Testing", test_df),
]:
    if TARGET not in data.columns:
        raise ValueError(
            f"{name} engineered data is missing {TARGET}."
        )

    missing = [
        feature for feature in FEATURES
        if feature not in data.columns
    ]

    if missing:
        raise ValueError(
            f"{name} data is missing features: {missing}"
        )

X_train = train_df[FEATURES].apply(
    pd.to_numeric, errors="coerce"
)
X_test = test_df[FEATURES].apply(
    pd.to_numeric, errors="coerce"
)

y_train = pd.to_numeric(
    train_df[TARGET], errors="coerce"
)
y_test = pd.to_numeric(
    test_df[TARGET], errors="coerce"
)

if (
    X_train.isna().any().any()
    or X_test.isna().any().any()
    or y_train.isna().any()
    or y_test.isna().any()
):
    raise ValueError(
        "Missing or non-numeric values were found. "
        "Investigate these before continuing."
    )

print("\n[1] DATA LOADED")
print("-" * 70)
print(f"Training records: {len(train_df):,}")
print(f"Testing records : {len(test_df):,}")
print(f"Features        : {FEATURES}")


# ============================================================
# 3. REPEATED FEATURE COMBINATIONS
# ============================================================

print("\n[2] REPEATED FEATURE COMBINATIONS")
print("-" * 70)

def duplicate_summary(X, y, split_name):
    combined = X.copy()
    combined[TARGET] = np.asarray(y)

    unique_features = X.drop_duplicates().shape[0]
    unique_complete = combined.drop_duplicates().shape[0]

    feature_counts = X.value_counts(dropna=False)
    repeated_feature_rows = int(
        len(X) - unique_features
    )

    # Exact same feature vector associated with multiple labels.
    grouped = (
        combined.groupby(FEATURES, dropna=False)[TARGET]
        .nunique()
    )

    conflicts = int((grouped > 1).sum())

    result = {
        "split": split_name,
        "records": len(X),
        "unique_feature_vectors": unique_features,
        "repeated_feature_rows": repeated_feature_rows,
        "unique_feature_and_label_rows": unique_complete,
        "conflicting_feature_vectors": conflicts,
        "largest_feature_group": (
            int(feature_counts.max())
            if len(feature_counts) else 0
        ),
    }

    print(f"\n{split_name}:")
    for key, value in result.items():
        if key != "split":
            print(f"  {key}: {value:,}")

    return result


duplicate_results = [
    duplicate_summary(X_train, y_train, "Training"),
    duplicate_summary(X_test, y_test, "Testing"),
]

duplicate_df = pd.DataFrame(duplicate_results)
duplicate_df.to_csv(
    OUTPUT_DIR / "duplicate_feature_audit.csv",
    index=False,
)

# Check whether any feature vectors occur in both train and test.
train_index = pd.MultiIndex.from_frame(
    X_train.drop_duplicates()
)
test_index = pd.MultiIndex.from_frame(
    X_test.drop_duplicates()
)

shared_vectors = len(train_index.intersection(test_index))

print(
    "\nUnique feature vectors shared between train and test:",
    shared_vectors,
)

pd.DataFrame([{
    "unique_training_vectors": len(train_index),
    "unique_testing_vectors": len(test_index),
    "shared_unique_vectors": shared_vectors,
}]).to_csv(
    OUTPUT_DIR / "train_test_feature_overlap.csv",
    index=False,
)


# ============================================================
# 4. OVERALL TRAINING VS TESTING DISTRIBUTION
# ============================================================

print("\n[3] OVERALL TRAIN/TEST FEATURE DISTRIBUTIONS")
print("-" * 70)

distribution_rows = []

for feature in FEATURES:
    train_values = X_train[feature].to_numpy(dtype=float)
    test_values = X_test[feature].to_numpy(dtype=float)

    train_mean = np.mean(train_values)
    test_mean = np.mean(test_values)

    train_std = np.std(train_values, ddof=1)
    test_std = np.std(test_values, ddof=1)

    pooled_std = np.sqrt(
        (train_std ** 2 + test_std ** 2) / 2
    )

    if pooled_std > 0:
        standardized_mean_difference = (
            test_mean - train_mean
        ) / pooled_std
    else:
        standardized_mean_difference = np.nan

    ks_result = ks_2samp(
        train_values,
        test_values,
        alternative="two-sided",
        method="auto",
    )

    row = {
        "feature": feature,
        "train_mean": train_mean,
        "test_mean": test_mean,
        "train_std": train_std,
        "test_std": test_std,
        "standardized_mean_difference":
            standardized_mean_difference,
        "ks_statistic": ks_result.statistic,
        "ks_p_value": ks_result.pvalue,
    }

    distribution_rows.append(row)

    print(f"\n{feature}")
    print(f"  Train mean: {train_mean:.6f}")
    print(f"  Test mean : {test_mean:.6f}")
    print(
        "  Standardized mean difference: "
        f"{standardized_mean_difference:.4f}"
    )
    print(f"  KS statistic: {ks_result.statistic:.4f}")
    print(f"  KS p-value  : {ks_result.pvalue:.6g}")

distribution_df = pd.DataFrame(distribution_rows)

distribution_df.to_csv(
    OUTPUT_DIR / "train_test_distribution.csv",
    index=False,
)


# ============================================================
# 5. COMPARE DISTRIBUTIONS WITHIN EACH CLASS
# ============================================================

print("\n[4] TRAIN/TEST COMPARISON WITHIN EACH CLASS")
print("-" * 70)

class_distribution_rows = []

for label in CLASS_ORDER:
    train_mask = y_train == label
    test_mask = y_test == label

    print(f"\nClass: {LABEL_NAMES[label]}")

    for feature in FEATURES:
        train_values = X_train.loc[
            train_mask, feature
        ].to_numpy(dtype=float)

        test_values = X_test.loc[
            test_mask, feature
        ].to_numpy(dtype=float)

        if len(train_values) == 0 or len(test_values) == 0:
            continue

        ks_result = ks_2samp(
            train_values,
            test_values,
            method="auto",
        )

        train_mean = float(np.mean(train_values))
        test_mean = float(np.mean(test_values))

        row = {
            "class": LABEL_NAMES[label],
            "feature": feature,
            "train_count": len(train_values),
            "test_count": len(test_values),
            "train_mean": train_mean,
            "test_mean": test_mean,
            "ks_statistic": ks_result.statistic,
            "ks_p_value": ks_result.pvalue,
        }

        class_distribution_rows.append(row)

        print(
            f"  {feature}: "
            f"train mean={train_mean:.4f}, "
            f"test mean={test_mean:.4f}, "
            f"KS={ks_result.statistic:.4f}, "
            f"p={ks_result.pvalue:.4g}"
        )

class_distribution_df = pd.DataFrame(
    class_distribution_rows
)

class_distribution_df.to_csv(
    OUTPUT_DIR / "class_conditional_distribution.csv",
    index=False,
)


# ============================================================
# 6. TEST WHETHER FEATURES DIFFER ACROSS CLASSES
# ============================================================

print("\n[5] FEATURE DIFFERENCES ACROSS CONGESTION CLASSES")
print("-" * 70)

class_test_rows = []

for feature in FEATURES:
    groups = []

    for label in CLASS_ORDER:
        values = X_train.loc[
            y_train == label, feature
        ].to_numpy(dtype=float)

        if len(values) > 0:
            groups.append(values)

    if len(groups) < 2:
        continue

    # Kruskal-Wallis is a non-parametric test for differences
    # between multiple groups. It does not measure predictive
    # usefulness and does not prove causation.
    result = kruskal(*groups)

    row = {
        "feature": feature,
        "kruskal_statistic": result.statistic,
        "p_value": result.pvalue,
        "training_class_means_differ":
            bool(result.pvalue < 0.05),
    }

    class_test_rows.append(row)

    print(f"\nFeature: {feature}")
    print(f"  Kruskal-Wallis statistic: {result.statistic:.4f}")
    print(f"  p-value: {result.pvalue:.6g}")

class_tests_df = pd.DataFrame(class_test_rows)

class_tests_df.to_csv(
    OUTPUT_DIR / "class_difference_tests.csv",
    index=False,
)


# ============================================================
# 7. PLOT CLASS-CONDITIONAL DISTRIBUTIONS
# ============================================================

print("\n[6] GENERATING FEATURE DISTRIBUTION PLOTS")
print("-" * 70)

plot_df = pd.concat([
    X_train.assign(split="Training", **{}),
    X_test.assign(split="Testing", **{}),
], ignore_index=True)

plot_df[TARGET] = pd.concat(
    [y_train.reset_index(drop=True),
     y_test.reset_index(drop=True)],
    ignore_index=True,
)

plot_df["class_name"] = plot_df[TARGET].map(LABEL_NAMES)

for feature in FEATURES:
    fig, ax = plt.subplots(figsize=(9, 5))

    sns.boxplot(
        data=plot_df,
        x="class_name",
        y=feature,
        hue="split",
        order=["High", "Low", "Medium"],
        ax=ax,
    )

    ax.set_title(
        f"{feature}: Training vs Testing by Class"
    )
    ax.set_xlabel("Assigned Congestion Class")
    ax.set_ylabel(feature)

    plt.tight_layout()

    output_path = OUTPUT_DIR / (
        feature + "_train_test_boxplot.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path.name}")


# ============================================================
# 8. AUDIT TIMESTAMP OVERLAP, IF AVAILABLE
# ============================================================

print("\n[7] TIMESTAMP AUDIT")
print("-" * 70)

timestamp_summary = []

if (
    STAGE2_TRAIN_FILE.exists()
    and STAGE2_TEST_FILE.exists()
):
    stage2_train = pd.read_csv(STAGE2_TRAIN_FILE)
    stage2_test = pd.read_csv(STAGE2_TEST_FILE)

    if (
        "timestamp" in stage2_train.columns
        and "timestamp" in stage2_test.columns
    ):
        train_times = pd.to_datetime(
            stage2_train["timestamp"],
            errors="coerce",
            utc=True,
        ).dropna()

        test_times = pd.to_datetime(
            stage2_test["timestamp"],
            errors="coerce",
            utc=True,
        ).dropna()

        train_unique = set(train_times.unique())
        test_unique = set(test_times.unique())

        overlap = train_unique.intersection(test_unique)

        timestamp_summary.append({
            "training_unique_timestamps": len(train_unique),
            "testing_unique_timestamps": len(test_unique),
            "shared_timestamps": len(overlap),
        })

        print(
            "Training unique timestamps:",
            len(train_unique),
        )
        print(
            "Testing unique timestamps:",
            len(test_unique),
        )
        print(
            "Shared timestamps:",
            len(overlap),
        )

        if len(overlap) > 0:
            print(
                "WARNING: Some timestamps appear in both "
                "training and testing data."
            )
        else:
            print(
                "No timestamp overlap found in the available "
                "Stage 2 files."
            )
    else:
        print(
            "Timestamp column is not present in both Stage 2 files."
        )
else:
    print(
        "Stage 2 train/test files were not found; "
        "timestamp audit skipped."
    )

pd.DataFrame(timestamp_summary).to_csv(
    OUTPUT_DIR / "timestamp_overlap_audit.csv",
    index=False,
)


# ============================================================
# 9. REVIEW ORIGINAL DATASET LABEL DISTRIBUTIONS
# ============================================================

print("\n[8] ORIGINAL DATASET REVIEW")
print("-" * 70)

if ORIGINAL_FILE.exists():
    original = pd.read_csv(ORIGINAL_FILE)

    if TARGET in original.columns:
        print("\nOriginal class counts:")
        print(original[TARGET].value_counts().to_string())

        original_counts = (
            original[TARGET]
            .value_counts()
            .rename_axis("congestion_level")
            .reset_index(name="record_count")
        )

        original_counts.to_csv(
            OUTPUT_DIR / "original_class_counts.csv",
            index=False,
        )

    numeric_original = [
        col for col in [
            "packet_count",
            "byte_count",
            "duration_sec",
            "duration_nsec",
        ]
        if col in original.columns
    ]

    if numeric_original and TARGET in original.columns:
        raw_means = original.groupby(TARGET)[
            numeric_original
        ].mean()

        raw_means.to_csv(
            OUTPUT_DIR / "original_raw_feature_means.csv"
        )

        print("\nOriginal raw feature means by assigned class:")
        print(raw_means.round(3).to_string())

    if "timestamp" in original.columns and TARGET in original.columns:
        original["timestamp"] = pd.to_datetime(
            original["timestamp"],
            errors="coerce",
            utc=True,
        )

        time_ranges = (
            original.dropna(subset=["timestamp"])
            .groupby(TARGET)["timestamp"]
            .agg(["min", "max", "nunique"])
        )

        time_ranges.to_csv(
            OUTPUT_DIR / "original_timestamp_ranges.csv"
        )

        print("\nOriginal timestamp ranges by assigned class:")
        print(time_ranges.to_string())
else:
    print("Original dataset not found; skipped.")


# ============================================================
# 10. WRITE THE FINAL AUDIT SUMMARY
# ============================================================

summary = [
    "STAGE 7: DATASET DISTRIBUTION AUDIT",
    "=" * 60,
    f"Training records: {len(X_train)}",
    f"Testing records: {len(X_test)}",
    f"Features analysed: {', '.join(FEATURES)}",
    "",
    "DUPLICATE FEATURE AUDIT:",
]

for _, row in duplicate_df.iterrows():
    summary.append(
        f"{row['split']}: "
        f"{row['unique_feature_vectors']} unique feature vectors; "
        f"{row['repeated_feature_rows']} repeated feature rows; "
        f"{row['conflicting_feature_vectors']} conflicting vectors."
    )

summary.extend([
    "",
    f"Unique feature vectors shared between train and test: "
    f"{shared_vectors}",
    "",
    "OVERALL TRAIN/TEST DISTRIBUTION:",
])

for _, row in distribution_df.iterrows():
    summary.append(
        f"{row['feature']}: "
        f"train mean={row['train_mean']:.6f}, "
        f"test mean={row['test_mean']:.6f}, "
        f"KS={row['ks_statistic']:.4f}, "
        f"p={row['ks_p_value']:.6g}"
    )

summary.extend([
    "",
    "CLASS DIFFERENCE TESTS:",
])

for _, row in class_tests_df.iterrows():
    summary.append(
        f"{row['feature']}: "
        f"Kruskal-Wallis p={row['p_value']:.6g}"
    )

summary.extend([
    "",
    "INTERPRETATION NOTES:",
    "The Kolmogorov-Smirnov test compares empirical distributions. "
    "A small p-value indicates evidence of a difference, but does "
    "not by itself establish that the difference is practically "
    "important or that the model is invalid.",
    "The Kruskal-Wallis test checks whether distributions differ "
    "across classes. Statistical significance alone does not prove "
    "that a feature is useful for prediction.",
    "Repeated feature vectors can arise naturally from simulated "
    "traffic, counters, or rounded measurements. They should be "
    "interpreted in the context of how the data was generated.",
    "The congestion labels originate from source scenarios. "
    "Without independently measured congestion ground truth, "
    "classification results cannot establish real-world detection "
    "performance.",
])

summary_path = OUTPUT_DIR / "stage7_audit_summary.txt"

summary_path.write_text(
    "\n".join(summary),
    encoding="utf-8",
)

print("\n" + "=" * 70)
print("STAGE 7 COMPLETED SUCCESSFULLY")
print("=" * 70)
print(f"Results directory: {OUTPUT_DIR.resolve()}")
print(f"Summary file: {summary_path.resolve()}")

print("\nPlease send me:")
print("- stage7_results/stage7_audit_summary.txt")
print("- stage7_results/train_test_distribution.csv")
print("- stage7_results/class_conditional_distribution.csv")
print("- stage7_results/duplicate_feature_audit.csv")
