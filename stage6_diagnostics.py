
"""
STAGE 6: DIAGNOSTIC ANALYSIS AND MODEL VALIDATION

This stage investigates the poor baseline model results.
It does not change the existing training or testing datasets.

Inputs:
    stage4_results/scaled_train.csv
    stage4_results/scaled_test.csv
    stage5_results/trained_models/*.joblib

Outputs:
    stage6_results/
        diagnostic_summary.txt
        per_class_metrics.csv
        train_test_performance.csv
        feature_summary_by_class.csv
        conflicting_feature_combinations.csv
        majority_baseline.txt
        class_feature_means.png
        model_train_test_comparison.png
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    f1_score,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

STAGE4_DIR = BASE_DIR / "stage4_results"
MODEL_DIR = BASE_DIR / "stage5_results" / "trained_models"
OUTPUT_DIR = BASE_DIR / "stage6_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = STAGE4_DIR / "scaled_train.csv"
TEST_FILE = STAGE4_DIR / "scaled_test.csv"

TARGET = "congestion_level"

# LabelEncoder from Stage 2 assigned:
# 0 = High, 1 = Low, 2 = Medium
LABEL_NAMES = {
    0: "High",
    1: "Low",
    2: "Medium",
}

LABEL_ORDER = [0, 1, 2]

sns.set_theme(style="whitegrid")


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

print("=" * 70)
print("STAGE 6: DIAGNOSTIC ANALYSIS")
print("=" * 70)

for file_path in [TRAIN_FILE, TEST_FILE]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

for name, data in [
    ("Training", train_df),
    ("Testing", test_df),
]:
    if TARGET not in data.columns:
        raise ValueError(
            f"{name} data is missing '{TARGET}'."
        )

    if data[TARGET].isna().any():
        raise ValueError(
            f"{name} data has missing target labels."
        )

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET].astype(int)

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET].astype(int)

if list(X_train.columns) != list(X_test.columns):
    raise ValueError(
        "Training and testing feature columns do not match."
    )

if not set(y_train.unique()).issubset(LABEL_ORDER):
    raise ValueError("Unexpected training labels.")

if not set(y_test.unique()).issubset(LABEL_ORDER):
    raise ValueError("Unexpected testing labels.")

print("\n[1] DATASET CHECK")
print("-" * 70)
print(f"Training records: {len(train_df):,}")
print(f"Testing records : {len(test_df):,}")
print(f"Features        : {list(X_train.columns)}")


# ============================================================
# 3. CLASS DISTRIBUTION AND MAJORITY BASELINE
# ============================================================

print("\n[2] CLASS DISTRIBUTION")
print("-" * 70)

class_rows = []

for split_name, y in [
    ("Training", y_train),
    ("Testing", y_test),
]:
    counts = y.value_counts().reindex(
        LABEL_ORDER, fill_value=0
    )

    print(f"\n{split_name}:")
    for label in LABEL_ORDER:
        count = int(counts[label])
        percentage = count / len(y) * 100

        print(
            f"  {LABEL_NAMES[label]}: "
            f"{count:,} ({percentage:.2f}%)"
        )

        class_rows.append({
            "split": split_name,
            "class": LABEL_NAMES[label],
            "count": count,
            "percentage": percentage,
        })

pd.DataFrame(class_rows).to_csv(
    OUTPUT_DIR / "class_distribution.csv",
    index=False,
)

# Baseline: always predict the most frequent training class.
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
dummy_predictions = dummy.predict(X_test)

dummy_accuracy = accuracy_score(
    y_test, dummy_predictions
)

dummy_macro_f1 = f1_score(
    y_test,
    dummy_predictions,
    labels=LABEL_ORDER,
    average="macro",
    zero_division=0,
)

majority_text = (
    "MAJORITY-CLASS BASELINE\n"
    "=======================\n"
    f"Most frequent training class: "
    f"{LABEL_NAMES[int(y_train.mode().iloc[0])]}\n"
    f"Test accuracy: {dummy_accuracy:.4f}\n"
    f"Test macro-F1: {dummy_macro_f1:.4f}\n"
)

(OUTPUT_DIR / "majority_baseline.txt").write_text(
    majority_text,
    encoding="utf-8",
)

print("\nMajority-class baseline:")
print(majority_text)


# ============================================================
# 4. FEATURE SUMMARY BY CLASS
# ============================================================

print("\n[3] FEATURE SUMMARY BY CLASS")
print("-" * 70)

train_for_summary = X_train.copy()
train_for_summary[TARGET] = y_train.to_numpy()

feature_summary = (
    train_for_summary
    .groupby(TARGET)[list(X_train.columns)]
    .agg(["mean", "median", "std", "min", "max"])
)

feature_summary.to_csv(
    OUTPUT_DIR / "feature_summary_by_class.csv"
)

class_means = (
    train_for_summary
    .groupby(TARGET)[list(X_train.columns)]
    .mean()
    .reindex(LABEL_ORDER)
)

class_means.index = [
    LABEL_NAMES[int(label)] for label in class_means.index
]

print(class_means.round(4).to_string())

ax = class_means.plot(
    kind="bar",
    figsize=(10, 6),
)

ax.set_title("Mean Scaled Feature Values by Class")
ax.set_xlabel("Congestion Class")
ax.set_ylabel("Mean Scaled Value")
ax.tick_params(axis="x", rotation=0)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "class_feature_means.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 5. IDENTICAL FEATURES WITH CONFLICTING LABELS
# ============================================================

print("\n[4] CHECKING IDENTICAL FEATURE COMBINATIONS")
print("-" * 70)

# Exact matching is appropriate here because the scaled CSV
# values are saved with decimal precision. This audit checks
# whether the same stored feature vector appears with different
# target labels.

train_combinations = X_train.copy()
train_combinations[TARGET] = y_train.to_numpy()

grouped = (
    train_combinations
    .groupby(list(X_train.columns), dropna=False)[TARGET]
    .agg(["size", "nunique", "unique"])
    .reset_index()
)

conflicts = grouped[grouped["nunique"] > 1].copy()

print(
    "Unique feature combinations in training:",
    len(grouped),
)

print(
    "Feature combinations linked to multiple classes:",
    len(conflicts),
)

if len(conflicts) > 0:
    conflicts["class_labels"] = conflicts["unique"].apply(
        lambda labels: ", ".join(
            LABEL_NAMES[int(label)] for label in labels
        )
    )

    conflicts = conflicts.drop(columns=["unique"])

conflicts.to_csv(
    OUTPUT_DIR / "conflicting_feature_combinations.csv",
    index=False,
)

# Count repeated complete rows, including their target.
complete_duplicates = train_combinations.duplicated().sum()

print(
    "Repeated complete training rows:",
    int(complete_duplicates),
)

if len(conflicts) > 0:
    print("\nExamples of conflicting feature combinations:")
    print(conflicts.head(10).to_string(index=False))
else:
    print(
        "No exact stored feature combinations were found "
        "with conflicting labels."
    )


# ============================================================
# 6. LOAD MODELS AND COMPARE TRAINING VS TEST PERFORMANCE
# ============================================================

print("\n[5] TRAINING AND TESTING PERFORMANCE")
print("-" * 70)

model_files = {
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
    "K-Nearest Neighbors": "k-nearest_neighbors.joblib",
    "Logistic Regression": "logistic_regression.joblib",
}

performance_rows = []
per_class_rows = []
report_lines = []

for model_name, filename in model_files.items():

    model_path = MODEL_DIR / filename

    if not model_path.exists():
        print(f"Skipping missing model: {model_path}")
        continue

    model = joblib.load(model_path)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_accuracy = accuracy_score(
        y_train, train_predictions
    )
    test_accuracy = accuracy_score(
        y_test, test_predictions
    )

    train_macro_f1 = f1_score(
        y_train,
        train_predictions,
        labels=LABEL_ORDER,
        average="macro",
        zero_division=0,
    )

    test_macro_f1 = f1_score(
        y_test,
        test_predictions,
        labels=LABEL_ORDER,
        average="macro",
        zero_division=0,
    )

    performance_rows.append({
        "model": model_name,
        "training_accuracy": train_accuracy,
        "testing_accuracy": test_accuracy,
        "training_macro_f1": train_macro_f1,
        "testing_macro_f1": test_macro_f1,
        "accuracy_gap": train_accuracy - test_accuracy,
        "macro_f1_gap": train_macro_f1 - test_macro_f1,
    })

    report_lines.append(
        f"\n{'=' * 70}\n"
        f"MODEL: {model_name}\n"
        f"{'=' * 70}\n"
        "TRAINING CLASSIFICATION REPORT\n"
        + classification_report(
            y_train,
            train_predictions,
            labels=LABEL_ORDER,
            target_names=[
                LABEL_NAMES[x] for x in LABEL_ORDER
            ],
            zero_division=0,
            digits=4,
        )
        + "\nTEST CLASSIFICATION REPORT\n"
        + classification_report(
            y_test,
            test_predictions,
            labels=LABEL_ORDER,
            target_names=[
                LABEL_NAMES[x] for x in LABEL_ORDER
            ],
            zero_division=0,
            digits=4,
        )
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_test,
            test_predictions,
            labels=LABEL_ORDER,
            zero_division=0,
        )
    )

    for i, label in enumerate(LABEL_ORDER):
        per_class_rows.append({
            "model": model_name,
            "class": LABEL_NAMES[label],
            "precision": precision[i],
            "recall": recall[i],
            "f1_score": f1[i],
            "support": support[i],
        })

    print(f"\n{model_name}")
    print(
        f"  Train accuracy : {train_accuracy:.4f}"
    )
    print(
        f"  Test accuracy  : {test_accuracy:.4f}"
    )
    print(
        f"  Train macro-F1 : {train_macro_f1:.4f}"
    )
    print(
        f"  Test macro-F1  : {test_macro_f1:.4f}"
    )
    print(
        f"  Accuracy gap   : "
        f"{train_accuracy - test_accuracy:.4f}"
    )

performance_df = pd.DataFrame(performance_rows)

performance_df.to_csv(
    OUTPUT_DIR / "train_test_performance.csv",
    index=False,
)

pd.DataFrame(per_class_rows).to_csv(
    OUTPUT_DIR / "per_class_metrics.csv",
    index=False,
)

(OUTPUT_DIR / "detailed_classification_reports.txt").write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)


# ============================================================
# 7. PLOT TRAIN VS TEST PERFORMANCE
# ============================================================

if not performance_df.empty:

    plot_df = performance_df.melt(
        id_vars="model",
        value_vars=[
            "training_accuracy",
            "testing_accuracy",
            "training_macro_f1",
            "testing_macro_f1",
        ],
        var_name="metric",
        value_name="score",
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(
        data=plot_df,
        x="model",
        y="score",
        hue="metric",
        ax=ax,
    )

    ax.set_title("Training vs Testing Model Performance")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Metric")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "model_train_test_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 8. WRITE DIAGNOSTIC SUMMARY
# ============================================================

summary = [
    "STAGE 6: DIAGNOSTIC ANALYSIS SUMMARY",
    "=" * 60,
    f"Training records: {len(X_train)}",
    f"Testing records: {len(X_test)}",
    f"Features: {', '.join(X_train.columns)}",
    "",
    "MAJORITY-CLASS BASELINE:",
    f"Test accuracy: {dummy_accuracy:.4f}",
    f"Test macro-F1: {dummy_macro_f1:.4f}",
    "",
    "IDENTICAL FEATURE COMBINATIONS:",
    f"Unique combinations: {len(grouped)}",
    f"Conflicting-label combinations: {len(conflicts)}",
    f"Repeated complete rows: {int(complete_duplicates)}",
    "",
    "TRAIN/TEST PERFORMANCE:",
]

for _, row in performance_df.iterrows():
    summary.append(
        f"{row['model']}: "
        f"train accuracy={row['training_accuracy']:.4f}, "
        f"test accuracy={row['testing_accuracy']:.4f}, "
        f"train macro-F1={row['training_macro_f1']:.4f}, "
        f"test macro-F1={row['testing_macro_f1']:.4f}"
    )

summary.extend([
    "",
    "INTERPRETATION:",
    "A large training-to-testing performance gap may indicate "
    "overfitting or differences between the training and test data.",
    "Low performance on both sets suggests the current features "
    "may not adequately distinguish the assigned classes.",
    "Conflicting identical feature combinations indicate that "
    "the current feature set alone cannot uniquely determine "
    "the label for those observations.",
    "",
    "RESEARCH LIMITATION:",
    "The labels were assigned from scenario filenames rather "
    "than independently measured congestion ground truth. "
    "Results therefore evaluate prediction of these assigned "
    "labels and do not, by themselves, validate real-world "
    "congestion detection.",
])

summary_path = OUTPUT_DIR / "diagnostic_summary.txt"

summary_path.write_text(
    "\n".join(summary),
    encoding="utf-8",
)

print("\n" + "=" * 70)
print("STAGE 6 COMPLETED")
print("=" * 70)
print(f"Results directory: {OUTPUT_DIR.resolve()}")
print(f"Summary: {summary_path}")
print("\nSend me these files:")
print("- stage6_results/diagnostic_summary.txt")
print("- stage6_results/per_class_metrics.csv")
print("- stage6_results/conflicting_feature_combinations.csv")
print("- stage6_results/train_test_performance.csv")
