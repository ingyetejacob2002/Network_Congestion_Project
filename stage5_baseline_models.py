
"""
STAGE 5: BASELINE MACHINE LEARNING MODELS

Project:
Machine Learning-Based Prediction of Network Traffic
Congestion in Computer Networks

Inputs:
    stage4_results/scaled_train.csv
    stage4_results/scaled_test.csv

Outputs:
    stage5_results/
        baseline_model_comparison.csv
        classification_reports.txt
        confusion_matrix_*.png
        model_comparison.png
        trained_models/
        stage5_summary.txt
"""

from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "stage4_results"
OUTPUT_DIR = BASE_DIR / "stage5_results"
MODEL_DIR = OUTPUT_DIR / "trained_models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = RESULTS_DIR / "scaled_train.csv"
TEST_FILE = RESULTS_DIR / "scaled_test.csv"

TARGET = "congestion_level"

# The Stage 4 output uses numeric class labels:
# 0 = High, 1 = Low, 2 = Medium
CLASS_NAMES = ["High", "Low", "Medium"]
CLASS_LABELS = [0, 1, 2]

RANDOM_STATE = 42

sns.set_theme(style="whitegrid")


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("STAGE 5: BASELINE MACHINE LEARNING MODELS")
print("=" * 70)

if not TRAIN_FILE.exists():
    raise FileNotFoundError(
        f"Training file not found: {TRAIN_FILE}\n"
        "Confirm that Stage 4 generated scaled_train.csv."
    )

if not TEST_FILE.exists():
    raise FileNotFoundError(
        f"Testing file not found: {TEST_FILE}\n"
        "Confirm that Stage 4 generated scaled_test.csv."
    )

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

for split_name, split_df in [
    ("Training", train_df),
    ("Testing", test_df),
]:
    if TARGET not in split_df.columns:
        raise ValueError(
            f"{split_name} data does not contain "
            f"the target column '{TARGET}'."
        )

    if split_df[TARGET].isna().any():
        raise ValueError(
            f"{split_name} data contains missing target labels."
        )

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET].astype(int)

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET].astype(int)

if list(X_train.columns) != list(X_test.columns):
    raise ValueError(
        "Training and testing features do not match."
    )

if X_train.empty or X_test.empty:
    raise ValueError("Training or testing features are empty.")

if not np.isfinite(X_train.to_numpy(dtype=float)).all():
    raise ValueError("Training features contain infinite or invalid values.")

if not np.isfinite(X_test.to_numpy(dtype=float)).all():
    raise ValueError("Testing features contain infinite or invalid values.")

if not set(y_train.unique()).issubset(CLASS_LABELS):
    raise ValueError("Unexpected class labels in training data.")

if not set(y_test.unique()).issubset(CLASS_LABELS):
    raise ValueError("Unexpected class labels in testing data.")

if set(y_train.unique()) != set(CLASS_LABELS):
    raise ValueError(
        "Not all three congestion classes appear in training data."
    )

print("\n[1] DATA LOADED")
print("-" * 70)
print(f"Training records: {len(X_train):,}")
print(f"Testing records : {len(X_test):,}")
print(f"Features        : {list(X_train.columns)}")
print(f"Training classes: {sorted(y_train.unique())}")
print(f"Testing classes : {sorted(y_test.unique())}")

print("\nTraining class distribution:")
print(
    y_train.value_counts()
    .reindex(CLASS_LABELS, fill_value=0)
    .rename(index=dict(enumerate(CLASS_NAMES)))
    .to_string()
)

print("\nTesting class distribution:")
print(
    y_test.value_counts()
    .reindex(CLASS_LABELS, fill_value=0)
    .rename(index=dict(enumerate(CLASS_NAMES)))
    .to_string()
)


# ============================================================
# 3. DEFINE BASELINE MODELS
# ============================================================

models = {
    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        max_depth=10,
        min_samples_leaf=5,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
    ),

    "Support Vector Machine": SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight=None,
    ),

    "K-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
        n_jobs=-1,
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    ),
}


# ============================================================
# 4. TRAIN AND EVALUATE MODELS
# ============================================================

print("\n" + "=" * 70)
print("[2] MODEL TRAINING AND EVALUATION")
print("=" * 70)

results = []
classification_reports = []
predictions = {}

for model_name, model in models.items():

    print(f"\nTraining: {model_name}")
    print("-" * 50)

    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    training_seconds = time.perf_counter() - start_time

    prediction_start = time.perf_counter()
    y_pred = model.predict(X_test)
    prediction_seconds = time.perf_counter() - prediction_start

    predictions[model_name] = y_pred

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        labels=CLASS_LABELS,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        labels=CLASS_LABELS,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        labels=CLASS_LABELS,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        labels=CLASS_LABELS,
        average="weighted",
        zero_division=0,
    )

    results.append({
        "model": model_name,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "weighted_f1": weighted_f1,
        "training_time_seconds": training_seconds,
        "prediction_time_seconds": prediction_seconds,
    })

    report = classification_report(
        y_test,
        y_pred,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        zero_division=0,
        digits=4,
    )

    classification_reports.append(
        f"\n{'=' * 70}\n"
        f"MODEL: {model_name}\n"
        f"{'=' * 70}\n"
        f"{report}\n"
        f"Training time: {training_seconds:.4f} seconds\n"
        f"Prediction time: {prediction_seconds:.4f} seconds\n"
    )

    model_path = MODEL_DIR / (
        model_name.lower().replace(" ", "_") + ".joblib"
    )

    joblib.dump(model, model_path)

    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Macro precision: {precision:.4f}")
    print(f"Macro recall   : {recall:.4f}")
    print(f"Macro F1-score : {f1:.4f}")
    print(f"Training time  : {training_seconds:.4f} seconds")
    print(f"Saved model    : {model_path.name}")


# ============================================================
# 5. SAVE COMPARISON TABLE AND REPORTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="macro_f1",
    ascending=False,
).reset_index(drop=True)

results_file = OUTPUT_DIR / "baseline_model_comparison.csv"
results_df.to_csv(results_file, index=False)

reports_file = OUTPUT_DIR / "classification_reports.txt"
reports_file.write_text(
    "\n".join(classification_reports),
    encoding="utf-8",
)

print("\n" + "=" * 70)
print("[3] BASELINE MODEL COMPARISON")
print("=" * 70)

print(results_df.round(4).to_string(index=False))


# ============================================================
# 6. CONFUSION MATRICES
# ============================================================

print("\n" + "=" * 70)
print("[4] GENERATING CONFUSION MATRICES")
print("=" * 70)

for model_name, y_pred in predictions.items():

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=CLASS_LABELS,
    )

    safe_name = model_name.lower().replace(" ", "_")

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )

    ax.set_title(f"Confusion Matrix: {model_name}")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")

    plt.tight_layout()

    plot_path = OUTPUT_DIR / f"confusion_matrix_{safe_name}.png"

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {plot_path.name}")


# ============================================================
# 7. MODEL COMPARISON CHART
# ============================================================

print("\n" + "=" * 70)
print("[5] GENERATING MODEL COMPARISON CHART")
print("=" * 70)

plot_df = results_df.sort_values(
    by="macro_f1",
    ascending=True,
)

fig, ax = plt.subplots(figsize=(10, 6))

positions = np.arange(len(plot_df))
bar_width = 0.25

ax.barh(
    positions - bar_width,
    plot_df["accuracy"],
    height=bar_width,
    label="Accuracy",
)

ax.barh(
    positions,
    plot_df["macro_precision"],
    height=bar_width,
    label="Macro Precision",
)

ax.barh(
    positions + bar_width,
    plot_df["macro_f1"],
    height=bar_width,
    label="Macro F1",
)

ax.set_yticks(positions)
ax.set_yticklabels(plot_df["model"])
ax.set_xlim(0, 1.05)
ax.set_xlabel("Score")
ax.set_title("Baseline Machine Learning Model Comparison")
ax.legend(loc="lower right")

plt.tight_layout()

comparison_plot = OUTPUT_DIR / "model_comparison.png"

plt.savefig(
    comparison_plot,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"Saved: {comparison_plot.name}")


# ============================================================
# 8. SAVE AUTOMATIC SUMMARY
# ============================================================

best_model_row = results_df.iloc[0]

summary_lines = [
    "STAGE 5: BASELINE MODEL TRAINING SUMMARY",
    "=" * 60,
    f"Training records: {len(X_train)}",
    f"Testing records: {len(X_test)}",
    f"Features: {', '.join(X_train.columns)}",
    f"Models evaluated: {len(results_df)}",
    "",
    "MODEL PERFORMANCE:",
]

for _, row in results_df.iterrows():
    summary_lines.append(
        f"{row['model']}: "
        f"Accuracy={row['accuracy']:.4f}, "
        f"Macro-F1={row['macro_f1']:.4f}, "
        f"Weighted-F1={row['weighted_f1']:.4f}"
    )

summary_lines.extend([
    "",
    f"Highest observed macro-F1 model: {best_model_row['model']}",
    f"Highest observed macro-F1: {best_model_row['macro_f1']:.4f}",
    "",
    "IMPORTANT LIMITATION:",
    "The highest-scoring model is the best observed baseline on the "
    "current test set, not necessarily the best model for real-world "
    "deployment. Labels were assigned from scenario filenames, and "
    "independent congestion ground truth has not been established.",
    "Use the results as an initial comparison. Before making strong "
    "claims, investigate scenario separation, duplicate feature rows, "
    "and generalisation to independent traffic scenarios.",
])

summary_path = OUTPUT_DIR / "stage5_summary.txt"

summary_path.write_text(
    "\n".join(summary_lines),
    encoding="utf-8",
)

print("\n" + "=" * 70)
print("STAGE 5 COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"Comparison CSV : {results_file}")
print(f"Reports        : {reports_file}")
print(f"Model chart    : {comparison_plot}")
print(f"Summary        : {summary_path}")
print(f"Saved models   : {MODEL_DIR}")

print("\nNEXT: Review the results before proceeding to Stage 6.")
print("Send me the complete terminal output and the contents of:")
print("stage5_results/stage5_summary.txt")
print("Also upload baseline_model_comparison.csv if possible.")
