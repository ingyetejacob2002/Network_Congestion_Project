"""
STAGE 8: VALIDATION STRATEGY AND FEATURE AUDIT
Project: Machine Learning-Based Prediction of Network Traffic Congestion
Run from the project root:
    python stage8_validation.py

Reads:
    network_congestion_labeled.csv

Writes:
    stage8_results/
        stage8_summary.txt
        validation_comparison.csv
        rowwise_fold_results.csv
        timestamp_group_fold_results.csv
        engineered_feature_summary.csv
        class_feature_summary.csv
        timestamp_class_audit.csv
        confusion_matrix_rowwise_<model>.png
        confusion_matrix_grouped_<model>.png

Important:
The target labels are scenario-derived (Low/Medium/High), not independently
verified congestion ground truth. Results measure prediction of those labels,
not proven real-world congestion detection.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GroupKFold,
    cross_validate,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "network_congestion_labeled.csv"
OUTPUT_DIR = BASE_DIR / "stage8_results"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "congestion_level"
RANDOM_STATE = 42
N_SPLITS = 5

LABEL_ORDER = ["Low", "Medium", "High"]


def write_line(handle, text=""):
    print(text)
    handle.write(str(text) + "\n")


def find_timestamp_column(df):
    candidates = ["timestamp", "time_stamp", "datetime", "time"]
    for name in candidates:
        if name in df.columns:
            return name
    return None


def build_features(df):
    """Create interpretable numeric features from available raw columns."""
    features = pd.DataFrame(index=df.index)

    numeric_candidates = [
        "packet_count", "byte_count", "duration_sec", "duration_nsec"
    ]
    for col in numeric_candidates:
        if col in df.columns:
            features[col] = pd.to_numeric(df[col], errors="coerce")

    # Combine the duration fields. duration_nsec is treated as a nanosecond
    # fraction of a second, as in typical flow-stat representations.
    if "duration_sec" in features.columns:
        duration = features["duration_sec"].fillna(0.0)
        if "duration_nsec" in features.columns:
            duration = duration + features["duration_nsec"].fillna(0.0) / 1e9
        features["duration_total_sec"] = duration.clip(lower=1e-9)

    if "packet_count" in features.columns:
        features["log_packet_count"] = np.log1p(
            features["packet_count"].clip(lower=0)
        )
    if "byte_count" in features.columns:
        features["log_byte_count"] = np.log1p(
            features["byte_count"].clip(lower=0)
        )

    if "packet_count" in features.columns and "duration_total_sec" in features.columns:
        features["packet_rate"] = (
            features["packet_count"].clip(lower=0)
            / features["duration_total_sec"].clip(lower=1e-9)
        )
    if "byte_count" in features.columns and "duration_total_sec" in features.columns:
        features["byte_rate"] = (
            features["byte_count"].clip(lower=0)
            / features["duration_total_sec"].clip(lower=1e-9)
        )
    if "packet_count" in features.columns and "byte_count" in features.columns:
        features["bytes_per_packet"] = (
            features["byte_count"].clip(lower=0)
            / features["packet_count"].clip(lower=1)
        )

    # Remove raw duration components if the combined duration is available,
    # avoiding duplicate representations of the same quantity.
    for col in ["duration_sec", "duration_nsec"]:
        if col in features.columns:
            features.drop(columns=col, inplace=True)

    # Keep only numeric, non-constant features. Ports are included only when
    # present and numeric; IDs, timestamps, MAC addresses, and labels are excluded.
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.loc[:, features.nunique(dropna=True) > 1]
    return features


def make_models():
    # Scaling is fitted independently inside each CV fold by the Pipeline.
    return {
        "Majority Baseline": DummyClassifier(strategy="most_frequent"),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=5, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=12, min_samples_leaf=3,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=250, max_depth=12, min_samples_leaf=3,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", class_weight="balanced"))
        ]),
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE
            ))
        ]),
        "KNN": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=7))
        ]),
    }


def as_pipeline(model):
    # Ensure every model has missing-value handling. Scaling remains within CV.
    if isinstance(model, Pipeline):
        return model
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def save_confusion_plot(y_true, y_pred, labels, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    display.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    if not DATA_FILE.exists():
        print(f"ERROR: Cannot find dataset: {DATA_FILE}")
        print("Put network_congestion_labeled.csv in the same folder as this script.")
        return

    df = pd.read_csv(DATA_FILE)
    if TARGET not in df.columns:
        print(f"ERROR: Required target column '{TARGET}' was not found.")
        print("Columns found:", list(df.columns))
        return

    df = df.dropna(subset=[TARGET]).copy()
    y = df[TARGET].astype(str).str.strip()
    timestamp_col = find_timestamp_column(df)

    # Build numeric features and remove rows with no usable feature values.
    X = build_features(df)
    valid = X.notna().any(axis=1)
    X = X.loc[valid].copy()
    y = y.loc[valid].copy()
    df_valid = df.loc[valid].copy()

    # Imputation is performed within each fold. Remove entirely empty features.
    X = X.loc[:, X.notna().any(axis=0)]
    if X.shape[1] == 0:
        print("ERROR: No usable numeric features were found.")
        return

    # Group identifiers are timestamps. If timestamp is absent, use a row-wise
    # fallback and clearly mark the limitation in the report.
    if timestamp_col:
        groups = df_valid[timestamp_col].astype(str).fillna("MISSING_TIMESTAMP")
        grouping_note = f"Timestamp grouping used: {timestamp_col}"
    else:
        groups = pd.Series(np.arange(len(df_valid)), index=df_valid.index).astype(str)
        grouping_note = (
            "WARNING: No timestamp column found. Grouped validation falls back "
            "to one group per row and therefore cannot prevent timestamp leakage."
        )

    class_counts = y.value_counts().sort_index()
    minimum_class_count = int(class_counts.min())
    splits = min(N_SPLITS, minimum_class_count)
    if splits < 2:
        print("ERROR: Not enough examples in each class for cross-validation.")
        return

    # Prefer stratified grouped CV, which tries to preserve class proportions
    # while keeping each timestamp wholly in one fold.
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        grouped_cv = StratifiedGroupKFold(
            n_splits=splits, shuffle=True, random_state=RANDOM_STATE
        )
        grouped_cv_name = "StratifiedGroupKFold by timestamp"
    except ImportError:
        grouped_cv = GroupKFold(n_splits=splits)
        grouped_cv_name = "GroupKFold by timestamp (sklearn lacks StratifiedGroupKFold)"

    rowwise_cv = StratifiedKFold(
        n_splits=splits, shuffle=True, random_state=RANDOM_STATE
    )

    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }

    print("=" * 72)
    print("STAGE 8: VALIDATION STRATEGY AND FEATURE AUDIT")
    print("=" * 72)
    print(f"Dataset: {DATA_FILE}")
    print(f"Rows: {len(df_valid)}")
    print(f"Numeric features: {list(X.columns)}")
    print(f"Classes: {dict(class_counts)}")
    print(grouping_note)
    print("Important: target labels are source-scenario-derived, not independent ground truth.")

    # Dataset summaries
    feature_summary = X.describe().T
    feature_summary.to_csv(OUTPUT_DIR / "engineered_feature_summary.csv")

    class_summary = X.assign(congestion_level=y).groupby("congestion_level").agg(
        ["mean", "median", "std"]
    )
    class_summary.to_csv(OUTPUT_DIR / "class_feature_summary.csv")

    if timestamp_col:
        timestamp_audit = (
            df_valid.assign(_target=y.values)
            .groupby(timestamp_col)
            .agg(
                row_count=("_target", "size"),
                number_of_classes=("_target", "nunique"),
                classes=("_target", lambda values: ", ".join(sorted(set(values)))),
            )
            .reset_index()
        )
        timestamp_audit.to_csv(OUTPUT_DIR / "timestamp_class_audit.csv", index=False)

    results = []
    fold_rows = []
    fitted_models = make_models()

    for model_name, base_model in fitted_models.items():
        estimator = as_pipeline(base_model)
        print(f"\nEvaluating: {model_name}")

        for validation_name, cv in [
            ("Row-wise stratified CV", rowwise_cv),
            (grouped_cv_name, grouped_cv),
        ]:
            try:
                cv_results = cross_validate(
                    estimator,
                    X,
                    y,
                    cv=cv,
                    groups=groups if "Group" in type(cv).__name__ else None,
                    scoring=scoring,
                    return_train_score=False,
                    n_jobs=1,
                    error_score="raise",
                )
                row = {
                    "model": model_name,
                    "validation": validation_name,
                    "folds": splits,
                }
                for metric in scoring:
                    vals = cv_results[f"test_{metric}"]
                    row[f"{metric}_mean"] = float(np.mean(vals))
                    row[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                results.append(row)

                for fold_index in range(splits):
                    fold_entry = {
                        "model": model_name,
                        "validation": validation_name,
                        "fold": fold_index + 1,
                    }
                    for metric in scoring:
                        fold_entry[metric] = float(cv_results[f"test_{metric}"][fold_index])
                    fold_rows.append(fold_entry)

                print(
                    f"  {validation_name}: "
                    f"accuracy={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f}, "
                    f"macro-F1={row['f1_macro_mean']:.4f}±{row['f1_macro_std']:.4f}"
                )
            except Exception as exc:
                print(f"  ERROR during {validation_name}: {exc}")
                results.append({
                    "model": model_name,
                    "validation": validation_name,
                    "folds": splits,
                    "error": str(exc),
                })

    comparison = pd.DataFrame(results)
    comparison.to_csv(OUTPUT_DIR / "validation_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUTPUT_DIR / "validation_fold_results.csv", index=False)

    # Produce out-of-fold confusion matrices and reports for the two validation
    # strategies using selected transparent models. These are diagnostics only.
    for model_name in ["Decision Tree", "Random Forest", "SVM", "Logistic Regression"]:
        base_model = fitted_models[model_name]
        estimator = as_pipeline(base_model)
        for validation_name, cv, groups_arg, suffix in [
            ("rowwise", rowwise_cv, None, "rowwise"),
            (
                "grouped",
                grouped_cv,
                groups if "Group" in type(grouped_cv).__name__ else None,
                "grouped",
            ),
        ]:
            try:
                predictions = cross_val_predict(
                    estimator, X, y, cv=cv, groups=groups_arg, n_jobs=1
                )
                report = classification_report(y, predictions, zero_division=0)
                with open(OUTPUT_DIR / f"report_{suffix}_{model_name.lower().replace(' ', '_')}.txt",
                          "w", encoding="utf-8") as f:
                    f.write(f"Model: {model_name}\nValidation: {validation_name}\n\n")
                    f.write(report)
                save_confusion_plot(
                    y, predictions, labels=sorted(y.unique()),
                    title=f"{model_name} - {validation_name} CV",
                    path=OUTPUT_DIR / f"confusion_matrix_{suffix}_{model_name.lower().replace(' ', '_')}.png",
                )
            except Exception as exc:
                print(f"Could not create {validation_name} confusion matrix for {model_name}: {exc}")

    # Check label by timestamp and scenario-time separation.
    timestamp_ranges = None
    if timestamp_col:
        tmp = df_valid.copy()
        tmp["_target"] = y.values
        tmp[timestamp_col] = pd.to_datetime(tmp[timestamp_col], errors="coerce", utc=True)
        timestamp_ranges = tmp.groupby("_target")[timestamp_col].agg(["min", "max", "nunique"])
        timestamp_ranges.to_csv(OUTPUT_DIR / "timestamp_ranges_by_class.csv")

    summary_path = OUTPUT_DIR / "stage8_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as report:
        write_line(report, "STAGE 8: VALIDATION STRATEGY AND FEATURE AUDIT")
        write_line(report, "=" * 65)
        write_line(report, f"Rows evaluated: {len(df_valid)}")
        write_line(report, f"Feature columns: {list(X.columns)}")
        write_line(report, f"Class counts: {dict(class_counts)}")
        write_line(report, grouping_note)
        write_line(report, f"Grouped validation: {grouped_cv_name}")
        write_line(report)
        write_line(report, "LABEL/TIMESTAMP AUDIT")
        if timestamp_col:
            write_line(report, f"Timestamp column: {timestamp_col}")
            if timestamp_ranges is not None:
                write_line(report, timestamp_ranges.to_string())
            write_line(
                report,
                "Review timestamp_ranges_by_class.csv. If class time ranges do not overlap, "
                "class identity is confounded with collection period/scenario."
            )
        else:
            write_line(report, "No timestamp column found; timestamp-group validation is unavailable.")
        write_line(report)
        write_line(report, "VALIDATION COMPARISON (sorted by grouped macro-F1)")
        grouped_rows = comparison[comparison["validation"].str.contains("Group", case=False, na=False)]
        if not grouped_rows.empty and "f1_macro_mean" in grouped_rows.columns:
            write_line(report, grouped_rows.sort_values("f1_macro_mean", ascending=False).to_string(index=False))
        else:
            write_line(report, comparison.to_string(index=False))
        write_line(report)
        write_line(report, "INTERPRETATION")
        write_line(
            report,
            "Use timestamp-grouped validation as the more cautious estimate when repeated records "
            "come from the same timestamp. Row-wise CV can be optimistic if closely related records "
            "are spread across folds."
        )
        write_line(
            report,
            "A difference between row-wise and grouped scores indicates sensitivity to the validation "
            "strategy. Neither strategy proves real-world congestion detection because the target labels "
            "were assigned from source scenarios."
        )
        write_line(
            report,
            "Do not select a final model solely from accuracy. Review macro-F1, per-class reports, "
            "confusion matrices, timestamp ranges, and how the labels were created."
        )

    print("\n" + "=" * 72)
    print("STAGE 8 COMPLETED")
    print("=" * 72)
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Main summary: {summary_path}")
    print("Next, paste the terminal output and the contents of stage8_summary.txt.")


if __name__ == "__main__":
    main()
