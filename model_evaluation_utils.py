"""Shared data preparation, preprocessing, and evaluation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42


def resolve_cleaned_dataset(output_dir: Path) -> Path:
    """Return the available cleaned dataset using the project's naming conventions."""
    candidates = (
        output_dir / "02_clinical_data_cleaned.csv",
        output_dir / "clinical_data_cleaned.csv",
        output_dir / "03_clinical_feature_engineered.csv",
    )
    for path in candidates:
        if path.exists():
            return path
    expected = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Cleaned dataset not found. Expected one of: {expected}")


def load_cleaned_dataset(output_dir: Path) -> pd.DataFrame:
    """Load the cleaned clinical dataset."""
    return pd.read_csv(resolve_cleaned_dataset(output_dir), low_memory=False)


def prepare_data(
    df: pd.DataFrame, target_col: str
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Create reproducible stratified 60/20/20 train, validation, and test splits."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    data = df.drop(
        columns=[column for column in ["patient_id",
                                       "admission_date"] if column in df.columns]
    ).dropna(subset=[target_col]).copy()
    data[target_col] = data[target_col].astype(int)

    X = data.drop(
        columns=[target_col, "adverse_event", "readmission_30d"],
        errors="ignore",
    )
    y = data[target_col]
    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_remaining,
        y_remaining,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_remaining,
    )
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a leakage-safe imputation and encoding transformer from training columns."""
    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def evaluate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
) -> dict[str, object]:
    """Calculate the metrics used by the model training scripts."""
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "samples": len(y),
        "accuracy": accuracy_score(y, predictions),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
        "pr_auc": average_precision_score(y, probabilities),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "classification_report": classification_report(
            y, predictions, target_names=["No", "Yes"], zero_division=0
        ),
    }


def write_metrics(
    path: Path,
    model_name: str,
    split_sizes: dict[str, int],
    results: dict[str, dict[str, object]],
) -> None:
    """Write human-readable evaluation metrics for each data split."""
    lines = [
        f"Model: {model_name}",
        "Split: 60% training / 20% validation / 20% testing",
        "",
        "Split sizes:",
    ]
    lines.extend(f"{name}: {size}" for name, size in split_sizes.items())

    for split_name, metrics in results.items():
        matrix = np.asarray(metrics["confusion_matrix"])
        lines.extend(
            [
                "",
                f"{split_name} metrics:",
                f"Samples: {metrics['samples']}",
                f"Accuracy: {metrics['accuracy']:.4f}",
                f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}",
                f"Precision: {metrics['precision']:.4f}",
                f"Recall: {metrics['recall']:.4f}",
                f"F1-score: {metrics['f1']:.4f}",
                f"ROC-AUC: {metrics['roc_auc']:.4f}",
                f"PR-AUC: {metrics['pr_auc']:.4f}",
                f"Specificity: {metrics['specificity']:.4f}",
                f"Confusion Matrix: {matrix.tolist()}",
                "Classification Report:",
                str(metrics["classification_report"]),
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
