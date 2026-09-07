"""Train a Decision Tree Classifier on the cleaned clinical dataset."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from model_evaluation_utils import (
    build_preprocessor,
    evaluate_model,
    prepare_data,
    resolve_cleaned_dataset,
    write_metrics,
)

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
DATA_PATH = ROOT / "output" / "clinical_data_cleaned.csv"
MODEL_PATH = ROOT / "output" / "decision_tree_treatment_outcome_model.joblib"
METRICS_PATH = ROOT / "output" / "decision_tree_treatment_outcome_metrics.txt"
TARGET_COL = "treatment_outcome"


def main() -> None:
    data_path = resolve_cleaned_dataset(OUTPUT_DIR)
    df = pd.read_csv(data_path, low_memory=False)
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_data(
        df, TARGET_COL
    )
    preprocessor = build_preprocessor(X_train)

    model = DecisionTreeClassifier(
        random_state=42,
        max_depth=8,
        min_samples_leaf=100,
        class_weight="balanced",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    print("Decision Tree Classifier training summary")
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_validation)}")
    print(f"Test set size: {len(X_test)}")

    results = {
        "Training": evaluate_model(pipeline, X_train, y_train),
        "Validation": evaluate_model(pipeline, X_validation, y_validation),
        "Testing": evaluate_model(pipeline, X_test, y_test),
    }
    for split_name, metrics in results.items():
        print(f"{split_name} accuracy: {metrics['accuracy']:.4f}")

    joblib.dump(pipeline, MODEL_PATH)
    write_metrics(
        METRICS_PATH,
        "Decision Tree",
        {"Training": len(X_train), "Validation": len(
            X_validation), "Testing": len(X_test)},
        results,
    )

    print(f"\nSaved trained pipeline: {MODEL_PATH}")
    print(f"Saved evaluation metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
