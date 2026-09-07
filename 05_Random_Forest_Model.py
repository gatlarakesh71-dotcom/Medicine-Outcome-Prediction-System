"""Train and evaluate a Random Forest on a stratified 60/20/20 split."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from model_evaluation_utils import (
    build_preprocessor,
    evaluate_model,
    prepare_data,
    load_cleaned_dataset,
    write_metrics,
)

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
MODEL_PATH = OUTPUT_DIR / "random_forest_treatment_outcome_model.joblib"
METRICS_PATH = OUTPUT_DIR / "random_forest_treatment_outcome_metrics.txt"
TARGET_COL = "treatment_outcome"


def main() -> None:
    df = load_cleaned_dataset(OUTPUT_DIR)
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_data(
        df, TARGET_COL)

    model = RandomForestClassifier(
        n_estimators=10,
        max_depth=12,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    results = {
        "Training": evaluate_model(pipeline, X_train, y_train),
        "Validation": evaluate_model(pipeline, X_validation, y_validation),
        "Testing": evaluate_model(pipeline, X_test, y_test),
    }
    for split_name, metrics in results.items():
        print(f"{split_name} accuracy: {metrics['accuracy']:.4f}")

    write_metrics(
        METRICS_PATH,
        "Random Forest",
        {"Training": len(X_train), "Validation": len(
            X_validation), "Testing": len(X_test)},
        results,
    )
    joblib.dump(pipeline, MODEL_PATH)

    print(f"\nSaved trained pipeline: {MODEL_PATH}")
    print(f"Saved evaluation metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
