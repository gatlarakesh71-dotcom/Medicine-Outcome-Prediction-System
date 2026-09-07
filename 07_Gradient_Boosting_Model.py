"""Train and evaluate Gradient Boosting on a stratified 60/20/20 split."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
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
MODEL_PATH = OUTPUT_DIR / "07_gradient_boosting_treatment_outcome_model.joblib"
METRICS_PATH = OUTPUT_DIR / "07_gradient_boosting_treatment_outcome_metrics.txt"
TARGET_COL = "treatment_outcome"


def main() -> None:
    df = load_cleaned_dataset(OUTPUT_DIR)
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_data(
        df, TARGET_COL)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=3,
                    max_features="sqrt",
                    subsample=0.8,
                    random_state=42,
                ),
            ),
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

    joblib.dump(pipeline, MODEL_PATH)
    write_metrics(
        METRICS_PATH,
        "Gradient Boosting",
        {"Training": len(X_train), "Validation": len(
            X_validation), "Testing": len(X_test)},
        results,
    )
    print(f"Saved trained pipeline: {MODEL_PATH}")
    print(f"Saved evaluation metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
