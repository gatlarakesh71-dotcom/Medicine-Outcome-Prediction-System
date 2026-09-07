"""Train and evaluate XGBoost on a stratified 60/20/20 split."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from model_evaluation_utils import (
    build_preprocessor,
    evaluate_model,
    prepare_data,
    load_cleaned_dataset,
    write_metrics,
)

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
MODEL_PATH = OUTPUT_DIR / "06_xgboost_treatment_outcome_model.joblib"
METRICS_PATH = OUTPUT_DIR / "06_xgboost_treatment_outcome_metrics.txt"
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
                XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    min_child_weight=2,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
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
        "XGBoost",
        {"Training": len(X_train), "Validation": len(
            X_validation), "Testing": len(X_test)},
        results,
    )
    print(f"Saved trained pipeline: {MODEL_PATH}")
    print(f"Saved evaluation metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
