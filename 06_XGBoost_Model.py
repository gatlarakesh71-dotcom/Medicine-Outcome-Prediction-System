"""Train and evaluate XGBoost on a stratified 60/20/20 split."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve
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

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos

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
                    eval_metric="aucpr",
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    validation_probabilities = pipeline.predict_proba(X_validation)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(
        y_validation, validation_probabilities
    )
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (
        precisions[:-1] + recalls[:-1] + 1e-9
    )
    best_index = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_index])
    print(
        f"Best validation threshold: {best_threshold:.3f}, "
        f"F1: {f1_scores[best_index]:.3f}"
    )

    results = {
        "Training": evaluate_model(
            pipeline, X_train, y_train, threshold=best_threshold
        ),
        "Validation": evaluate_model(
            pipeline, X_validation, y_validation, threshold=best_threshold
        ),
        "Testing": evaluate_model(
            pipeline, X_test, y_test, threshold=best_threshold
        ),
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
