"""Train a Random Forest Classifier on the cleaned clinical dataset."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
MODEL_PATH = OUTPUT_DIR / "random_forest_treatment_outcome_model.joblib"
METRICS_PATH = OUTPUT_DIR / "random_forest_treatment_outcome_metrics.txt"
TARGET_COL = "treatment_outcome"


def resolve_cleaned_dataset() -> Path:
    target_name = "02_clinical_data_cleaned.csv"
    data_path = OUTPUT_DIR / target_name
    if data_path.exists():
        return data_path
    candidates = sorted(OUTPUT_DIR.glob("*.csv"))
    preferred = [
        p for p in candidates
        if "cleaned" in p.name.lower() and "clinical" in p.name.lower()
    ]
    if preferred:
        return preferred[0]
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No cleaned CSV found in {OUTPUT_DIR}")


def main() -> None:
    DATA_PATH = resolve_cleaned_dataset()
    df = pd.read_csv(DATA_PATH, low_memory=False)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found in the cleaned data.")

    df = df.drop(columns=[col for col in [
                 "patient_id", "admission_date"] if col in df.columns], errors="ignore")
    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    X = df.drop(columns=[TARGET_COL, "adverse_event",
                "readmission_30d"], errors="ignore")
    y = df[TARGET_COL]

    numeric_cols = [
        col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["No", "Yes"])
    cm = confusion_matrix(y_test, y_pred)

    print("Random Forest Classifier training summary")
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)

    joblib.dump(pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))

    print(f"\nSaved trained pipeline: {MODEL_PATH}")
    print(f"Saved evaluation metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
