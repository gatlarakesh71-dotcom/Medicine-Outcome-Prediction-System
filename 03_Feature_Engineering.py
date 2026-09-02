"""Create engineered clinical features from the cleaned dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
CLEAN_PATH = ROOT / "output" / "clinical_data_cleaned.csv"
OUT_PATH = ROOT / "output" / "clinical_feature_engineered.csv"

TARGET_COLS = ["treatment_outcome", "adverse_event", "readmission_30d"]


def age_group(age_value: float) -> str:
    if pd.isna(age_value):
        return "Unknown"
    if age_value < 30:
        return "18-29"
    if age_value < 45:
        return "30-44"
    if age_value < 60:
        return "45-59"
    if age_value < 75:
        return "60-74"
    return "75+"


def bmi_category(bmi_value: float) -> str:
    if pd.isna(bmi_value):
        return "Unknown"
    if bmi_value < 18.5:
        return "Underweight"
    if bmi_value < 25:
        return "Normal"
    if bmi_value < 30:
        return "Overweight"
    return "Obese"


def duration_bucket(days_value: float) -> str:
    if pd.isna(days_value):
        return "Unknown"
    if days_value <= 7:
        return "1-7d"
    if days_value <= 30:
        return "8-30d"
    if days_value <= 90:
        return "31-90d"
    if days_value <= 180:
        return "91-180d"
    if days_value <= 365:
        return "181-365d"
    return "365+ d"


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Keep identifiers and target columns safe
    id_cols = ["patient_id"] if "patient_id" in df.columns else []

    # Date extraction
    if "admission_date" in df.columns:
        df["admission_date"] = pd.to_datetime(
            df["admission_date"], errors="coerce")
        df["admission_year"] = df["admission_date"].dt.year
        df["admission_month"] = df["admission_date"].dt.month
        df["admission_day_of_week"] = df["admission_date"].dt.dayofweek
        df["admission_season"] = df["admission_month"].map(
            {
                12: "Winter",
                1: "Winter",
                2: "Winter",
                3: "Spring",
                4: "Spring",
                5: "Spring",
                6: "Summer",
                7: "Summer",
                8: "Summer",
                9: "Autumn",
                10: "Autumn",
                11: "Autumn",
            }
        )
    else:
        df["admission_year"] = np.nan
        df["admission_month"] = np.nan
        df["admission_day_of_week"] = np.nan
        df["admission_season"] = "Unknown"

    # Clinical risk and grouping features
    df["age_group"] = df["age"].apply(age_group)
    df["bmi_category"] = df["bmi"].apply(bmi_category)
    df["duration_category"] = df["duration_days"].apply(duration_bucket)

    df["obesity_flag"] = (df["bmi"] >= 30).fillna(False).astype(int)
    df["underweight_flag"] = (df["bmi"] < 18.5).fillna(False).astype(int)
    df["hypertension_flag"] = ((df["systolic_bp"] >= 140) | (
        df["diastolic_bp"] >= 90)).fillna(False).astype(int)
    df["tachycardia_flag"] = (df["heart_rate"] > 100).fillna(False).astype(int)
    df["fever_flag"] = (df["temperature_f"] > 100.4).fillna(False).astype(int)
    df["anemia_flag"] = (df["hemoglobin"] < 12).fillna(False).astype(int)
    df["high_cholesterol_flag"] = (
        df["total_cholesterol"] >= 200).fillna(False).astype(int)
    df["diabetes_risk_flag"] = (df["hba1c"] >= 6.5).fillna(False).astype(int)
    df["renal_risk_flag"] = (df["creatinine"] > 1.2).fillna(False).astype(int)
    df["polypharmacy_flag"] = (
        df["concurrent_drugs"] >= 5).fillna(False).astype(int)
    df["smoking_or_alcohol_flag"] = (
        (df["smoking_status"].fillna("Unknown").str.lower() == "current")
        | (df["alcohol_use"].fillna("Unknown").str.lower().isin(["heavy", "yes"]))
    ).astype(int)

    # Lab abnormality count
    abnormal_checks = [
        "hemoglobin",
        "wbc_count",
        "alt",
        "ast",
        "creatinine",
        "egfr",
        "hba1c",
        "total_cholesterol",
    ]
    abnormal_map = {
        "hemoglobin": lambda x: x < 12 if pd.notna(x) else False,
        "wbc_count": lambda x: x > 11000 if pd.notna(x) else False,
        "alt": lambda x: x > 40 if pd.notna(x) else False,
        "ast": lambda x: x > 40 if pd.notna(x) else False,
        "creatinine": lambda x: x > 1.2 if pd.notna(x) else False,
        "egfr": lambda x: x < 60 if pd.notna(x) else False,
        "hba1c": lambda x: x >= 6.5 if pd.notna(x) else False,
        "total_cholesterol": lambda x: x >= 200 if pd.notna(x) else False,
    }

    abnormal_count = pd.Series(0, index=df.index, dtype="int64")
    for col, rule in abnormal_map.items():
        abnormal_count += df[col].apply(rule).fillna(False).astype(int)
    df["lab_abnormality_count"] = abnormal_count

    # Numeric columns: fill missing values with median to keep model-ready
    numeric_cols = [
        "age",
        "weight_kg",
        "height_cm",
        "bmi",
        "systolic_bp",
        "diastolic_bp",
        "heart_rate",
        "temperature_f",
        "hemoglobin",
        "wbc_count",
        "alt",
        "ast",
        "creatinine",
        "egfr",
        "hba1c",
        "total_cholesterol",
        "dosage_mg",
        "duration_days",
        "concurrent_drugs",
        "admission_year",
        "admission_month",
        "admission_day_of_week",
    ]
    for col in numeric_cols:
        if col in df.columns:
            med = df[col].median()
            df[col] = df[col].fillna(med)

    # Categorical columns: fill missing values with 'Unknown'
    cat_cols = [
        "gender",
        "ethnicity",
        "drug_name",
        "route",
        "diagnosis",
        "smoking_status",
        "alcohol_use",
        "age_group",
        "bmi_category",
        "duration_category",
        "admission_season",
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Keep target columns in final frame if present
    final_cols = id_cols + [
        "age",
        "gender",
        "ethnicity",
        "weight_kg",
        "height_cm",
        "bmi",
        "systolic_bp",
        "diastolic_bp",
        "heart_rate",
        "temperature_f",
        "hemoglobin",
        "wbc_count",
        "alt",
        "ast",
        "creatinine",
        "egfr",
        "hba1c",
        "total_cholesterol",
        "drug_name",
        "dosage_mg",
        "duration_days",
        "route",
        "concurrent_drugs",
        "diagnosis",
        "smoking_status",
        "alcohol_use",
        "admission_year",
        "admission_month",
        "admission_day_of_week",
        "admission_season",
        "age_group",
        "bmi_category",
        "duration_category",
        "obesity_flag",
        "underweight_flag",
        "hypertension_flag",
        "tachycardia_flag",
        "fever_flag",
        "anemia_flag",
        "high_cholesterol_flag",
        "diabetes_risk_flag",
        "renal_risk_flag",
        "polypharmacy_flag",
        "smoking_or_alcohol_flag",
        "lab_abnormality_count",
    ] + TARGET_COLS

    final_cols = [c for c in final_cols if c in df.columns]
    features_df = df[final_cols].copy()

    # One-hot encode only the most informative categorical variables
    encode_cols = [
        "gender",
        "ethnicity",
        "diagnosis",
        "smoking_status",
        "alcohol_use",
        "route",
        "age_group",
        "bmi_category",
        "duration_category",
        "admission_season",
    ]
    encode_cols = [c for c in encode_cols if c in features_df.columns]

    encoded = pd.get_dummies(
        features_df[encode_cols], prefix=encode_cols, drop_first=False)
    feature_matrix = pd.concat(
        [features_df.drop(columns=encode_cols, errors="ignore"), encoded],
        axis=1,
    )

    # Keep patient_id in front if available
    if "patient_id" in feature_matrix.columns:
        feature_matrix = feature_matrix[[
            "patient_id"] + [c for c in feature_matrix.columns if c != "patient_id"]]

    return feature_matrix


def main() -> None:
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"Cleaned data not found: {CLEAN_PATH}")

    df = pd.read_csv(CLEAN_PATH, low_memory=False)
    feature_df = build_feature_frame(df)
    feature_df.to_csv(OUT_PATH, index=False)

    print(f"Loaded cleaned data: {df.shape}")
    print(f"Engineered features saved to: {OUT_PATH}")
    print(f"Output shape: {feature_df.shape}")
    print("Feature columns sample:")
    print(feature_df.columns[:20].tolist())


if __name__ == "__main__":
    main()
