"""Clean clinical_data_raw.csv into analysis-ready clinical_data_cleaned.csv."""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "Medicine Outcome Prediction" / "clinical_data_raw.csv"
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "clinical_data_cleaned.csv"

PLACEHOLDERS = {
    "",
    " ",
    "nan",
    "none",
    "null",
    "n/a",
    "na",
    "unknown",
    "-",
    "--",
    ".",
    "prefer not to say",
}

DOSAGE_TEXT = {
    "50mg": 50.0,
    "100 mg": 100.0,
    "250mg daily": 250.0,
    "2x daily 50mg": 50.0,
    "1/2 tab": 0.5,
    "two tablets": np.nan,
    "unknown": np.nan,
    "n/a": np.nan,
    "as needed": np.nan,
    "prn": np.nan,
    "0": np.nan,
    "-1": np.nan,
}

DURATION_TEXT = {
    "ongoing": 365.0,
    "chronic": 365.0,
    "lifetime": 365.0,
    "n/a": np.nan,
}

GENDER_MAP = {
    "male": "Male",
    "m": "Male",
    "mal": "Male",
    "female": "Female",
    "f": "Female",
    "femal": "Female",
    "other": "Other",
    "o": "Other",
}

ETHNICITY_MAP = {
    "caucasian": "White",
    "white": "White",
    "african american": "African American",
    "black": "African American",
    "aa": "African American",
    "hispanic": "Hispanic",
    "latino": "Hispanic",
    "latina": "Hispanic",
    "asian": "Asian",
    "chinese": "Asian",
    "indian": "Asian",
    "japanese": "Asian",
    "native american": "Native American",
    "pacific islander": "Pacific Islander",
    "mixed": "Mixed",
    "other": "Other",
}

DIAGNOSIS_MAP = {
    "type 2 diabetes": "Type 2 Diabetes",
    "dm2": "Type 2 Diabetes",
    "t2dm": "Type 2 Diabetes",
    "hypertension": "Hypertension",
    "htn": "Hypertension",
    "hyperlipidemia": "Hyperlipidemia",
    "hld": "Hyperlipidemia",
    "asthma": "Asthma",
    "copd": "COPD",
    "depression": "Depression",
    "anxiety": "Anxiety",
    "osteoarthritis": "Osteoarthritis",
    "oa": "Osteoarthritis",
    "heart failure": "Heart Failure",
    "chf": "Heart Failure",
    "atrial fibrillation": "Atrial Fibrillation",
    "afib": "Atrial Fibrillation",
    "chronic kidney disease": "Chronic Kidney Disease",
    "ckd": "Chronic Kidney Disease",
    "hypothyroidism": "Hypothyroidism",
    "gerd": "GERD",
    "migraine": "Migraine",
    "epilepsy": "Epilepsy",
    "rheumatoid arthritis": "Rheumatoid Arthritis",
    "osteoporosis": "Osteoporosis",
    "dvt": "DVT",
    "pneumonia": "Pneumonia",
    "uti": "UTI",
}

DRUG_MAP = {
    "metformin": "Metformin",
    "metformine": "Metformin",
    "met formin": "Metformin",
    "lisinopril": "Lisinopril",
    "lisinipril": "Lisinopril",
    "lisino pril": "Lisinopril",
    "atorvastatin": "Atorvastatin",
    "atorvastain": "Atorvastatin",
    "atorvastatine": "Atorvastatin",
    "amoxicillin": "Amoxicillin",
    "amoxicilin": "Amoxicillin",
    "amoxycillin": "Amoxicillin",
    "omeprazole": "Omeprazole",
    "omeprazol": "Omeprazole",
    "losartan": "Losartan",
    "amlodipine": "Amlodipine",
    "metoprolol": "Metoprolol",
    "simvastatin": "Simvastatin",
    "levothyroxine": "Levothyroxine",
    "azithromycin": "Azithromycin",
    "hydrochlorothiazide": "Hydrochlorothiazide",
    "ibuprofen": "Ibuprofen",
    "ibuprophen": "Ibuprofen",
    "ibuprofin": "Ibuprofen",
    "gabapentin": "Gabapentin",
    "gabapentine": "Gabapentin",
    "sertraline": "Sertraline",
    "tramadol": "Tramadol",
    "tramadole": "Tramadol",
    "prednisone": "Prednisone",
    "furosemide": "Furosemide",
    "pantoprazole": "Pantoprazole",
    "clopidogrel": "Clopidogrel",
    "montelukast": "Montelukast",
    "escitalopram": "Escitalopram",
    "duloxetine": "Duloxetine",
    "rosuvastatin": "Rosuvastatin",
    "warfarin": "Warfarin",
    "warfarine": "Warfarin",
    "insulin glargine": "Insulin Glargine",
    "albuterol": "Albuterol",
    "fluticasone": "Fluticasone",
    "ciprofloxacin": "Ciprofloxacin",
    "doxycycline": "Doxycycline",
}

ROUTE_MAP = {
    "oral": "Oral",
    "intravenous": "Intravenous",
    "iv": "Intravenous",
    "topical": "Topical",
    "subcutaneous": "Subcutaneous",
    "sc": "Subcutaneous",
    "intramuscular": "Intramuscular",
    "im": "Intramuscular",
    "inhaled": "Inhaled",
    "rectal": "Rectal",
    "ophthalmic": "Ophthalmic",
}

SMOKING_MAP = {
    "current": "Current",
    "yes": "Current",
    "y": "Current",
    "1": "Current",
    "true": "Current",
    "former": "Former",
    "ex-smoker": "Former",
    "ex smoker": "Former",
    "never": "Never",
    "no": "Never",
    "n": "Never",
    "0": "Never",
    "false": "Never",
}

ALCOHOL_MAP = {
    "none": "None",
    "no": "None",
    "social": "Occasional",
    "occasional": "Occasional",
    "1-2 drinks/week": "Occasional",
    "heavy": "Heavy",
    "daily": "Heavy",
    "yes": "Yes",
}

OUTCOME_MAP = {
    "1": 1,
    "yes": 1,
    "effective": 1,
    "true": 1,
    "0": 0,
    "no": 0,
    "ineffective": 0,
    "false": 0,
}

ADVERSE_MAP = {
    "1": 1,
    "yes": 1,
    "true": 1,
    "0": 0,
    "no": 0,
    "false": 0,
}


def log(msg: str) -> None:
    print(msg, flush=True)


def normalize_key(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.lower()
    )


def to_missing(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    key = s.str.lower()
    return s.mask(key.isin(PLACEHOLDERS) | s.isna(), pd.NA)


def map_lookup(series: pd.Series, mapping: dict) -> pd.Series:
    key = normalize_key(to_missing(series))
    mapped = key.map(mapping)
    return mapped.astype("string")


def coalesce_series(*cols: pd.Series) -> pd.Series:
    out = cols[0]
    for col in cols[1:]:
        out = out.where(out.notna() & (out.astype("string").str.strip() != ""), col)
        out = out.mask(normalize_key(out).isin(PLACEHOLDERS), col)
    return out


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def clip_valid(series: pd.Series, low: float, high: float) -> pd.Series:
    num = to_numeric(series)
    return num.where(num.between(low, high))


def parse_dosage(series: pd.Series) -> pd.Series:
    raw = to_missing(series)
    key = normalize_key(raw)
    mapped = key.map(DOSAGE_TEXT)
    numeric = to_numeric(raw)
    extracted = raw.str.extract(r"(\d+(?:\.\d+)?)", expand=False)
    extracted_num = to_numeric(extracted)
    out = mapped.fillna(numeric).fillna(extracted_num)
    out = out.where(out > 0)
    return out.astype("float64")


def parse_duration(series: pd.Series) -> pd.Series:
    raw = to_missing(series)
    key = normalize_key(raw)
    mapped = key.map(DURATION_TEXT)
    numeric = to_numeric(raw)
    out = mapped.fillna(numeric)
    out = out.where(out.between(1, 730))
    return out.astype("float64")


def parse_binary(series: pd.Series, mapping: dict) -> pd.Series:
    key = normalize_key(to_missing(series))
    mapped = key.map(mapping)
    numeric = to_numeric(series)
    numeric = numeric.where(numeric.isin([0, 1]))
    out = mapped.fillna(numeric)
    return out.astype("float64")


def parse_admission_dates(series: pd.Series) -> pd.Series:
    raw = to_missing(series)
    s = raw.astype("string").str.strip()
    year_only = s.str.fullmatch(r"\d{4}")

    parsed = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    for fmt in (
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%m-%d-%Y",
        "%d-%b-%Y",
        "%B %d, %Y",
        "%d.%m.%Y",
        "%d-%m-%Y",
    ):
        parsed = parsed.fillna(pd.to_datetime(s, format=fmt, errors="coerce"))

    parsed = parsed.fillna(pd.to_datetime(s.where(year_only), format="%Y", errors="coerce"))
    parsed = parsed.where(parsed.dt.year.between(2018, 2025))
    return parsed.dt.strftime("%Y-%m-%d").astype("string")


def parse_blood_pressure(bp: pd.Series, systolic: pd.Series, diastolic: pd.Series) -> tuple[pd.Series, pd.Series]:
    text = to_missing(bp).astype("string")
    parts = text.str.extract(r"^(-?\d+)\s*[\\/]\s*(-?\d+)$")
    sys_from_text = to_numeric(parts[0])
    dia_from_text = to_numeric(parts[1])
    sys_only = to_numeric(text.where(text.str.fullmatch(r"-?\d+")))

    sys_out = sys_from_text.fillna(to_numeric(systolic)).fillna(sys_only)
    dia_out = dia_from_text.fillna(to_numeric(diastolic))

    sys_out = sys_out.where(sys_out.between(60, 250))
    dia_out = dia_out.where(dia_out.between(30, 150))
    invalid_pair = sys_out.notna() & dia_out.notna() & (sys_out <= dia_out)
    sys_out = sys_out.mask(invalid_pair)
    dia_out = dia_out.mask(invalid_pair)
    return sys_out.astype("float64"), dia_out.astype("float64")


def clean_height(series: pd.Series) -> pd.Series:
    height = to_numeric(series)
    feet = height.between(4.5, 8.0)
    height = height.mask(feet, height * 30.48)
    return height.where(height.between(120, 230))


def clean_weight(kg: pd.Series, lbs: pd.Series) -> pd.Series:
    weight_kg = clip_valid(kg, 30, 250)
    from_lbs = clip_valid(to_numeric(lbs) / 2.20462, 30, 250)
    return weight_kg.fillna(from_lbs)


def main() -> int:
    t0 = time.time()
    log(f"Loading {RAW_PATH} ...")
    df = pd.read_csv(RAW_PATH, low_memory=False)
    n_raw, p_raw = df.shape
    log(f"Raw shape: {n_raw:,} rows x {p_raw} columns")

    df.columns = [c.strip() for c in df.columns]

    # Duplicate / junk columns from the generator
    age = coalesce_series(df["Age"], df["age"])
    gender = coalesce_series(df["Gender"], df["gender"])
    drug = coalesce_series(df["Drug_Name"], df["drug_name"])

    systolic, diastolic = parse_blood_pressure(
        df["blood_pressure"], df["Systolic_BP"], df["Diastolic_BP"]
    )
    height_cm = clean_height(df["Height_cm"])
    weight_kg = clean_weight(df["Weight_kg"], df["Weight_lbs"])
    bmi = (weight_kg / (height_cm / 100.0) ** 2).round(1)
    bmi = bmi.where(bmi.between(12, 60))

    cleaned = pd.DataFrame(
        {
            "patient_id": to_numeric(df["Patient_ID"]).astype("Int64"),
            "age": clip_valid(age, 1, 120).round(0).astype("Int64"),
            "gender": map_lookup(gender, GENDER_MAP),
            "ethnicity": map_lookup(df["Ethnicity"], ETHNICITY_MAP),
            "weight_kg": weight_kg.round(1),
            "height_cm": height_cm.round(1),
            "bmi": bmi,
            "systolic_bp": systolic.round(0).astype("Int64"),
            "diastolic_bp": diastolic.round(0).astype("Int64"),
            "heart_rate": clip_valid(df["Heart_Rate"], 30, 220).round(0).astype("Int64"),
            "temperature_f": clip_valid(df["Temperature_F"], 95, 108).round(1),
            "hemoglobin": clip_valid(df["Hemoglobin"], 5, 22).round(1),
            "wbc_count": clip_valid(df["WBC_Count"], 1000, 30000).round(0).astype("Int64"),
            "alt": clip_valid(df["ALT_Enzyme"], 1, 1000).round(1),
            "ast": clip_valid(df["AST_Enzyme"], 1, 1000).round(1),
            "creatinine": clip_valid(df["Creatinine"], 0.2, 15).round(2),
            "egfr": clip_valid(df["eGFR"], 5, 150).round(1),
            "hba1c": clip_valid(df["HbA1c"], 3.5, 18).round(1),
            "total_cholesterol": clip_valid(df["Total_Cholesterol"], 80, 450).round(0).astype("Int64"),
            "drug_name": map_lookup(drug, DRUG_MAP),
            "dosage_mg": parse_dosage(df["Dosage"]).round(1),
            "duration_days": parse_duration(df["Duration_Days"]).round(0).astype("Int64"),
            "route": map_lookup(df["Route"], ROUTE_MAP),
            "concurrent_drugs": clip_valid(df["Concurrent_Drugs"], 0, 20).round(0).astype("Int64"),
            "diagnosis": map_lookup(df["Diagnosis"], DIAGNOSIS_MAP),
            "smoking_status": map_lookup(df["Smoking_Status"], SMOKING_MAP),
            "alcohol_use": map_lookup(df["Alcohol_Use"], ALCOHOL_MAP),
            "admission_date": parse_admission_dates(df["Admission_Date"]),
            "treatment_outcome": parse_binary(df["Treatment_Outcome"], OUTCOME_MAP).astype("Int64"),
            "adverse_event": parse_binary(df["Adverse_Event"], ADVERSE_MAP).astype("Int64"),
            "readmission_30d": parse_binary(df["Readmission_30d"], OUTCOME_MAP).astype("Int64"),
        }
    )

    before_dedup = len(cleaned)
    exact_dupes = int(cleaned.duplicated().sum())
    cleaned["_missing"] = cleaned.isna().sum(axis=1)
    cleaned = (
        cleaned.sort_values(["patient_id", "_missing"], kind="mergesort")
        .drop_duplicates(subset=["patient_id"], keep="first")
        .drop(columns=["_missing"])
        .sort_values("patient_id", kind="mergesort")
        .reset_index(drop=True)
    )

    cleaned.to_csv(OUT_PATH, index=False)

    elapsed = time.time() - t0
    n_clean, p_clean = cleaned.shape
    missing_pct = (cleaned.isna().mean() * 100).round(2)

    log("")
    log("=" * 64)
    log("CLEANING SUMMARY")
    log("=" * 64)
    log(f"Input:              {RAW_PATH}")
    log(f"Output:             {OUT_PATH}")
    log(f"Raw rows/cols:      {n_raw:,} / {p_raw}")
    log(f"Exact duplicate rows before patient dedupe: {exact_dupes:,}")
    log(f"Rows dropped (dupes / extra patient copies): {before_dedup - n_clean:,}")
    log(f"Clean rows/cols:    {n_clean:,} / {p_clean}")
    log(f"File size:          {os.path.getsize(OUT_PATH) / (1024 * 1024):.1f} MB")
    log(f"Elapsed:            {elapsed:.1f}s")
    log("")
    log("Missingness after cleaning (%):")
    for col, pct in missing_pct.items():
        log(f"  {col:22s} {pct:6.2f}")
    log("")
    log("Value counts (top categories):")
    for col in (
        "gender",
        "ethnicity",
        "drug_name",
        "route",
        "diagnosis",
        "smoking_status",
        "alcohol_use",
        "treatment_outcome",
        "adverse_event",
        "readmission_30d",
    ):
        log(f"\n  {col}:")
        counts = cleaned[col].value_counts(dropna=False).head(12)
        for val, n in counts.items():
            log(f"    {val}: {n:,}")
    log("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
