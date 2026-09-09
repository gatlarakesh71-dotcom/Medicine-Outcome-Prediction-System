"""Run clinical SQL analysis against the cleaned treatment-outcome dataset."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "output" / "02_clinical_data_cleaned.csv"
RESULTS_DIR = ROOT / "output" / "13_clinical_sql_results"
DATABASE_PATH = RESULTS_DIR / "clinical_data.sqlite"
TABLE_NAME = "clinical_data"
CHUNK_SIZE = 100_000


QUERIES = {
    "Q1_Overall_Treatment_Outcomes": (
        "Q1: Overall Treatment Outcomes",
        """
        SELECT
            treatment_outcome,
            COUNT(*) AS patient_count,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS percentage
        FROM clinical_data
        WHERE treatment_outcome IS NOT NULL
        GROUP BY treatment_outcome
        ORDER BY treatment_outcome;
        """,
    ),
    "Q2_Efficacy_by_Age_Group": (
        "Q2: Efficacy by Age Group",
        """
        WITH age_groups AS (
            SELECT
                CASE
                    WHEN age < 18 THEN 'Under 18'
                    WHEN age BETWEEN 18 AND 39 THEN '18-39'
                    WHEN age BETWEEN 40 AND 64 THEN '40-64'
                    WHEN age >= 65 THEN '65+'
                    ELSE 'Unknown'
                END AS age_group,
                treatment_outcome
            FROM clinical_data
        )
        SELECT
            age_group,
            COUNT(*) AS patient_count,
            ROUND(AVG(treatment_outcome), 4) AS average_outcome,
            ROUND(100.0 * AVG(treatment_outcome), 2) AS efficacy_rate_percent
        FROM age_groups
        WHERE age_group <> 'Unknown' AND treatment_outcome IS NOT NULL
        GROUP BY age_group
        ORDER BY CASE age_group
            WHEN 'Under 18' THEN 1
            WHEN '18-39' THEN 2
            WHEN '40-64' THEN 3
            WHEN '65+' THEN 4
        END;
        """,
    ),
    "Q3_Drug_Efficacy_Ranking": (
        "Q3: Drug Efficacy Ranking",
        """
        SELECT
            drug_name,
            COUNT(*) AS patient_count,
            SUM(CASE WHEN treatment_outcome = 1 THEN 1 ELSE 0 END) AS successful_outcomes,
            ROUND(100.0 * AVG(treatment_outcome), 2) AS efficacy_rate_percent
        FROM clinical_data
        WHERE drug_name IS NOT NULL AND treatment_outcome IS NOT NULL
        GROUP BY drug_name
        HAVING COUNT(*) >= 100
        ORDER BY efficacy_rate_percent DESC, patient_count DESC;
        """,
    ),
    "Q4_ADR_Risk_by_Drug_and_Age": (
        "Q4: ADR Risk by Drug and Age",
        """
        WITH age_groups AS (
            SELECT
                drug_name,
                CASE
                    WHEN age < 18 THEN 'Under 18'
                    WHEN age BETWEEN 18 AND 39 THEN '18-39'
                    WHEN age BETWEEN 40 AND 64 THEN '40-64'
                    WHEN age >= 65 THEN '65+'
                    ELSE 'Unknown'
                END AS age_group,
                adverse_event
            FROM clinical_data
        )
        SELECT
            drug_name,
            age_group,
            COUNT(*) AS patient_count,
            SUM(CASE WHEN adverse_event = 1 THEN 1 ELSE 0 END) AS adverse_event_count,
            ROUND(100.0 * AVG(adverse_event), 2) AS adverse_event_rate_percent
        FROM age_groups
        WHERE drug_name IS NOT NULL
          AND age_group <> 'Unknown'
          AND adverse_event IS NOT NULL
        GROUP BY drug_name, age_group
        HAVING COUNT(*) >= 50
        ORDER BY adverse_event_rate_percent DESC, patient_count DESC;
        """,
    ),
    "Q5_High_Risk_Patients": (
        "Q5: High-Risk Patients (Multiple Risk Factors)",
        """
        WITH risk_scored AS (
            SELECT
                patient_id,
                age,
                diagnosis,
                drug_name,
                concurrent_drugs,
                egfr,
                adverse_event,
                readmission_30d,
                (
                    CASE WHEN age >= 65 THEN 1 ELSE 0 END
                    + CASE WHEN concurrent_drugs >= 5 THEN 1 ELSE 0 END
                    + CASE WHEN egfr < 60 THEN 1 ELSE 0 END
                    + CASE WHEN adverse_event = 1 THEN 1 ELSE 0 END
                    + CASE WHEN readmission_30d = 1 THEN 1 ELSE 0 END
                ) AS risk_factor_count
            FROM clinical_data
        )
        SELECT
            patient_id,
            age,
            diagnosis,
            drug_name,
            concurrent_drugs,
            egfr,
            adverse_event,
            readmission_30d,
            risk_factor_count
        FROM risk_scored
        WHERE risk_factor_count >= 3
        ORDER BY risk_factor_count DESC, age DESC, patient_id;
        """,
    ),
    "Q6_Outcome_vs_Dosage_Level": (
        "Q6: Outcome vs Dosage Level",
        """
        WITH dosage_groups AS (
            SELECT
                CASE
                    WHEN dosage_mg < 100 THEN 'Low (<100 mg)'
                    WHEN dosage_mg < 500 THEN 'Moderate (100-499 mg)'
                    WHEN dosage_mg < 1000 THEN 'High (500-999 mg)'
                    WHEN dosage_mg >= 1000 THEN 'Very high (1000+ mg)'
                    ELSE 'Unknown'
                END AS dosage_level,
                dosage_mg,
                treatment_outcome
            FROM clinical_data
        )
        SELECT
            dosage_level,
            COUNT(*) AS patient_count,
            ROUND(AVG(dosage_mg), 2) AS average_dosage_mg,
            ROUND(AVG(treatment_outcome), 4) AS average_outcome,
            ROUND(100.0 * AVG(treatment_outcome), 2) AS efficacy_rate_percent
        FROM dosage_groups
        WHERE dosage_level <> 'Unknown' AND treatment_outcome IS NOT NULL
        GROUP BY dosage_level
        ORDER BY CASE dosage_level
            WHEN 'Low (<100 mg)' THEN 1
            WHEN 'Moderate (100-499 mg)' THEN 2
            WHEN 'High (500-999 mg)' THEN 3
            WHEN 'Very high (1000+ mg)' THEN 4
        END;
        """,
    ),
}


def load_cleaned_data(connection: sqlite3.Connection) -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned data file not found: {DATA_PATH}")

    connection.execute(f'DROP TABLE IF EXISTS "{TABLE_NAME}"')
    for chunk_number, chunk in enumerate(
        pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE, low_memory=False)
    ):
        chunk.to_sql(
            TABLE_NAME,
            connection,
            if_exists="replace" if chunk_number == 0 else "append",
            index=False,
        )
    connection.execute(
        f'CREATE INDEX IF NOT EXISTS idx_clinical_drug ON "{TABLE_NAME}" (drug_name)'
    )
    connection.execute(
        f'CREATE INDEX IF NOT EXISTS idx_clinical_outcome ON "{TABLE_NAME}" (treatment_outcome)'
    )
    connection.commit()


def run_query(connection: sqlite3.Connection, title: str, query: str) -> pd.DataFrame:
    result = pd.read_sql_query(query, connection)
    print(f"\n{title}\n{'-' * len(title)}")
    print(result.head(20).to_string(index=False))
    return result


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if DATABASE_PATH.exists():
        DATABASE_PATH.unlink()

    with sqlite3.connect(DATABASE_PATH) as connection:
        load_cleaned_data(connection)
        with (RESULTS_DIR / "13_clinical_sql_results.txt").open(
            "w", encoding="utf-8"
        ) as text_file:
            for file_stem, (title, query) in QUERIES.items():
                result = run_query(connection, title, query)
                result.to_csv(RESULTS_DIR / f"{file_stem}.csv", index=False)
                text_file.write(f"{title}\n{'=' * len(title)}\n")
                text_file.write(result.to_string(index=False))
                text_file.write("\n\n")

    print(f"\nSaved SQL database and results in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
