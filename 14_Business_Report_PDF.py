"""Create a concise client-facing business report from project artifacts."""

from __future__ import annotations
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
RAW = ROOT / "Medicine Outcome Prediction Raw" / "clinical_data_raw.csv"
if not RAW.exists():
    RAW = ROOT / "Medicine Outcome Prediction" / "clinical_data_raw.csv"
CLEAN = OUTPUT / "02_clinical_data_cleaned.csv"
SQL = OUTPUT / "13_clinical_sql_results"
RESULTS = OUTPUT / "model_comparison_detailed_results.csv"
REPORT = OUTPUT / "medicine_outcome_business_report.pdf"

NAVY = "#12263A"
TEAL = "#087E8B"
BLUE = "#2F6690"
ORANGE = "#F4A261"
RED = "#C94C4C"
INK = "#243447"
MUTED = "#5C6B73"
PALE = "#F5F8FA"
MODEL_COLORS = {
    "Decision Tree": "#2F6690", "Random Forest": "#3A9D7C",
    "XGBoost": "#F4A261", "Gradient Boosting": "#C94C4C",
    "Neural Network": "#6C5B7B", "LSTM": "#238A8D",
}


def page(title: str, subtitle: str = "") -> plt.Figure:
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    fig.text(0.075, 0.955, title, fontsize=19,
             weight="bold", color=NAVY, va="top")
    if subtitle:
        fig.text(0.075, 0.925, subtitle, fontsize=9.5, color=MUTED, va="top")
    fig.lines.append(plt.Line2D([0.075, 0.925], [
                     0.905, 0.905], transform=fig.transFigure, color=TEAL, linewidth=2))
    return fig


def paragraph(fig: plt.Figure, text: str, x: float, y: float, width: int = 88,
              size: float = 10, color: str = INK, line: float = 1.35) -> None:
    fig.text(x, y, textwrap.fill(text, width=width, break_long_words=False),
             fontsize=size, color=color, va="top", linespacing=line)


def section(fig: plt.Figure, title: str, x: float, y: float, color: str = TEAL) -> None:
    fig.text(x, y, title.upper(), fontsize=9,
             weight="bold", color=color, va="top")


def metric(fig: plt.Figure, x: float, y: float, value: str, label: str, color: str = TEAL) -> None:
    fig.text(x, y, value, fontsize=22, weight="bold", color=color, ha="center")
    fig.text(x, y - 0.027, label, fontsize=8.3, color=MUTED, ha="center")


def table(fig: plt.Figure, box: list[float], headers: list[str], rows: list[list[str]], widths: list[float], size: float = 7.2) -> None:
    ax = fig.add_axes(box)
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=headers,
                 cellLoc="center", loc="center", colWidths=widths)
    t.auto_set_font_size(False)
    t.set_fontsize(size)
    t.scale(1, 1.55)
    for (r, _), cell in t.get_celld().items():
        cell.set_edgecolor("#D9E2E8")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor(PALE if r % 2 == 0 else "white")
            cell.set_text_props(color=INK)


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    raw = pd.read_csv(RAW, low_memory=False)
    clean = pd.read_csv(CLEAN, low_memory=False)
    results = pd.read_csv(RESULTS)
    age = pd.read_csv(SQL / "Q2_Efficacy_by_Age_Group.csv")
    drugs = pd.read_csv(SQL / "Q3_Drug_Efficacy_Ranking.csv")
    adr = pd.read_csv(SQL / "Q4_ADR_Risk_by_Drug_and_Age.csv")
    dosage = pd.read_csv(SQL / "Q6_Outcome_vs_Dosage_Level.csv")
    high_risk = pd.read_csv(SQL / "Q5_High_Risk_Patients.csv")
    positive_rate = float(clean["treatment_outcome"].mean())
    best = results.loc[results["f1"].idxmax()]
    best_balanced = results.loc[results["balanced_accuracy"].idxmax()]
    total = len(clean)
    test_n = int(results["test_samples"].iloc[0])
    raw_missing = int(raw.isna().sum().sum())
    raw_dupes = int(raw.duplicated().sum())
    top_adr = adr.iloc[0]

    with PdfPages(REPORT) as pdf:
        # Cover and decision.
        fig = plt.figure(figsize=(8.27, 11.69), facecolor=NAVY)
        fig.text(0.075, 0.88, "MEDICINE OUTCOME\nPREDICTION SYSTEM",
                 fontsize=27, weight="bold", color="white", va="top", linespacing=1.08)
        fig.text(0.08, 0.64, "Business Report", fontsize=20,
                 color="#8BD3DD", weight="bold")
        fig.text(0.08, 0.59, "Evidence review, model recommendation, and implementation gate",
                 fontsize=11, color="#D9E2E8")
        fig.lines.append(plt.Line2D([0.08, 0.92], [
                         0.53, 0.53], transform=fig.transFigure, color=ORANGE, linewidth=3))
        fig.text(0.08, 0.46, "CLIENT DECISION",
                 fontsize=9, weight="bold", color=ORANGE)
        paragraph(fig, "Proceed to controlled clinical validation, not production deployment. Use the Neural Network as the primary screening candidate and the Decision Tree as the transparent challenger.",
                  0.08, 0.425, width=72, size=15, color="white", line=1.3)
        metric(fig, 0.22, 0.23, f"{total:,}", "cleaned records", "#8BD3DD")
        metric(fig, 0.50, 0.23, pct(positive_rate),
               "positive outcomes", ORANGE)
        metric(fig, 0.78, 0.23, f"{best['f1']:.3f}", "best test F1", "#8BD3DD")
        fig.text(0.08, 0.08, "Prepared from the project artifacts | 9 September 2026",
                 fontsize=8.5, color="#B8C8D0")
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

        # Executive summary.
        fig = page("Executive Decision",
                   "What the analysis means for a client decision")
        section(fig, "Recommendation", 0.075, 0.855)
        paragraph(
            fig, f"The project has produced a credible analytical baseline, but not a deployment-ready clinical predictor. The Neural Network leads the held-out test set on F1 ({best['f1']:.3f}) and balanced accuracy ({best['balanced_accuracy']:.3f}); it is therefore the best current candidate for controlled validation. The Decision Tree is the preferred challenger because its reasoning can be inspected directly.", 0.075, 0.825, width=91, size=10.5)
        section(fig, "Key evidence", 0.075, 0.675)
        bullets = [
            f"Outcome prevalence is {positive_rate:.2%}; accuracy alone rewards predicting the majority class.",
            f"The recommended candidate recalls {best['recall']:.1%} of positive outcomes at threshold 0.50, with precision of {best['precision']:.1%}.",
            f"XGBoost and Gradient Boosting show about 77.7% accuracy but recall only {results.loc[results.model == 'XGBoost', 'recall'].iloc[0]:.1%} and {results.loc[results.model == 'Gradient Boosting', 'recall'].iloc[0]:.1%}; they should not be selected on accuracy.",
            f"Age is the clearest business segmentation signal: efficacy falls from 41.23% under 18 to 13.86% at 65+.",
            "Model outputs are predictive associations, not treatment effects or causal recommendations.",
        ]
        y = 0.64
        for bullet in bullets:
            fig.text(0.095, y, "•", fontsize=15, color=ORANGE, va="top")
            paragraph(fig, bullet, 0.12, y, width=82, size=9.8)
            y -= 0.082
        section(fig, "Decision gates before use", 0.075, 0.205, color=RED)
        paragraph(fig, "1. Lock the intended use and false-negative cost.  2. Calibrate probabilities and select a threshold on validation data.  3. Test performance by age, sex, ethnicity, drug, diagnosis, and site.  4. Run temporal and external validation.  5. Deploy only with clinician review, audit logging, monitoring, and rollback.", 0.075, 0.175, width=91, size=9.7, color="#7A2F2F")
        pdf.savefig(fig)
        plt.close(fig)

        # 13 steps.
        fig = page("13-Step Delivery Summary",
                   "Complete project coverage from raw data to clinical analytics")
        steps = [
            ("01", "Data collection and EDA",
             "Profiled raw clinical records, distributions, missingness, duplicates, data types, and correlations."),
            ("02", "Data preprocessing",
             "Standardized placeholders, dates, dosage, duration, demographics, diagnoses, drugs, routes, and binary outcomes."),
            ("03", "Feature engineering",
             "Created age, BMI, duration, risk flags, lab abnormality count, seasonal/date features, and encoded categories."),
            ("04", "Decision Tree",
             "Interpretable baseline with class balancing and held-out evaluation."),
            ("05", "Random Forest",
             "Ensemble baseline with class balancing and held-out evaluation."),
            ("06", "XGBoost", "Boosted model with threshold analysis; accuracy is not a sufficient selection criterion."),
            ("07", "Gradient Boosting",
             "Boosted model with strong majority-class behavior at threshold 0.50."),
            ("08", "Model comparison", "Common stratified 60/20/20 split, scorecard, ROC/PR, calibration, threshold, confusion, and stability analysis."),
            ("09", "Neural Network",
             "Dense Keras classifier; highest test F1 and balanced accuracy in the saved comparison."),
            ("10", "LSTM", "Sequence model evaluated with sequence length 1; it does not yet exploit longitudinal sequences."),
            ("11", "SHAP explainability",
             "Global feature contribution analysis across the trained models."),
            ("12", "SHAP/LIME comparison",
             "Compared global and local explanations; agreement is descriptive and LIME artifacts require caution."),
            ("13", "Clinical SQL analysis",
             "Outcome, age, drug, adverse-event, high-risk, and dosage analyses exported to SQLite and CSV."),
        ]
        y = 0.865
        for number, title, body in steps:
            fig.text(0.08, y, number, fontsize=11,
                     weight="bold", color=ORANGE, va="top")
            fig.text(0.14, y, title, fontsize=10.3,
                     weight="bold", color=NAVY, va="top")
            paragraph(fig, body, 0.14, y - 0.022, width=91,
                      size=8.6, color=MUTED, line=1.2)
            y -= 0.062
        pdf.savefig(fig)
        plt.close(fig)

        # Data and clinical findings.
        fig = page("Data and Clinical Findings",
                   "Descriptive insights for segmentation, safety review, and prioritization")
        metric(fig, 0.20, 0.845, f"{len(raw):,}", "raw records", BLUE)
        metric(fig, 0.40, 0.845, f"{len(raw.columns)}", "raw columns", BLUE)
        metric(fig, 0.60, 0.845, f"{raw_missing:,}",
               "raw missing cells", ORANGE)
        metric(fig, 0.80, 0.845, f"{raw_dupes:,}",
               "exact duplicate rows", ORANGE)
        section(fig, "Outcome and age", 0.075, 0.755)
        ax = fig.add_axes([0.08, 0.50, 0.39, 0.20])
        ax.bar(age["age_group"], age["efficacy_rate_percent"],
               color=[TEAL, BLUE, ORANGE, RED])
        ax.set_ylabel("Outcome rate (%)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=.2)
        ax.set_title("Outcome rate declines with age",
                     fontsize=9, weight="bold", color=NAVY)
        for i, value in enumerate(age["efficacy_rate_percent"]):
            ax.text(i, value + 1, f"{value:.1f}%", ha="center", fontsize=7)
        ax2 = fig.add_axes([0.57, 0.50, 0.35, 0.20])
        ax2.pie([1 - positive_rate, positive_rate], labels=["No", "Yes"], autopct="%1.1f%%",
                colors=["#D9E2E8", ORANGE], textprops={"fontsize": 8}, startangle=90)
        ax2.set_title("Outcome mix", fontsize=9, weight="bold", color=NAVY)
        section(fig, "Business interpretation", 0.075, 0.445)
        paragraph(fig, "The outcome is imbalanced and the age gradient is large enough to support targeted review workflows. Drug efficacy rates are tightly grouped, so the data does not support declaring a single drug superior without adjustment for case mix. Dosage bands are similarly flat, which argues against dose-level conclusions from this descriptive analysis.", 0.075, 0.415, width=91, size=9.5)
        table(fig, [0.075, 0.17, 0.85, 0.20], ["Finding", "Observed result", "Business use"], [
            ["Top efficacy drug",
                f"{drugs.iloc[0].drug_name}: {drugs.iloc[0].efficacy_rate_percent:.2f}%", "Hypothesis for adjusted study"],
            ["Lowest efficacy drug",
                f"{drugs.iloc[-1].drug_name}: {drugs.iloc[-1].efficacy_rate_percent:.2f}%", "Investigate case mix"],
            ["Highest ADR cell",
                f"{top_adr.drug_name}, {top_adr.age_group}: {top_adr.adverse_event_rate_percent:.2f}%", "Prioritize safety review"],
            ["High-risk cohort",
                f"{len(high_risk):,} records with 3+ flags", "Route to clinician review"],
            ["Dose range", f"{dosage.efficacy_rate_percent.min():.2f}% to {dosage.efficacy_rate_percent.max():.2f}%",
             "No raw dose recommendation"],
        ], [0.22, 0.34, 0.44], 7.2)
        pdf.savefig(fig)
        plt.close(fig)

        # Methodology/features.
        fig = page("Analytical Method",
                   "How the project turns clinical records into a decision-support candidate")
        section(fig, "Data controls", 0.075, 0.855)
        paragraph(fig, "The project standardizes malformed text and dates, parses dosage and duration, maps clinical categories to canonical values, removes duplicate patient records by retaining the most complete row, and uses leakage-safe imputation/encoding in the modeling pipeline. Target leakage fields are excluded from model inputs.", 0.075, 0.825, width=91, size=9.7)
        section(fig, "Feature set", 0.075, 0.695)
        features = ["Demographics: age, gender, ethnicity, weight, height, BMI", "Vitals and labs: blood pressure, heart rate, temperature, hemoglobin, WBC, ALT, AST, creatinine, eGFR, HbA1c, cholesterol",
                    "Treatment context: drug, dose, duration, route, concurrent drugs, diagnosis", "Engineered flags: obesity, underweight, hypertension, tachycardia, fever, anemia, cholesterol, diabetes, renal, polypharmacy, smoking/alcohol, lab abnormality count"]
        y = 0.665
        for item in features:
            fig.text(0.095, y, "•", fontsize=14, color=TEAL, va="top")
            paragraph(fig, item, 0.12, y, width=86, size=9.3)
            y -= 0.072
        section(fig, "Evaluation design", 0.075, 0.355)
        table(fig, [0.075, 0.13, 0.85, 0.18], ["Partition", "Records", "Purpose"], [
            ["Training", "591,525", "Fit model parameters"], ["Validation", "197,175", "Tune threshold and decisions"], [
                "Testing", f"{test_n:,}", "One-time held-out comparison"], ["Metrics", "F1, PR-AUC, recall, specificity, Brier", "Balance retrieval, error burden, and probability quality"],
        ], [0.22, 0.23, 0.55], 7.8)
        paragraph(fig, "The same stratified 60% / 20% / 20% split is used across models. Threshold 0.50 is shown for comparability; it must be replaced by a validated operating point tied to clinical and operational costs.",
                  0.075, 0.085, width=91, size=9.2, color=MUTED)
        pdf.savefig(fig)
        plt.close(fig)

        # Model scorecard.
        fig = page("Model Scorecard",
                   "Held-out test results at threshold 0.50")
        score_rows = []
        for _, row in results.iterrows():
            score_rows.append([row.model, f"{row.accuracy:.3f}", f"{row.balanced_accuracy:.3f}", f"{row.precision:.3f}",
                              f"{row.recall:.3f}", f"{row.f1:.3f}", f"{row.pr_auc:.3f}", f"{row.brier:.3f}"])
        table(fig, [0.045, 0.60, 0.91, 0.25], ["Model", "Acc", "Bal acc", "Prec", "Recall",
              "F1", "PR-AUC", "Brier"], score_rows, [0.20, .10, .11, .10, .11, .10, .11, .10], 6.8)
        ax = fig.add_axes([0.10, 0.30, 0.80, 0.22])
        order = results.sort_values("f1")
        ax.barh(order["model"], order["f1"], color=[MODEL_COLORS[m]
                for m in order["model"]])
        ax.set_xlim(0, 0.46)
        ax.set_xlabel("Positive-class F1", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="x", alpha=.2)
        for i, value in enumerate(order["f1"]):
            ax.text(value + 0.005, i, f"{value:.3f}", va="center", fontsize=7)
        ax.set_title("F1 is the clearest screening comparison",
                     fontsize=10, weight="bold", color=NAVY)
        section(fig, "Selection logic", 0.075, 0.235)
        paragraph(fig, f"Neural Network is the current leader on F1 ({best['f1']:.3f}) and balanced accuracy ({best['balanced_accuracy']:.3f}); its performance is close to the Decision Tree while offering the strongest aggregate screening result. Decision Tree remains the interpretability challenger. XGBoost has the highest Brier performance among the saved models, but its default classification behavior produces too many false negatives for a positive-outcome screening workflow.", 0.075, 0.205, width=91, size=9.2)
        pdf.savefig(fig)
        plt.close(fig)

        # Explainability.
        fig = page("Explainability and Clinical Use",
                   "What the models appear to use, and what that does not prove")
        section(fig, "Global SHAP signal", 0.075, 0.855)
        paragraph(fig, "Across the tree-based models, the most stable global contributors are age, concurrent drugs, and creatinine. Other recurring variables include drug name, diagnosis, weight/BMI, liver or renal markers, and cardiovascular measures. These are model-attribution signals, not causal effects and not prescribing instructions.", 0.075, 0.825, width=91, size=9.8)
        top_features = ["age", "concurrent_drugs", "creatinine", "drug_name",
                        "diagnosis", "weight_kg", "bmi", "egfr", "alt", "heart_rate"]
        vals = [0.083, 0.052, 0.035, 0.0020, 0.0019,
                0.0017, 0.0022, 0.0008, 0.0012, 0.0008]
        ax = fig.add_axes([0.13, 0.45, 0.75, 0.28])
        ax.barh(top_features[::-1], vals[::-1], color=TEAL)
        ax.set_xlabel("Illustrative mean |SHAP| scale", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="x", alpha=.2)
        ax.set_title("Recurring drivers in saved SHAP summaries",
                     fontsize=10, weight="bold", color=NAVY)
        section(fig, "Explainability controls", 0.075, 0.385)
        table(fig, [0.075, 0.13, 0.85, 0.20], ["Artifact", "What it provides", "Control needed"], [
            ["SHAP", "Global contribution ranking", "Review by subgroup and time"],
            ["LIME", "Local contribution ranking",
                "Re-run/verify zero-valued artifacts"],
            ["Decision Tree", "Human-readable rules",
                "Check stability and clinical plausibility"],
            ["Model card", "Intended use and limits",
                "Maintain versioned governance record"],
        ], [0.20, 0.40, 0.40], 7.3)
        paragraph(fig, "The SHAP/LIME comparison reports limited top-10 overlap and warns that several available LIME contributions are zero. That result should be treated as an explainability QA issue, not as evidence that the corresponding models have no influential features.", 0.075, 0.085, width=91, size=9.1, color="#7A2F2F")
        pdf.savefig(fig)
        plt.close(fig)

        # Roadmap and close.
        fig = page("Client Action Plan",
                   "A practical path from analysis to governed validation")
        section(fig, "Phase 1 | Reframe", 0.075, 0.855)
        paragraph(fig, "Define whether the product is screening, monitoring, or retrospective analytics. Confirm the positive outcome definition, acceptable false-negative rate, review capacity, and whether the model supports care rather than automates a treatment decision.", 0.075, 0.825, width=91, size=9.5)
        section(fig, "Phase 2 | Strengthen evidence", 0.075, 0.695)
        paragraph(fig, "Tune and calibrate the Neural Network and Decision Tree using validation data only. Add confidence intervals for all priority metrics, temporal holdout, external-site testing, subgroup performance, missingness sensitivity, and leakage review. Re-run LIME with validated local explanations.", 0.075, 0.665, width=91, size=9.5)
        section(fig, "Phase 3 | Pilot with safeguards", 0.075, 0.535)
        paragraph(fig, "Expose risk scores and explanation summaries to trained clinicians as a second opinion. Log inputs, model version, threshold, output, human action, and outcome. Monitor calibration, drift, subgroup gaps, overrides, and false negatives; define rollback ownership before launch.", 0.075, 0.505, width=91, size=9.5)
        section(fig, "Final recommendation", 0.075, 0.365, color=ORANGE)
        fig.text(0.075, 0.325, "PRIMARY CANDIDATE",
                 fontsize=9, weight="bold", color=ORANGE)
        fig.text(0.075, 0.285, "Neural Network",
                 fontsize=22, weight="bold", color=NAVY)
        paragraph(fig, f"Best saved test F1: {best['f1']:.3f} | Recall: {best['recall']:.1%} | PR-AUC: {best['pr_auc']:.3f}",
                  0.075, 0.245, width=91, size=10, color=MUTED)
        fig.text(0.075, 0.185, "CHALLENGER",
                 fontsize=9, weight="bold", color=TEAL)
        fig.text(0.075, 0.145, "Decision Tree",
                 fontsize=18, weight="bold", color=NAVY)
        paragraph(fig, "Use for transparent review and as a benchmark for whether extra model complexity is justified.",
                  0.075, 0.11, width=91, size=9.2, color=MUTED)
        fig.text(0.075, 0.055, "This report summarizes observational project outputs. It does not establish treatment efficacy, safety, or causality.", fontsize=8, color="#7A2F2F")
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Saved business report: {REPORT}")


if __name__ == "__main__":
    main()
