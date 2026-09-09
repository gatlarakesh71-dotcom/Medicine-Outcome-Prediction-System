"""Compare the generated SHAP and LIME feature-importance artifacts."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
SHAP_PATH = ROOT / "output" / "11_shap_feature_importance_all_models.csv"
LIME_PATH = ROOT / "Outputs" / "11_lime_feature_importance_all_models.csv"
OUTPUT_DIR = ROOT / "output"
COMPARISON_CSV = OUTPUT_DIR / "11_shap_lime_comparison.csv"
SUMMARY_TXT = OUTPUT_DIR / "11_shap_lime_comparison_summary.txt"
COMPARISON_PDF = OUTPUT_DIR / "11_shap_lime_comparison_report.pdf"
TOP_N = 10


def base_feature_name(feature: str) -> str:
    """Map transformed LIME names to the original clinical variable."""
    name = re.sub(r"^(numeric|categorical|num|cat)__", "", str(feature))
    categorical_columns = {
        "diagnosis", "drug_name", "gender", "smoking_status", "ethnicity",
        "route", "alcohol_use",
    }
    for column in sorted(categorical_columns, key=len, reverse=True):
        if name == column or name.startswith(f"{column}_"):
            return column
    return name


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not SHAP_PATH.exists() or not LIME_PATH.exists():
        raise FileNotFoundError(
            f"Expected SHAP and LIME files: {SHAP_PATH} and {LIME_PATH}"
        )
    shap = pd.read_csv(SHAP_PATH)
    lime = pd.read_csv(LIME_PATH)
    shap = shap.rename(columns={"mean_abs_shap": "importance"})
    lime["feature"] = lime["feature"].map(base_feature_name)
    lime = lime.groupby(["model", "feature"], as_index=False)[
        "mean_abs_lime"].sum()
    lime = lime.rename(columns={"mean_abs_lime": "importance"})
    return shap[["model", "feature", "importance"]], lime


def compare_model(model: str, shap: pd.DataFrame, lime: pd.DataFrame) -> pd.DataFrame:
    shap_model = shap[shap["model"] == model].sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)
    lime_model = lime[lime["model"] == model].sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)
    shap_model["shap_rank"] = shap_model.index + 1
    lime_model["lime_rank"] = lime_model.index + 1
    merged = shap_model.merge(
        lime_model, on=["model", "feature"], how="outer", suffixes=("_shap", "_lime")
    )
    merged["shap_rank"] = merged["shap_rank"].fillna(0).astype(int)
    merged["lime_rank"] = merged["lime_rank"].fillna(0).astype(int)
    merged["shap_top10"] = merged["shap_rank"].between(1, TOP_N)
    merged["lime_top10"] = merged["lime_rank"].between(1, TOP_N)
    merged["top10_overlap"] = merged["shap_top10"] & merged["lime_top10"]
    return merged[[
        "model", "feature", "importance_shap", "shap_rank", "importance_lime",
        "lime_rank", "shap_top10", "lime_top10", "top10_overlap",
    ]].sort_values(["model", "shap_rank", "lime_rank"])


def save_pdf(comparison: pd.DataFrame, models: list[str]) -> None:
    with PdfPages(COMPARISON_PDF) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.92, "Medicine Outcome Prediction System",
                 fontsize=21, weight="bold")
        fig.text(0.08, 0.875, "SHAP versus LIME explainability comparison",
                 fontsize=14, color="#334155")
        fig.text(0.08, 0.78, "Report scope", fontsize=14, weight="bold")
        fig.text(
            0.08, 0.74,
            "SHAP and LIME feature rankings are compared across the trained models. "
            "LIME one-hot encoded columns are aggregated back to their original "
            "clinical variables before comparison.",
            fontsize=11, wrap=True, va="top",
        )
        fig.text(0.08, 0.59, "Interpretation", fontsize=14, weight="bold")
        fig.text(
            0.08, 0.55,
            "Top-10 overlap describes agreement between two model explanation "
            "methods; it does not establish clinical importance or causation. "
            "Zero LIME values are reported as an explanation limitation, not as "
            "proof that a model has no feature influence.",
            fontsize=11, wrap=True, va="top", color="#7f1d1d",
        )
        fig.text(0.08, 0.38, f"Models compared: {len(models)}", fontsize=12)
        fig.text(0.08, 0.34,
                 "Methods: global SHAP importance and aggregated local LIME importance",
                 fontsize=11)
        pdf.savefig(fig)
        plt.close(fig)

        for model in models:
            rows = comparison[comparison["model"] == model].copy()
            shap_rows = rows[rows["shap_top10"]].sort_values(
                "shap_rank").head(TOP_N)
            lime_rows = rows[rows["lime_top10"]].sort_values(
                "lime_rank").head(TOP_N)
            overlap = rows[rows["top10_overlap"]]["feature"].tolist()
            lime_values = rows["importance_lime"].fillna(0)

            fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
            fig.suptitle(f"{model}: Top explainability features", fontsize=17,
                         weight="bold")
            for axis, data, value_column, title, color in [
                (axes[0], shap_rows, "importance_shap", "SHAP", "#2563eb"),
                (axes[1], lime_rows, "importance_lime", "LIME", "#d97706"),
            ]:
                if data.empty:
                    axis.text(0.5, 0.5, "No values available",
                              ha="center", va="center")
                    axis.axis("off")
                    continue
                plot_data = data.sort_values(value_column)
                axis.barh(plot_data["feature"],
                          plot_data[value_column], color=color)
                axis.set_title(title, weight="bold")
                axis.set_xlabel("Mean absolute contribution")
                axis.grid(axis="x", linestyle="--", alpha=0.25)
            warning = (
                "LIME values are all zero in the available artifact."
                if lime_values.max() == 0 else ""
            )
            fig.text(
                0.08, 0.08,
                f"Top-10 overlap ({len(overlap)}): "
                f"{', '.join(overlap) or 'none'}\n{warning}",
                fontsize=10,
                color="#7f1d1d" if warning else "#334155",
            )
            fig.tight_layout(rect=(0, 0.14, 1, 0.92))
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    shap, lime = load_inputs()
    models = list(dict.fromkeys(shap["model"].tolist()))
    comparison = pd.concat(
        [compare_model(model, shap, lime) for model in models], ignore_index=True
    )
    comparison.to_csv(COMPARISON_CSV, index=False)

    lines = [
        "Medicine Outcome Prediction System - SHAP vs LIME Comparison",
        "=" * 64,
        "SHAP uses mean absolute contribution on the held-out sample.",
        "LIME uses mean absolute local contribution, aggregated back to original clinical variables.",
        "Top-10 overlap is descriptive model agreement, not proof of clinical importance.",
        "",
    ]
    for model in models:
        model_rows = comparison[comparison["model"] == model]
        shap_top = model_rows[model_rows["shap_top10"]]["feature"].tolist()
        lime_top = model_rows[model_rows["lime_top10"]]["feature"].tolist()
        overlap = model_rows[model_rows["top10_overlap"]]["feature"].tolist()
        lime_values = model_rows["importance_lime"].fillna(0)
        lines.extend([
            f"{model}",
            "-" * len(model),
            f"SHAP top features: {', '.join(shap_top) or 'none'}",
            f"LIME top features: {', '.join(lime_top) or 'none'}",
            f"Top-10 overlap ({len(overlap)}): {', '.join(overlap) or 'none'}",
        ])
        if lime_values.max() == 0:
            lines.append(
                "Caution: all available LIME contributions for this model are zero; "
                "this is not evidence that every feature has no influence."
            )
        lines.append("")

    SUMMARY_TXT.write_text("\n".join(lines), encoding="utf-8")
    save_pdf(comparison, models)
    print(f"Saved comparison CSV: {COMPARISON_CSV}")
    print(f"Saved comparison summary: {SUMMARY_TXT}")
    print(f"Saved comparison PDF: {COMPARISON_PDF}")


if __name__ == "__main__":
    main()
