"""Generate SHAP explanations for every model in the project report."""

from __future__ import annotations
import gc
import sys
from sklearn.model_selection import train_test_split
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import joblib

import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")


try:
    import shap
except ImportError as exc:
    raise ImportError(
        "SHAP is required. Install it in GSK_Venv with: "
        "python -m pip install shap"
    ) from exc

matplotlib.use("Agg")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

ROOT = Path(__file__).resolve().parent
SOURCE_OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR = ROOT / "Outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = SOURCE_OUTPUT_DIR / "02_clinical_data_cleaned.csv"
TARGET_COL = "treatment_outcome"
RANDOM_STATE = 42
BACKGROUND_SIZE = 25
EXPLANATION_SIZE = 50

MODEL_FILES = {
    "Decision Tree": SOURCE_OUTPUT_DIR / "decision_tree_treatment_outcome_model.joblib",
    "Random Forest": SOURCE_OUTPUT_DIR / "random_forest_treatment_outcome_model.joblib",
    "XGBoost": SOURCE_OUTPUT_DIR / "06_xgboost_treatment_outcome_model.joblib",
    "Gradient Boosting": SOURCE_OUTPUT_DIR / "07_gradient_boosting_treatment_outcome_model.joblib",
    "Neural Network": SOURCE_OUTPUT_DIR / "09_neural_network_treatment_outcome.keras",
    "LSTM": SOURCE_OUTPUT_DIR / "10_lstm_treatment_outcome.keras",
}
PREPROCESSOR_FILES = {
    "Neural Network": SOURCE_OUTPUT_DIR / "09_neural_network_preprocessor.joblib",
    "LSTM": SOURCE_OUTPUT_DIR / "10_lstm_preprocessor.joblib",
}
MODEL_ORDER = list(MODEL_FILES)
if "--include-deep" not in sys.argv:
    MODEL_ORDER = [
        name for name in MODEL_ORDER
        if name not in ("Neural Network", "LSTM")
    ]
COLORS = ["#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed", "#0891b2"]


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return only reproducible SHAP samples from the report's test split."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {DATA_PATH}")
    target = pd.read_csv(DATA_PATH, usecols=[TARGET_COL], low_memory=False)[
        TARGET_COL]
    valid_row_indices = target.dropna().index.to_numpy()
    y = target.loc[valid_row_indices].astype(int).reset_index(drop=True)
    all_indices = np.arange(len(valid_row_indices))
    train_indices, remaining_indices = train_test_split(
        all_indices, test_size=0.4, random_state=RANDOM_STATE, stratify=y
    )
    _, test_indices = train_test_split(
        remaining_indices, test_size=0.5, random_state=RANDOM_STATE,
        stratify=y.iloc[remaining_indices]
    )
    rng = np.random.default_rng(RANDOM_STATE)
    background_indices = rng.choice(
        train_indices, min(BACKGROUND_SIZE, len(train_indices)), replace=False)
    explanation_indices = rng.choice(
        test_indices, min(EXPLANATION_SIZE, len(test_indices)), replace=False)

    def read_rows(row_positions: np.ndarray) -> pd.DataFrame:
        selected_rows = set(int(index)
                            for index in valid_row_indices[row_positions])
        sampled = pd.read_csv(
            DATA_PATH,
            skiprows=lambda row: row > 0 and row - 1 not in selected_rows,
            low_memory=False,
        )
        return sampled.drop(
            columns=[TARGET_COL, "patient_id", "admission_date", "adverse_event",
                     "readmission_30d"],
            errors="ignore",
        )

    background_raw = read_rows(background_indices)
    explanation_raw = read_rows(explanation_indices)
    return background_raw, explanation_raw, list(background_raw.columns)


def transformed_feature_names(preprocessor) -> list[str]:
    names = list(preprocessor.get_feature_names_out())
    return [name.replace("num__", "").replace("cat__", "") for name in names]


def original_feature_name(transformed_name: str, source_columns: list[str]) -> str:
    for column in sorted(source_columns, key=len, reverse=True):
        if transformed_name == column or transformed_name.startswith(f"{column}_"):
            return column
    return transformed_name


def collapse_feature_names(
    values: np.ndarray, transformed_names: list[str], source_columns: list[str]
) -> pd.DataFrame:
    grouped: dict[str, np.ndarray] = {}
    for index, name in enumerate(transformed_names):
        source_name = original_feature_name(name, source_columns)
        grouped[source_name] = grouped.get(source_name, 0) + values[:, index]
    return pd.DataFrame(grouped)


def load_model_data(
    name: str, background_raw: pd.DataFrame, explanation_raw: pd.DataFrame
):
    if name in PREPROCESSOR_FILES:
        try:
            import keras
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "Install tensorflow-cpu in GSK_Venv before explaining the "
                "Keras models."
            ) from exc
        model = keras.models.load_model(MODEL_FILES[name])
        preprocessor = joblib.load(PREPROCESSOR_FILES[name])
        background = preprocessor.transform(background_raw).astype(np.float32)
        explain_data = preprocessor.transform(
            explanation_raw).astype(np.float32)
        is_lstm = name == "LSTM"
        if is_lstm:
            background = background.reshape(
                background.shape[0], 1, background.shape[1])
            explain_data = explain_data.reshape(
                explain_data.shape[0], 1, explain_data.shape[1])
        return model, preprocessor, background, explain_data, is_lstm

    pipeline = joblib.load(MODEL_FILES[name])
    preprocessor = pipeline.named_steps["preprocessor"]
    return (
        pipeline.named_steps["model"],
        preprocessor,
        preprocessor.transform(background_raw).astype(np.float32),
        preprocessor.transform(explanation_raw).astype(np.float32),
        False,
    )


def extract_shap_values(
    name: str, model, background: np.ndarray, explain_data: np.ndarray, is_lstm: bool
) -> np.ndarray:
    if is_lstm:
        background_2d = background.reshape(
            background.shape[0], background.shape[2])
        explain_2d = explain_data.reshape(
            explain_data.shape[0], explain_data.shape[2])

        def predict(features: np.ndarray) -> np.ndarray:
            sequence = features.reshape(
                features.shape[0], 1, features.shape[1])
            return model.predict(sequence, batch_size=256, verbose=0).ravel()

        explainer = shap.KernelExplainer(predict, background_2d)
        explanation = explainer.shap_values(
            explain_2d,
            nsamples=min(2 * explain_2d.shape[1] + 1, 200),
            l1_reg="num_features(10)",
        )
    elif name == "Neural Network":
        def predict(features): return model.predict(
            features, batch_size=256, verbose=0).ravel()
        explainer = shap.KernelExplainer(predict, background)
        explanation = explainer.shap_values(
            explain_data,
            nsamples=min(2 * explain_data.shape[1] + 1, 200),
            l1_reg="num_features(10)",
        )
    else:
        def predict(features):
            return model.predict_proba(features)[:, 1]
        explainer = shap.KernelExplainer(predict, background)
        explanation = explainer.shap_values(
            explain_data,
            nsamples=min(2 * explain_data.shape[1] + 1, 200),
            l1_reg="num_features(10)",
        )

    values = explanation.values if hasattr(
        explanation, "values") else explanation
    if isinstance(values, list):
        values = values[-1]
    if values.ndim == 3:
        values = values[:, :, -1]
    return np.asarray(values)


def save_model_plot(name: str, importance: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / f"11_shap_{name.lower().replace(' ', '_')}.png"
    top = importance.head(15).sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["mean_abs_shap"],
            color=COLORS[MODEL_ORDER.index(name)])
    ax.set_title(f"{name}: SHAP feature importance on held-out test data")
    ax.set_xlabel("Mean absolute SHAP value for Yes probability")
    ax.set_ylabel("Clinical feature")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    background_raw, explanation_raw, source_columns = prepare_data()

    all_importance: list[pd.DataFrame] = []
    plot_paths: dict[str, Path] = {}
    for name in MODEL_ORDER:
        print(f"Explaining {name}...", flush=True)
        model, preprocessor, background, explain_data, is_lstm = load_model_data(
            name, background_raw, explanation_raw
        )
        shap_values = extract_shap_values(
            name, model, background, explain_data, is_lstm)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, 0, :]
        names = transformed_feature_names(preprocessor)
        names = [
            original_feature_name(name, source_columns) for name in names
        ]
        mean_abs = np.abs(shap_values).mean(axis=0)
        importance = pd.DataFrame(
            {"feature": names, "mean_abs_shap": mean_abs})
        importance = importance.groupby("feature", as_index=False)[
            "mean_abs_shap"].sum()
        importance["model"] = name
        importance = importance.sort_values("mean_abs_shap", ascending=False)
        importance.to_csv(
            OUTPUT_DIR / f"11_shap_{name.lower().replace(' ', '_')}.csv", index=False)
        all_importance.append(importance)
        plot_paths[name] = save_model_plot(name, importance)
        del model, preprocessor, background, explain_data, shap_values
        gc.collect()

    combined = pd.concat(all_importance, ignore_index=True)
    combined.to_csv(
        OUTPUT_DIR / "11_shap_feature_importance_all_models.csv", index=False)
    with PdfPages(OUTPUT_DIR / "11_shap_model_explainability_report.pdf") as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.94, "Medicine Outcome Prediction System",
                 fontsize=19, weight="bold")
        fig.text(0.07, 0.905, "SHAP model explainability | Held-out test-set interpretation",
                 fontsize=11, color="#475569")
        fig.text(0.07, 0.84, "Purpose", fontsize=13, weight="bold")
        fig.text(0.07, 0.81, "SHAP values explain how each clinical feature moves the predicted probability of a Yes treatment outcome. Models are compared using the same report split; explanations use a reproducible sample of the held-out test set.", wrap=True, fontsize=10, va="top")
        fig.text(0.07, 0.69, "Important interpretation",
                 fontsize=13, weight="bold", color="#991b1b")
        fig.text(0.07, 0.66, "A large SHAP value indicates influence on model output, not causation or treatment effectiveness. Feature importance can reflect correlations, missingness patterns, and the training data distribution. Use these results for audit and hypothesis generation, not independent clinical decisions.", wrap=True, fontsize=10, va="top", color="#7f1d1d")
        fig.text(0.07, 0.51, "Analysis settings", fontsize=13, weight="bold")
        fig.text(0.07, 0.48, f"Models: {len(MODEL_ORDER)}\nBackground rows: {len(background_raw):,}\nExplained test rows per model: {len(explanation_raw):,}\nTarget explained: positive-class probability (treatment_outcome = Yes)\nFeature names: one-hot encoded columns aggregated back to original clinical variables", fontsize=10, va="top")
        fig.text(0.07, 0.27, "The detailed feature rankings and model-specific plots are saved in the Outputs folder.",
                 fontsize=10, color="#334155")
        pdf.savefig(fig)
        plt.close(fig)
        for name in MODEL_ORDER:
            image = plt.imread(plot_paths[name])
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.imshow(image)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    with open(OUTPUT_DIR / "11_shap_model_explainability_summary.txt", "w", encoding="utf-8") as report:
        report.write(
            "Medicine Outcome Prediction System - SHAP Explainability\n")
        report.write(
            "Same stratified split and held-out test-set interpretation as the model comparison report.\n\n")
        report.write(
            "SHAP values describe contribution to predicted Yes probability; they do not establish causation.\n\n")
        for name in MODEL_ORDER:
            report.write(
                f"{name} - top features by mean absolute SHAP value\n")
            top = combined[combined["model"] == name].head(10)
            for rank, row in enumerate(top.itertuples(index=False), 1):
                report.write(
                    f"  {rank:2d}. {row.feature}: {row.mean_abs_shap:.6f}\n")
            report.write("\n")
    print(
        f"Saved SHAP report: {OUTPUT_DIR / '11_shap_model_explainability_report.pdf'}")
    print(
        f"Saved combined rankings: {OUTPUT_DIR / '11_shap_feature_importance_all_models.csv'}")


if __name__ == "__main__":
    main()
