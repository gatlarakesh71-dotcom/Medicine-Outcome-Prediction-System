"""Reproducible evaluation and PDF comparison for all trained models."""

from __future__ import annotations
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import joblib
import textwrap

import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")


matplotlib.use("Agg")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titleweight": "bold",
    "axes.edgecolor": "#cbd5e1",
    "axes.labelcolor": "#334155",
    "xtick.color": "#475569",
    "ytick.color": "#475569",
})

A4 = (8.27, 11.69)
PAGE_MARGIN = 0.065

try:
    import keras
except (ImportError, ModuleNotFoundError) as exc:
    raise ImportError(
        "Install tensorflow-cpu in GSK_Venv before running this script.") from exc

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
DATA_PATH = OUTPUT_DIR / "02_clinical_data_cleaned.csv"
TARGET_COL = "treatment_outcome"
RANDOM_STATE = 42
THRESHOLDS = np.arange(0.10, 0.91, 0.05)

MODEL_FILES = {
    "Decision Tree": OUTPUT_DIR / "decision_tree_treatment_outcome_model.joblib",
    "Random Forest": OUTPUT_DIR / "random_forest_treatment_outcome_model.joblib",
    "XGBoost": OUTPUT_DIR / "06_xgboost_treatment_outcome_model.joblib",
    "Gradient Boosting": OUTPUT_DIR / "07_gradient_boosting_treatment_outcome_model.joblib",
    "Neural Network": OUTPUT_DIR / "09_neural_network_treatment_outcome.keras",
    "LSTM": OUTPUT_DIR / "10_lstm_treatment_outcome.keras",
}
PREPROCESSOR_FILES = {
    "Neural Network": OUTPUT_DIR / "09_neural_network_preprocessor.joblib",
    "LSTM": OUTPUT_DIR / "10_lstm_preprocessor.joblib",
}
COLORS = {
    "Decision Tree": "#2563eb",
    "Random Forest": "#059669",
    "XGBoost": "#d97706",
    "Gradient Boosting": "#dc2626",
    "Neural Network": "#7c3aed",
    "LSTM": "#0891b2",
}
MODEL_ORDER = list(MODEL_FILES)


def prepare_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found")
    df = df.drop(columns=[col for col in ["patient_id",
                 "admission_date"] if col in df.columns])
    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL, "adverse_event",
                "readmission_30d"], errors="ignore")
    y = df[TARGET_COL]
    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X, y, test_size=0.4, random_state=RANDOM_STATE, stratify=y
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_remaining, y_remaining, test_size=0.5,
        random_state=RANDOM_STATE, stratify=y_remaining
    )
    return X_train, y_train, X_validation, y_validation, X_test, y_test


def load_predictions(name: str, X: pd.DataFrame) -> np.ndarray:
    if name in ("Neural Network", "LSTM"):
        model = keras.models.load_model(MODEL_FILES[name])
        preprocessor = joblib.load(PREPROCESSOR_FILES[name])
        features = preprocessor.transform(X).astype(np.float32)
        if name == "LSTM":
            features = features.reshape(
                features.shape[0], 1, features.shape[1])
        return model.predict(features, batch_size=4096, verbose=0).ravel()
    pipeline = joblib.load(MODEL_FILES[name])
    return pipeline.predict_proba(X)[:, 1]


def calculate_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, object]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "brier": brier_score_loss(y_true, probabilities),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_ci(y_true: pd.Series, probabilities: np.ndarray, metric: str) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    y_array = np.asarray(y_true)
    values = []
    sample_size = len(y_array)
    for _ in range(100):
        indices = rng.integers(0, sample_size, sample_size)
        sampled_y = y_array[indices]
        sampled_probabilities = probabilities[indices]
        values.append(calculate_metrics(
            pd.Series(sampled_y), sampled_probabilities)[metric])
    return tuple(np.quantile(values, [0.025, 0.975]))


def threshold_rows(y_true: pd.Series, probabilities: np.ndarray, name: str) -> list[dict[str, object]]:
    rows = []
    for threshold in THRESHOLDS:
        metrics = calculate_metrics(y_true, probabilities, threshold)
        rows.append({"model": name, "threshold": threshold, **metrics})
    return rows


def save_results(results: dict[str, dict[str, object]], threshold_data: list[dict[str, object]]) -> None:
    rows = []
    for name in MODEL_ORDER:
        result = results[name]
        test = result["Testing"]
        accuracy_ci = result["accuracy_ci"]
        balanced_ci = result["balanced_accuracy_ci"]
        rows.append({
            "model": name,
            "test_samples": result["test_samples"],
            **{key: test[key] for key in [
                "accuracy", "balanced_accuracy", "precision", "recall",
                "specificity", "f1", "roc_auc", "pr_auc", "brier",
                "tn", "fp", "fn", "tp",
            ]},
            "accuracy_ci_low": accuracy_ci[0], "accuracy_ci_high": accuracy_ci[1],
            "balanced_accuracy_ci_low": balanced_ci[0], "balanced_accuracy_ci_high": balanced_ci[1],
        })
    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "model_comparison_detailed_results.csv", index=False)
    pd.DataFrame(threshold_data).to_csv(
        OUTPUT_DIR / "model_threshold_analysis.csv", index=False)

    with open(OUTPUT_DIR / "model_comparison_detailed_results.txt", "w", encoding="utf-8") as file:
        file.write(
            "Medicine Outcome Prediction System - Detailed Model Evaluation\n")
        file.write(
            "Same stratified 60% training / 20% validation / 20% testing split\n\n")
        for name in MODEL_ORDER:
            file.write(f"{name}\n")
            for split in ("Training", "Validation", "Testing"):
                metrics = results[name][split]
                file.write(
                    f"  {split}: accuracy={metrics['accuracy']:.4f}, "
                    f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
                    f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
                    f"specificity={metrics['specificity']:.4f}, f1={metrics['f1']:.4f}, "
                    f"roc_auc={metrics['roc_auc']:.4f}, pr_auc={metrics['pr_auc']:.4f}, "
                    f"brier={metrics['brier']:.4f}\n"
                )
            file.write(
                f"  Test accuracy 95% CI: {results[name]['accuracy_ci'][0]:.4f}-"
                f"{results[name]['accuracy_ci'][1]:.4f}\n"
            )
            file.write(
                f"  Test balanced accuracy 95% CI: {results[name]['balanced_accuracy_ci'][0]:.4f}-"
                f"{results[name]['balanced_accuracy_ci'][1]:.4f}\n\n"
            )


def fmt(value: float) -> str:
    return f"{value:.3f}"


def add_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(PAGE_MARGIN, 0.955, title, fontsize=18,
             fontweight="bold", color="#0f172a", va="top")
    fig.text(PAGE_MARGIN, 0.918, subtitle, fontsize=9,
             color="#475569", va="top")
    fig.lines.append(plt.Line2D(
        [PAGE_MARGIN, 1 - PAGE_MARGIN], [0.895, 0.895],
        transform=fig.transFigure, color="#cbd5e1", linewidth=0.8
    ))


def add_paragraph(fig: plt.Figure, text: str, x: float, y: float,
                  width: int = 92, fontsize: float = 10,
                  color: str = "#334155", line_spacing: float = 1.35) -> None:
    wrapped = textwrap.fill(text, width=width, break_long_words=False)
    fig.text(x, y, wrapped, fontsize=fontsize, color=color,
             va="top", linespacing=line_spacing)


def add_table(ax: plt.Axes, headers: list[str], rows: list[list[str]], widths: list[float], size: float = 7) -> None:
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers,
                     loc="center", cellLoc="center", colWidths=widths)
    table.auto_set_font_size(False)
    table.set_fontsize(size)
    table.scale(1, 1.65)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        if row == 0:
            cell.set_facecolor("#0f172a")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f8fafc" if row % 2 == 0 else "white")
            cell.set_text_props(color="#1e293b")


def build_pdf(results: dict[str, dict[str, object]], threshold_data: list[dict[str, object]]) -> Path:
    pdf_path = OUTPUT_DIR / "model_comparison_detailed_report.pdf"
    test_rows = [{"model": name, **results[name]["Testing"]}
                 for name in MODEL_ORDER]
    best_f1 = max(test_rows, key=lambda row: row["f1"])
    best_pr = max(test_rows, key=lambda row: row["pr_auc"])
    best_recall = max(test_rows, key=lambda row: row["recall"])
    best_auc = max(test_rows, key=lambda row: row["roc_auc"])
    baseline = test_rows[0]

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=A4, facecolor="#f8fafc")
        add_header(fig, "Medicine Outcome Prediction System",
                   "Deep model evaluation and comparison | Reproducible test-set analysis")
        fig.text(PAGE_MARGIN, 0.865, "EXECUTIVE FINDINGS", fontsize=9,
                 fontweight="bold", color="#2563eb")
        findings = [
            f"Best positive-class F1: {best_f1['model']} ({fmt(best_f1['f1'])}).",
            f"Best PR-AUC: {best_pr['model']} ({fmt(best_pr['pr_auc'])}); this is more informative than raw accuracy for the imbalanced outcome.",
            f"Best positive recall: {best_recall['model']} ({fmt(best_recall['recall'])}); recall matters when missing a Yes outcome is costly.",
            f"Best ROC-AUC: {best_auc['model']} ({fmt(best_auc['roc_auc'])}), but ranking ability does not guarantee a useful 0.5 threshold.",
            "XGBoost and Gradient Boosting show high raw accuracy while positive recall is approximately zero: they mostly predict No.",
        ]
        y = 0.815
        for finding in findings:
            add_paragraph(fig, "- " + finding, 0.08, y, width=88, fontsize=10)
            y -= 0.065
        ax = fig.add_axes([0.10, 0.30, 0.82, 0.28])
        labels = MODEL_ORDER
        values = [results[name]["Testing"]["f1"] for name in labels]
        bars = ax.bar(labels, values, color=[COLORS[name] for name in labels])
        ax.set_ylim(0, max(values) * 1.35)
        ax.set_ylabel("Positive-class F1")
        ax.set_title(
            "Test-set F1 is the primary screening comparison", fontweight="bold")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value +
                    0.005, fmt(value), ha="center", fontsize=8)
        fig.text(PAGE_MARGIN, 0.20, "Decision warning", fontsize=11,
                 fontweight="bold", color="#991b1b")
        add_paragraph(fig, "These are predictive results, not evidence of clinical efficacy or treatment causality. Human review, calibration, subgroup testing, and prospective validation are required before use.",
                      PAGE_MARGIN, 0.165, width=92, fontsize=9.5, color="#7f1d1d")
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=A4, facecolor="white")
        add_header(fig, "Test-Set Scorecard",
                   "All models evaluated on the identical 197,176-record test partition")
        headers = ["Model", "Acc", "Bal acc", "Prec", "Recall",
                   "Spec", "F1", "ROC-AUC", "PR-AUC", "Brier"]
        rows = [[name] + [fmt(results[name]["Testing"][key]) for key in ["accuracy", "balanced_accuracy",
                                                                         "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "brier"]] for name in MODEL_ORDER]
        add_table(fig.add_axes([0.03, 0.59, 0.94, 0.22]),
                  headers, rows, [0.18] + [0.091] * 9, 6.2)
        fig.text(PAGE_MARGIN, 0.52, "Interpretation", fontsize=12,
                 fontweight="bold", color="#0f172a")
        text = ("Accuracy is not sufficient here: the positive class is the minority outcome. "
                "Balanced accuracy, positive recall, F1, PR-AUC, and the confusion matrix should drive selection. "
                "The Brier score measures probability error; lower is better. Confidence intervals below are bootstrap 95% intervals.")
        add_paragraph(fig, text, PAGE_MARGIN, 0.48, width=92, fontsize=9.5)
        ci_rows = [[name, f"{results[name]['accuracy_ci'][0]:.4f} - {results[name]['accuracy_ci'][1]:.4f}",
                    f"{results[name]['balanced_accuracy_ci'][0]:.4f} - {results[name]['balanced_accuracy_ci'][1]:.4f}"] for name in MODEL_ORDER]
        add_table(fig.add_axes([0.12, 0.20, 0.76, 0.22]), [
                  "Model", "Accuracy 95% CI", "Balanced accuracy 95% CI"], ci_rows, [0.35, 0.32, 0.33], 7)
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=A4, facecolor="white")
        add_header(fig, "Discrimination and Calibration",
                   "Ranking quality, positive-class retrieval, and probability reliability")
        roc_ax = fig.add_axes([0.09, 0.55, 0.38, 0.30])
        pr_ax = fig.add_axes([0.57, 0.55, 0.36, 0.30])
        cal_ax = fig.add_axes([0.13, 0.15, 0.36, 0.27])
        for name in MODEL_ORDER:
            test = results[name]
            fpr, tpr, _ = roc_curve(test["y_true"], test["probabilities"])
            precision, recall, _ = precision_recall_curve(
                test["y_true"], test["probabilities"])
            roc_ax.plot(
                fpr, tpr, label=f"{name} ({test['roc_auc']:.3f})", color=COLORS[name])
            pr_ax.plot(
                recall, precision, label=f"{name} ({test['pr_auc']:.3f})", color=COLORS[name])
            fraction, mean_predicted = calibration_curve(
                test["y_true"], test["probabilities"], n_bins=10, strategy="quantile")
            cal_ax.plot(mean_predicted, fraction, marker="o",
                        color=COLORS[name], label=name)
        roc_ax.plot([0, 1], [0, 1], "--", color="#94a3b8")
        roc_ax.set_title("ROC curves")
        roc_ax.set_xlabel("False positive rate")
        roc_ax.set_ylabel("True positive rate")
        pr_ax.axhline(
            np.mean(results[MODEL_ORDER[0]]["y_true"]), linestyle="--", color="#94a3b8")
        pr_ax.set_title("Precision-recall curves")
        pr_ax.set_xlabel("Recall")
        pr_ax.set_ylabel("Precision")
        cal_ax.plot([0, 1], [0, 1], "--", color="#94a3b8")
        cal_ax.set_title("Calibration curves")
        cal_ax.set_xlabel("Mean predicted probability")
        cal_ax.set_ylabel("Fraction positive")
        for ax in (roc_ax, pr_ax, cal_ax):
            ax.grid(True, linestyle="--", alpha=0.25)
            ax.legend(fontsize=5.5, frameon=False, loc="upper left",
                      bbox_to_anchor=(0, -0.18), ncol=2)
        add_paragraph(fig, "ROC-AUC measures ranking across thresholds. PR-AUC is preferable when the positive class is imbalanced. Calibration determines whether a probability such as 0.70 behaves like a roughly 70% event rate.", PAGE_MARGIN, 0.08, width=92, fontsize=9.2)
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=A4, facecolor="white")
        add_header(fig, "Threshold Analysis",
                   "The default 0.50 threshold is a policy choice, not a universal optimum")
        ax1 = fig.add_axes([0.10, 0.56, 0.80, 0.28])
        ax2 = fig.add_axes([0.10, 0.16, 0.80, 0.28])
        for name in MODEL_ORDER:
            data = [row for row in threshold_data if row["model"] == name]
            ax1.plot([row["threshold"] for row in data], [row["f1"]
                     for row in data], marker="o", label=name, color=COLORS[name])
            ax2.plot([row["threshold"] for row in data], [row["recall"]
                     for row in data], marker="o", label=name, color=COLORS[name])
        ax1.set_title("F1 by threshold")
        ax1.set_ylabel("F1")
        ax2.set_title("Recall by threshold")
        ax2.set_xlabel("Decision threshold")
        ax2.set_ylabel("Recall")
        for ax in (ax1, ax2):
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle="--", alpha=0.25)
            ax.legend(fontsize=5.5, frameon=False, ncol=3, loc="upper left",
                      bbox_to_anchor=(0, -0.18))
        add_paragraph(fig, "Lowering the threshold generally increases positive recall and follow-up workload; raising it generally increases precision while missing more positive outcomes. Select the operating point using explicit false-negative and false-positive costs.", PAGE_MARGIN, 0.08, width=92, fontsize=9.2)
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=A4, facecolor="#f8fafc")
        add_header(fig, "Confusion Matrices and Error Burden",
                   "Test-set counts at the default 0.50 threshold")
        for index, name in enumerate(MODEL_ORDER):
            row = index // 2
            col = index % 2
            ax = fig.add_axes(
                [0.10 + col * 0.47, 0.59 - row * 0.25, 0.30, 0.17])
            test = results[name]
            matrix = np.array(
                [[test["tn"], test["fp"]], [test["fn"], test["tp"]]])
            image = ax.imshow(matrix, cmap="Blues")
            ax.set_title(name, fontsize=10, fontweight="bold")
            ax.set_xticks([0, 1], ["No", "Yes"])
            ax.set_yticks([0, 1], ["No", "Yes"])
            ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("Actual", fontsize=8)
            for r in range(2):
                for c in range(2):
                    ax.text(c, r, f"{matrix[r, c]:,}",
                            ha="center", va="center", fontsize=8)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        add_paragraph(fig, "False negatives are actual Yes outcomes predicted as No. In a screening use case, false negatives usually deserve more attention than raw accuracy.",
                      PAGE_MARGIN, 0.08, width=92, fontsize=9.5)
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=A4, facecolor="white")
        add_header(fig, "Model Stability Across Splits",
                   "Validation-to-test drift is a useful warning signal, not proof of generalization")
        rows = []
        for name in MODEL_ORDER:
            validation = results[name]["Validation"]
            testing = results[name]["Testing"]
            rows.append([name, fmt(validation["f1"]), fmt(testing["f1"]), fmt(testing["f1"] - validation["f1"]),
                        fmt(validation["pr_auc"]), fmt(testing["pr_auc"]), fmt(testing["pr_auc"] - validation["pr_auc"])])
        add_table(fig.add_axes([0.05, 0.59, 0.90, 0.22]), ["Model", "Val F1", "Test F1", "Delta",
                  "Val PR-AUC", "Test PR-AUC", "Delta"], rows, [0.20, 0.12, 0.12, 0.12, 0.16, 0.16, 0.12], 7)
        fig.text(PAGE_MARGIN, 0.48, "Recommended operating posture",
                 fontsize=12, fontweight="bold", color="#0f172a")
        body = (f"For a positive-outcome screening workflow, start controlled validation with {best_f1['model']} because it has the strongest test F1 among the saved artifacts. "
                f"The LSTM and neural network are competitive with the tree models, but the current LSTM has sequence length 1, so it does not yet exploit temporal sequences. "
                "Do not select XGBoost or Gradient Boosting solely on accuracy: their default-threshold positive recall is effectively zero.")
        add_paragraph(fig, body, PAGE_MARGIN, 0.43, width=92, fontsize=9.5)
        fig.text(PAGE_MARGIN, 0.25, "Required before deployment",
                 fontsize=12, fontweight="bold", color="#1d4ed8")
        required = ["Threshold and cost analysis", "Probability calibration", "Subgroup fairness and error analysis",
                    "Temporal or external validation", "Human review, audit logging, drift monitoring, and rollback"]
        add_paragraph(fig, "\n".join("- " + item for item in required),
                      0.08, 0.20, width=82, fontsize=9.5, line_spacing=1.5)
        pdf.savefig(fig)
        plt.close(fig)

    return pdf_path


def main() -> None:
    X_train, y_train, X_validation, y_validation, X_test, y_test = prepare_data()
    results: dict[str, dict[str, object]] = {}
    threshold_data: list[dict[str, object]] = []
    for name in MODEL_ORDER:
        probabilities_train = load_predictions(name, X_train)
        probabilities_validation = load_predictions(name, X_validation)
        probabilities_test = load_predictions(name, X_test)
        results[name] = {
            "Training": calculate_metrics(y_train, probabilities_train),
            "Validation": calculate_metrics(y_validation, probabilities_validation),
            "Testing": calculate_metrics(y_test, probabilities_test),
            "test_samples": len(y_test),
            "y_true": np.asarray(y_test),
            "probabilities": probabilities_test,
        }
        results[name]["accuracy_ci"] = bootstrap_ci(
            y_test, probabilities_test, "accuracy")
        results[name]["balanced_accuracy_ci"] = bootstrap_ci(
            y_test, probabilities_test, "balanced_accuracy")
        threshold_data.extend(threshold_rows(y_test, probabilities_test, name))
        print(
            f"{name}: test F1={results[name]['Testing']['f1']:.4f}, PR-AUC={results[name]['Testing']['pr_auc']:.4f}")
    save_results(results, threshold_data)
    pdf_path = build_pdf(results, threshold_data)
    print(f"Saved detailed PDF: {pdf_path}")
    print(
        f"Saved CSV results: {OUTPUT_DIR / 'model_comparison_detailed_results.csv'}")
    print(
        f"Saved threshold analysis: {OUTPUT_DIR / 'model_threshold_analysis.csv'}")


if __name__ == "__main__":
    main()
