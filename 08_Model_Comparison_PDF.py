from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / 'output'

MODEL_FILES = {
    'Decision Tree': OUT_DIR / '04_decision_tree_treatment_outcome_metrics.txt',
    'Random Forest': OUT_DIR / '05_random_forest_treatment_outcome_metrics.txt',
    'XGBoost': OUT_DIR / '06_xgboost_treatment_outcome_metrics.txt',
    'Gradient Boosting': OUT_DIR / '07_gradient_boosting_treatment_outcome_metrics.txt',
}


def parse_confusion_matrix(text: str):
    match = re.search(
        r'Confusion Matrix:\s*\[\[\s*(\d+)\s+(\d+)\s*\]\s*\[\s*(\d+)\s+(\d+)\s*\]\]', text, re.S)
    if match is None:
        return [0, 0, 0, 0]
    return [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))]


def read_model_metrics(path: Path):
    text = path.read_text(encoding='utf-8')
    accuracy_match = re.search(r'Accuracy:\s*([0-9.]+)', text)
    yes_metrics = re.search(
        r'Yes\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)', text)
    macro_match = re.search(
        r'macro avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)', text, re.M)
    weighted_match = re.search(
        r'weighted avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)', text, re.M)

    if accuracy_match is None:
        raise ValueError(f'Accuracy not found in {path}')

    precision = float(yes_metrics.group(1)) if yes_metrics else 0.0
    recall = float(yes_metrics.group(2)) if yes_metrics else 0.0
    f1 = float(yes_metrics.group(3)) if yes_metrics else 0.0

    return {
        'accuracy': float(accuracy_match.group(1)),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_f1': float(macro_match.group(3)) if macro_match else 0.0,
        'weighted_f1': float(weighted_match.group(3)) if weighted_match else 0.0,
        'confusion_matrix': parse_confusion_matrix(text),
    }


rows = []
for model_name, path in MODEL_FILES.items():
    if not path.exists():
        raise FileNotFoundError(f'Metrics file not found: {path}')
    rows.append({'model': model_name, **read_model_metrics(path)})

rows = sorted(rows, key=lambda r: r['accuracy'], reverse=True)
recommended_model = rows[0]

pdf_path = OUT_DIR / 'model_comparison_summary.pdf'

with PdfPages(pdf_path) as pdf:
    # Page 1 - Executive summary
    fig1 = plt.figure(figsize=(8.27, 11.69), facecolor='#f7fafc')
    fig1.subplots_adjust(left=0.07, right=0.97, top=0.96, bottom=0.06)

    fig1.text(0.5, 0.96, 'Medicine Outcome Prediction System',
              ha='center', fontsize=18, fontweight='bold', color='#0f172a')
    fig1.text(0.5, 0.91, 'Model Comparison and Selection Report',
              ha='center', fontsize=20, fontweight='bold', color='#0f172a')

    summary_box = fig1.add_axes([0.10, 0.70, 0.80, 0.15])
    summary_box.axis('off')
    summary_box.add_patch(plt.Rectangle(
        (0.0, 0.0), 1.0, 1.0, facecolor='#dbeafe', edgecolor='#60a5fa', linewidth=1.2))
    summary_box.text(0.5, 0.70, 'Recommended Model', ha='center',
                     va='center', fontsize=12, fontweight='bold', color='#1d4ed8')
    summary_box.text(0.5, 0.28, recommended_model['model'], ha='center',
                     va='center', fontsize=24, fontweight='bold', color='#0f172a')

    info_ax = fig1.add_axes([0.12, 0.50, 0.76, 0.14])
    info_ax.axis('off')
    info_ax.text(0.02, 0.80, 'Selection Basis:', fontsize=11,
                 fontweight='bold', color='#0f172a')
    info_ax.text(
        0.02, 0.58, f"Accuracy: {recommended_model['accuracy']:.4f}", fontsize=11, color='#1f2937')
    info_ax.text(
        0.02, 0.36, f"Precision: {recommended_model['precision']:.4f}   Recall: {recommended_model['recall']:.4f}   F1-score: {recommended_model['f1']:.4f}", fontsize=11, color='#1f2937')

    fig1.text(0.12, 0.36, 'Data split:', fontsize=11,
              fontweight='bold', color='#0f172a')
    fig1.text(0.34, 0.36, '80% Training / 20% Testing (stratified)',
              fontsize=11, color='#1f2937')
    fig1.text(0.12, 0.28, 'Evaluation metrics:', fontsize=11,
              fontweight='bold', color='#0f172a')
    fig1.text(0.34, 0.28, 'Accuracy, Precision, Recall, F1-score, Confusion Matrix',
              fontsize=11, color='#1f2937')

    chart_ax = fig1.add_axes([0.12, 0.08, 0.76, 0.16])
    labels = [row['model'] for row in rows]
    values = [row['accuracy'] for row in rows]
    colors = ['#1d4ed8', '#10b981', '#f59e0b', '#8b5cf6']
    chart_ax.bar(labels, values, color=colors[:len(
        labels)], edgecolor='#1f2937', linewidth=1)
    chart_ax.set_ylim(0, 1.0)
    chart_ax.set_ylabel('Accuracy', fontsize=10, fontweight='bold')
    chart_ax.set_title('Accuracy Comparison', fontsize=12,
                       fontweight='bold', color='#0f172a')
    chart_ax.grid(axis='y', linestyle='--', alpha=0.4)
    chart_ax.set_axisbelow(True)
    for i, v in enumerate(values):
        chart_ax.text(i, v + 0.02, f'{v:.3f}', ha='center',
                      fontsize=9, fontweight='bold', color='#1f2937')
    chart_ax.tick_params(axis='x', rotation=18)
    for label in chart_ax.get_xticklabels():
        label.set_fontsize(8)
    pdf.savefig(fig1)

    # Page 2 - Detailed metrics table
    fig2 = plt.figure(figsize=(8.27, 11.69), facecolor='#ffffff')
    fig2.subplots_adjust(left=0.03, right=0.98, top=0.92, bottom=0.06)
    fig2.text(0.5, 0.94, 'Detailed Model Evaluation', ha='center',
              fontsize=22, fontweight='bold', color='#0f172a')

    table_data = [['Model', 'Accuracy', 'Precision',
                   'Recall', 'F1-score', 'Confusion Matrix']]
    for row in rows:
        tn, fp, fn, tp = row['confusion_matrix']
        table_data.append([
            row['model'],
            f"{row['accuracy']:.4f}",
            f"{row['precision']:.4f}",
            f"{row['recall']:.4f}",
            f"{row['f1']:.4f}",
            f"[[{tn}, {fp}]\n [{fn}, {tp}]]",
        ])

    table_ax = fig2.add_axes([0.04, 0.38, 0.92, 0.45])
    table_ax.axis('off')
    table = table_ax.table(
        cellText=table_data[1:], colLabels=table_data[0], loc='center', cellLoc='center', fontsize=8,
        colWidths=[0.18, 0.15, 0.15, 0.15, 0.15, 0.22])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#cbd5e1')
        if r == 0:
            cell.set_facecolor('#0f172a')
            cell.set_text_props(color='white', weight='bold')
        else:
            cell.set_facecolor('#ffffff' if r % 2 else '#f8fafc')
            cell.set_text_props(color='#0f172a')

    fig2.text(0.08, 0.22, 'Interpretation:', fontsize=11,
              fontweight='bold', color='#0f172a')
    fig2.text(0.08, 0.15, 'XGBoost is the strongest model in terms of overall predictive accuracy for this clinical outcome data.',
              fontsize=10.5, color='#1f2937')
    fig2.text(0.08, 0.08, 'For a clinical use case, recall and F1-score for the positive outcome should also be reviewed alongside accuracy.',
              fontsize=10.5, color='#1f2937')
    pdf.savefig(fig2)

    # Page 3 - Confusion matrices
    fig3 = plt.figure(figsize=(8.27, 11.69), facecolor='#f8fafc')
    fig3.subplots_adjust(left=0.06, right=0.96, top=0.90, bottom=0.10)
    fig3.text(0.5, 0.95, 'Confusion Matrix Comparison', ha='center',
              fontsize=22, fontweight='bold', color='#0f172a')

    matrix_axes = [fig3.add_axes(
        [0.12 + (i % 2) * 0.34, 0.72 - (i // 2) * 0.30, 0.24, 0.18]) for i in range(len(rows))]
    for ax, row in zip(matrix_axes, rows):
        tn, fp, fn, tp = row['confusion_matrix']
        cm = [[tn, fp], [fn, tp]]
        im = ax.imshow(cm, cmap='Blues')
        ax.set_title(row['model'], fontsize=10, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['No', 'Yes'])
        ax.set_yticklabels(['No', 'Yes'])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i][j]), ha='center',
                        va='center', color='black', fontsize=9)
        ax.set_aspect('equal')
        fig3.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig3.text(0.08, 0.06, 'Prepared for presentation',
              fontsize=10, color='#475569')
    fig3.text(0.72, 0.06, 'A4 Paper Size', fontsize=10, color='#475569')
    pdf.savefig(fig3)

print(f'Saved PDF: {pdf_path}')
