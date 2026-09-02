import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import matplotlib
matplotlib.use('Agg')

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'Medicine Outcome Prediction' / 'clinical_data_raw.csv'
OUTPUT_DIR = ROOT / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)
PDF_PATH = OUTPUT_DIR / 'clinical_eda_report.pdf'


def clean_numeric_df(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return pd.DataFrame()

    numeric_df = df[num_cols].apply(pd.to_numeric, errors='coerce')
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.dropna(axis=1, how='all')
    return numeric_df


def save_pdf_figure(pdf, fig, page_name):
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print(f'Added PDF page: {page_name}')


def add_title_page(pdf, df):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')
    fig.text(0.5, 0.96, 'Clinical Data EDA Report',
             ha='center', fontsize=20, fontweight='bold')
    fig.text(0.5, 0.90, f'Dataset: {DATA_PATH.name}', ha='center', fontsize=12)

    numeric_df = clean_numeric_df(df)
    summary_df = pd.DataFrame({
        'Metric': [
            'Rows',
            'Columns',
            'Missing cells',
            'Duplicate rows',
            'Missing rate (%)',
            'Numeric columns',
        ],
        'Value': [
            len(df),
            len(df.columns),
            int(df.isna().sum().sum()),
            int(df.duplicated().sum()),
            round((df.isna().sum().sum() / df.size) * 100, 2),
            len(numeric_df.columns),
        ],
    })

    ax = fig.add_axes([0.18, 0.22, 0.64, 0.36])
    ax.axis('off')
    table = ax.table(cellText=summary_df.values,
                     colLabels=summary_df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    fig.text(0.08, 0.08, 'This report contains visual summaries of data quality, distributions, and relationships.',
             fontsize=9, color='gray')
    save_pdf_figure(pdf, fig, 'Title')


def add_missingness_page(pdf, df):
    missing = df.isna().sum().sort_values(ascending=False).head(15)
    if missing.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.bar(missing.index.astype(str), missing.values, color='tomato')
    ax.set_title('Top Missing Values by Column', fontsize=14)
    ax.set_xlabel('Column')
    ax.set_ylabel('Missing Count')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    save_pdf_figure(pdf, fig, 'Missing values')


def add_numeric_distribution_page(pdf, df):
    numeric_df = clean_numeric_df(df)
    if numeric_df.empty:
        return
    cols = list(numeric_df.columns[:6])
    if not cols:
        return

    rows = (len(cols) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
    axes = np.ravel(axes)

    for i, col in enumerate(cols):
        ax = axes[i]
        values = numeric_df[col].dropna()
        if values.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.axis('off')
            continue
        ax.hist(values, bins=20, color='steelblue', edgecolor='black')
        ax.set_title(col)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')

    for j in range(len(cols), len(axes)):
        axes[j].axis('off')

    save_pdf_figure(pdf, fig, 'Numeric distribution')


def add_boxplot_page(pdf, df):
    numeric_df = clean_numeric_df(df)
    if numeric_df.empty:
        return
    cols = list(numeric_df.columns[:6])
    rows = (len(cols) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
    axes = np.ravel(axes)

    for i, col in enumerate(cols):
        ax = axes[i]
        values = numeric_df[col].dropna().to_numpy()
        if values.size == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.axis('off')
            continue
        ax.boxplot(values, patch_artist=True, boxprops={
                   'facecolor': 'lightgreen', 'edgecolor': 'darkgreen'})
        ax.set_title(col)
        ax.set_ylabel('Value')

    for j in range(len(cols), len(axes)):
        axes[j].axis('off')

    save_pdf_figure(pdf, fig, 'Boxplots')


def add_correlation_page(pdf, df):
    numeric_df = clean_numeric_df(df)
    if numeric_df.shape[1] < 2:
        return
    corr_cols = list(numeric_df.columns[:10])
    corr = numeric_df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_title('Correlation Matrix')
    ax.set_xticks(range(len(corr_cols)))
    ax.set_yticks(range(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=45, ha='right')
    ax.set_yticklabels(corr_cols)

    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center',
                    va='center', color='black', fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_pdf_figure(pdf, fig, 'Correlation matrix')


def add_categorical_page(pdf, df):
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if not cat_cols:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, col in zip(axes, cat_cols[:4]):
        counts = df[col].dropna().astype(str).value_counts().head(8)
        if counts.empty:
            ax.text(0.5, 0.5, 'No categorical values',
                    ha='center', va='center')
            ax.axis('off')
            continue
        counts.plot(kind='bar', ax=ax, color='seagreen', edgecolor='black')
        ax.set_title(f'{col} Distribution')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    for j in range(len(cat_cols[:4]), len(axes)):
        axes[j].axis('off')

    save_pdf_figure(pdf, fig, 'Categorical distribution')


def add_dtype_page(pdf, df):
    dtype_df = df.dtypes.reset_index().rename(
        columns={'index': 'Column', 0: 'Data Type'})
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.5, 0.96, 'Column Data Types',
             ha='center', fontsize=16, fontweight='bold')
    ax = fig.add_axes([0.12, 0.08, 0.76, 0.82])
    ax.axis('off')
    table = ax.table(cellText=dtype_df.values,
                     colLabels=dtype_df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.8)
    save_pdf_figure(pdf, fig, 'Data types')


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f'Data file not found: {DATA_PATH}')

    df = pd.read_csv(DATA_PATH, low_memory=False)
    with PdfPages(PDF_PATH) as pdf:
        add_title_page(pdf, df)
        add_missingness_page(pdf, df)
        add_numeric_distribution_page(pdf, df)
        add_boxplot_page(pdf, df)
        add_correlation_page(pdf, df)
        add_categorical_page(pdf, df)
        add_dtype_page(pdf, df)

    print(f'Dataset shape: {df.shape}')
    print(f'PDF saved to: {PDF_PATH}')


if __name__ == '__main__':
    main()
