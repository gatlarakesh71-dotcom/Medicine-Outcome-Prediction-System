"""Train an LSTM baseline for clinical outcome classification.

The current dataset contains one row per unique patient, so each sample has
one timestep. A multi-timestep LSTM requires repeated visits per patient.
"""

from __future__ import annotations
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")


try:
    import keras
    from keras import layers
except (ImportError, ModuleNotFoundError) as exc:
    raise ImportError(
        "Keras requires a working TensorFlow backend. Install tensorflow-cpu "
        "in GSK_Venv before running this script."
    ) from exc

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
MODEL_PATH = OUTPUT_DIR / "10_lstm_treatment_outcome.keras"
PREPROCESSOR_PATH = OUTPUT_DIR / "10_lstm_preprocessor.joblib"
METRICS_PATH = OUTPUT_DIR / "10_lstm_treatment_outcome_metrics.txt"
HISTORY_PLOT_PATH = OUTPUT_DIR / "10_lstm_training_history.png"
TARGET_COL = "treatment_outcome"
SEQUENCE_LENGTH = 1


def resolve_cleaned_dataset() -> Path:
    candidates = [
        OUTPUT_DIR / "02_clinical_data_cleaned.csv",
        OUTPUT_DIR / "clinical_data_cleaned.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Cleaned dataset not found in {OUTPUT_DIR}")


def build_model(input_width: int) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(SEQUENCE_LENGTH, input_width)),
            layers.LSTM(32, activation="tanh"),
            layers.Dropout(0.30),
            layers.Dense(16, activation="relu"),
            layers.Dropout(0.20),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="clinical_outcome_lstm",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def main() -> None:
    data_path = resolve_cleaned_dataset()
    df = pd.read_csv(data_path, low_memory=False)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found in cleaned data")

    df = df.drop(
        columns=[col for col in ["patient_id",
                                 "admission_date"] if col in df.columns]
    )
    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    X = df.drop(columns=[TARGET_COL, "adverse_event",
                "readmission_30d"], errors="ignore")
    y = df[TARGET_COL]
    numeric_cols = [
        col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(
                            handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    X_train_processed = preprocessor.fit_transform(X_train).astype(np.float32)
    X_test_processed = preprocessor.transform(X_test).astype(np.float32)
    X_train_sequence = X_train_processed.reshape(
        -1, SEQUENCE_LENGTH, X_train_processed.shape[1])
    X_test_sequence = X_test_processed.reshape(
        -1, SEQUENCE_LENGTH, X_test_processed.shape[1])

    model = build_model(X_train_processed.shape[1])
    class_counts = np.bincount(y_train.to_numpy())
    total = class_counts.sum()
    class_weight = {
        class_id: total / (len(class_counts) * count)
        for class_id, count in enumerate(class_counts)
        if count > 0
    }

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
    ]
    history = model.fit(
        X_train_sequence,
        y_train.to_numpy(),
        validation_split=0.1,
        epochs=100,
        batch_size=2048,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("LSTM Loss Over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot(history.history["auc"], label="Train AUC")
    axes[1].plot(history.history["val_auc"], label="Val AUC")
    axes[1].set_title("LSTM AUC Over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(HISTORY_PLOT_PATH, dpi=150)
    plt.close(fig)

    probabilities = model.predict(
        X_test_sequence, batch_size=4096, verbose=0).ravel()
    y_pred = (probabilities >= 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["No", "Yes"])
    cm = confusion_matrix(y_test, y_pred)

    print("LSTM Sequential Model training summary")
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Input features per timestep: {X_train_processed.shape[1]}")
    print(f"Epochs completed: {len(history.history['loss'])}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)

    model.save(MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as metrics_file:
        metrics_file.write(f"Accuracy: {accuracy:.4f}\n\n")
        metrics_file.write(report)
        metrics_file.write("\n\nConfusion Matrix:\n")
        metrics_file.write(str(cm))
        metrics_file.write(f"\n\nTraining set size: {len(X_train)}")
        metrics_file.write(f"\nTest set size: {len(X_test)}")
        metrics_file.write(f"\nSequence length: {SEQUENCE_LENGTH}")
        metrics_file.write(
            f"\nInput features per timestep: {X_train_processed.shape[1]}")
        metrics_file.write(
            f"\nEpochs completed: {len(history.history['loss'])}")
        metrics_file.write(
            "\nNote: The dataset has one row per unique patient; this is a sequence-length-1 LSTM baseline."
        )

    print(f"\nSaved trained LSTM model: {MODEL_PATH}")
    print(f"Saved preprocessing pipeline: {PREPROCESSOR_PATH}")
    print(f"Saved evaluation metrics: {METRICS_PATH}")
    print(f"Saved training history plot: {HISTORY_PLOT_PATH}")


if __name__ == "__main__":
    main()
