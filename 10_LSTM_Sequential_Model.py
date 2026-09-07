"""Train and evaluate an LSTM model on the cleaned clinical dataset."""

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
JOBLIB_MODEL_PATH = OUTPUT_DIR / "10_lstm_treatment_outcome_model.joblib"
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
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.30),
            layers.LSTM(32),
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


def evaluate_split(
    model: keras.Model, features: np.ndarray, labels: pd.Series
) -> dict[str, object]:
    probabilities = model.predict(features, batch_size=4096, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "report": classification_report(
            labels, predictions, target_names=["No", "Yes"]),
        "confusion_matrix": confusion_matrix(labels, predictions),
    }


class TestMetricsCallback(keras.callbacks.Callback):
    """Record held-out test metrics after each epoch for fit diagnostics."""

    def __init__(self, test_features: np.ndarray, test_labels: np.ndarray):
        super().__init__()
        self.test_features = test_features
        self.test_labels = test_labels
        self.test_loss: list[float] = []
        self.test_auc: list[float] = []

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        metrics = self.model.evaluate(
            self.test_features,
            self.test_labels,
            batch_size=4096,
            verbose=0,
            return_dict=True,
        )
        self.test_loss.append(float(metrics["loss"]))
        self.test_auc.append(float(metrics["auc"]))


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

    X = df.drop(
        columns=[TARGET_COL, "adverse_event", "readmission_30d"],
        errors="ignore",
    )
    y = df[TARGET_COL]
    numeric_cols = [
        col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])
    ]
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_remaining,
        y_remaining,
        test_size=0.5,
        random_state=42,
        stratify=y_remaining,
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
    X_validation_processed = preprocessor.transform(
        X_validation).astype(np.float32)
    X_test_processed = preprocessor.transform(X_test).astype(np.float32)

    input_width = X_train_processed.shape[1]
    X_train_sequence = X_train_processed.reshape(
        -1, SEQUENCE_LENGTH, input_width)
    X_validation_sequence = X_validation_processed.reshape(
        -1, SEQUENCE_LENGTH, input_width)
    X_test_sequence = X_test_processed.reshape(
        -1, SEQUENCE_LENGTH, input_width)

    model = build_model(input_width)
    class_counts = np.bincount(y_train.to_numpy())
    total = class_counts.sum()
    class_weight = {
        class_id: total / (len(class_counts) * count)
        for class_id, count in enumerate(class_counts)
        if count > 0
    }

    test_metrics_callback = TestMetricsCallback(
        X_test_sequence, y_test.to_numpy())
    history = model.fit(
        X_train_sequence,
        y_train.to_numpy(),
        validation_data=(X_validation_sequence, y_validation.to_numpy()),
        epochs=100,
        batch_size=2048,
        class_weight=class_weight,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            ),
            test_metrics_callback,
        ],
        verbose=2,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Validation Loss")
    axes[0].plot(
        test_metrics_callback.test_loss,
        label="Test Loss (diagnostic)",
        linestyle="--",
    )
    axes[0].set_title("LSTM Loss Over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot(history.history["auc"], label="Train AUC")
    axes[1].plot(history.history["val_auc"], label="Validation AUC")
    axes[1].plot(
        test_metrics_callback.test_auc,
        label="Test AUC (diagnostic)",
        linestyle="--",
    )
    axes[1].set_title("LSTM AUC Over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(HISTORY_PLOT_PATH, dpi=150)
    plt.close(fig)

    split_data = {
        "Training": (X_train_sequence, y_train),
        "Validation": (X_validation_sequence, y_validation),
        "Testing": (X_test_sequence, y_test),
    }
    results = {
        split_name: evaluate_split(model, features, labels)
        for split_name, (features, labels) in split_data.items()
    }

    print("LSTM Sequential Model training summary")
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_validation)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Input features per timestep: {input_width}")
    print(f"Epochs completed: {len(history.history['loss'])}")
    for split_name, metrics in results.items():
        print(f"{split_name} accuracy: {metrics['accuracy']:.4f}")

    model.save(MODEL_PATH)
    joblib.dump({"model": model, "preprocessor": preprocessor},
                JOBLIB_MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as metrics_file:
        for split_name, metrics in results.items():
            metrics_file.write(
                f"{split_name} accuracy: {metrics['accuracy']:.4f}\n\n")
            metrics_file.write(metrics["report"])
            metrics_file.write("\nConfusion Matrix:\n")
            metrics_file.write(str(metrics["confusion_matrix"]))
            metrics_file.write("\n\n")
        metrics_file.write(f"Training set size: {len(X_train)}")
        metrics_file.write(f"\nValidation set size: {len(X_validation)}")
        metrics_file.write(f"\nTest set size: {len(X_test)}")
        metrics_file.write(f"\nSequence length: {SEQUENCE_LENGTH}")
        metrics_file.write(f"\nInput features per timestep: {input_width}")
        metrics_file.write(
            f"\nEpochs completed: {len(history.history['loss'])}")
        metrics_file.write(
            "\nNote: The dataset has one row per unique patient; this is a "
            "sequence-length-1 LSTM baseline."
        )

    print(f"\nSaved trained LSTM model: {MODEL_PATH}")
    print(f"Saved joblib model bundle: {JOBLIB_MODEL_PATH}")
    print(f"Saved preprocessing pipeline: {PREPROCESSOR_PATH}")
    print(f"Saved evaluation metrics: {METRICS_PATH}")
    print(f"Saved training history plot: {HISTORY_PLOT_PATH}")


if __name__ == "__main__":
    main()
