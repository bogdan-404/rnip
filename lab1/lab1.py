from __future__ import annotations

import io
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "train.csv"
IMAGES_DIR = BASE_DIR / "images"
RESULTS_DIR = BASE_DIR / "results"

RANDOM_STATE = 101
BATCH_SIZE = 4096
INITIAL_EPOCHS = 4
WEIGHTED_EPOCHS = 4
OPTIMIZED_EPOCHS = 6


def set_reproducibility(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_directories() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_class_distribution(data: pd.DataFrame) -> Path:
    path = IMAGES_DIR / "01_distributia_clasei_tinta.png"
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=data, x="y", hue="y", palette=["#4C78A8", "#F58518"], legend=False)
    ax.set_title("Distribuția clasei țintă")
    ax.set_xlabel("Clasa țintă y")
    ax.set_ylabel("Număr de înregistrări")
    ax.set_xticklabels(["0 - nu subscrie", "1 - subscrie"])
    for container in ax.containers:
        ax.bar_label(container)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def save_correlation_heatmap(data: pd.DataFrame) -> Path:
    path = IMAGES_DIR / "02_harta_corelatiilor.png"
    numeric_data = data.select_dtypes(include=["number"])
    plt.figure(figsize=(12, 9))
    sns.heatmap(numeric_data.corr(), cmap="coolwarm", center=0, linewidths=0.2)
    plt.title("Harta corelațiilor pentru variabile numerice")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def preprocess_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict[str, dict[str, int]]]:
    clean_data = data.copy()
    if "id" in clean_data.columns:
        clean_data = clean_data.drop(columns=["id"])

    encoders: dict[str, dict[str, int]] = {}
    object_columns = clean_data.select_dtypes(include=["object"]).columns
    for column in object_columns:
        encoder = LabelEncoder()
        clean_data[column] = encoder.fit_transform(clean_data[column])
        encoders[column] = {
            str(class_name): int(encoded_value)
            for encoded_value, class_name in enumerate(encoder.classes_)
        }

    x = clean_data.drop(columns=["y"])
    y = clean_data["y"].astype(int)
    return x, y, clean_data, encoders


def build_initial_model(input_dim: int) -> Sequential:
    model = Sequential(
        [
            Input(shape=(input_dim,), name="intrare"),
            Dense(32, activation="relu", name="strat_dense_1"),
            Dense(16, activation="relu", name="strat_dense_2"),
            Dense(1, activation="sigmoid", name="iesire_sigmoid"),
        ],
        name="model_rna_initial",
    )
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_optimized_model(input_dim: int) -> Sequential:
    model = Sequential(
        [
            Input(shape=(input_dim,), name="intrare"),
            Dense(128, activation="relu", name="dense_128"),
            BatchNormalization(name="normalizare_lot_1"),
            Dropout(0.30, name="abandon_30_1"),
            Dense(64, activation="relu", name="dense_64"),
            BatchNormalization(name="normalizare_lot_2"),
            Dropout(0.25, name="abandon_25"),
            Dense(32, activation="relu", name="dense_32"),
            BatchNormalization(name="normalizare_lot_3"),
            Dropout(0.20, name="abandon_20"),
            Dense(1, activation="sigmoid", name="iesire_sigmoid"),
        ],
        name="model_rna_imbunatatit",
    )
    model.compile(
        optimizer=Adam(learning_rate=0.0007),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def model_summary_text(model: Sequential) -> str:
    stream = io.StringIO()
    model.summary(print_fn=lambda line: stream.write(line + "\n"))
    return stream.getvalue()


def save_history_plot(history: tf.keras.callbacks.History, title: str, filename: str) -> Path:
    path = IMAGES_DIR / filename
    metrics = history.history
    epochs = range(1, len(metrics["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(epochs, metrics["loss"], marker="o", label="Pierdere antrenare")
    axes[0].plot(epochs, metrics["val_loss"], marker="o", label="Pierdere validare")
    axes[0].set_title("Funcția de pierdere")
    axes[0].set_xlabel("Epocă")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(epochs, metrics["accuracy"], marker="o", label="Acuratețe antrenare")
    axes[1].plot(epochs, metrics["val_accuracy"], marker="o", label="Acuratețe validare")
    axes[1].set_title("Acuratețea modelului")
    axes[1].set_xlabel("Epocă")
    axes[1].set_ylabel("Acuratețe")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def evaluate_model(model: Sequential, x_test: np.ndarray, y_test: pd.Series) -> dict[str, object]:
    probabilities = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)
    matrix = confusion_matrix(y_test, predictions)
    report_dict = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, predictions, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report_dict,
        "classification_report_text": report_text,
        "predicted_counts": {
            "0": int((predictions == 0).sum()),
            "1": int((predictions == 1).sum()),
        },
    }


def save_results_plot(results: dict[str, object], title: str, filename: str) -> Path:
    path = IMAGES_DIR / filename
    matrix = np.array(results["confusion_matrix"])
    report_text = str(results["classification_report_text"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["0 - nu", "1 - da"],
        yticklabels=["0 - nu", "1 - da"],
        ax=axes[0],
    )
    axes[0].set_title("Matricea de confuzie")
    axes[0].set_xlabel("Clasa prezisă")
    axes[0].set_ylabel("Clasa reală")

    axes[1].axis("off")
    axes[1].set_title("Raport de clasificare")
    axes[1].text(
        0.0,
        0.98,
        report_text,
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def save_architecture_plot(input_dim: int) -> Path:
    path = IMAGES_DIR / "06_model_rna_imbunatatit.png"
    layers = [
        f"Intrare\n{input_dim} caracteristici",
        "Dense\n128 neuroni, ReLU",
        "BatchNormalization",
        "Dropout\n30%",
        "Dense\n64 neuroni, ReLU",
        "BatchNormalization",
        "Dropout\n25%",
        "Dense\n32 neuroni, ReLU",
        "BatchNormalization",
        "Dropout\n20%",
        "Ieșire\n1 neuron, sigmoid",
    ]

    fig, ax = plt.subplots(figsize=(15, 4.5))
    ax.axis("off")
    x_positions = np.linspace(0.05, 0.95, len(layers))
    for index, (x_pos, label) in enumerate(zip(x_positions, layers)):
        ax.text(
            x_pos,
            0.55,
            label,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#E8F1FF", edgecolor="#4C78A8"),
            fontsize=9,
        )
        if index < len(layers) - 1:
            ax.annotate(
                "",
                xy=(x_positions[index + 1] - 0.035, 0.55),
                xytext=(x_pos + 0.035, 0.55),
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.5),
            )
    ax.set_title("Modelul RNA îmbunătățit", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def class_weights_for(y_train: pd.Series) -> dict[int, float]:
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(class_value): float(weight) for class_value, weight in zip(classes, weights)}


def metrics_summary(results: dict[str, object]) -> dict[str, float]:
    report = results["classification_report"]
    return {
        "acuratete": float(results["accuracy"]),
        "precizie_clasa_1": float(report["1"]["precision"]),
        "recall_clasa_1": float(report["1"]["recall"]),
        "f1_clasa_1": float(report["1"]["f1-score"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
    }


def train_and_evaluate() -> dict[str, object]:
    set_reproducibility()
    ensure_directories()

    data = pd.read_csv(DATA_PATH)
    null_values = data.isna().sum()
    describe = data.describe(include="all").transpose()
    target_distribution = data["y"].value_counts().sort_index()

    class_distribution_image = save_class_distribution(data)
    x, y, encoded_data, encoders = preprocess_data(data)
    correlations_image = save_correlation_heatmap(encoded_data)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    input_dim = x_train_scaled.shape[1]

    initial_model = build_initial_model(input_dim)
    initial_summary = model_summary_text(initial_model)
    initial_history = initial_model.fit(
        x_train_scaled,
        y_train,
        validation_split=0.2,
        epochs=INITIAL_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )
    initial_history_image = save_history_plot(
        initial_history,
        "Figura 5.1. Funcțiile loss și accuracy pentru modelul inițial",
        "03_istoric_model_initial.png",
    )
    initial_results = evaluate_model(initial_model, x_test_scaled, y_test)
    initial_results_image = save_results_plot(
        initial_results,
        "Figura 5.2. Rezultatele clasificării cu RNA inițială",
        "04_rezultate_model_initial.png",
    )

    weights = class_weights_for(y_train)
    weighted_model = build_initial_model(input_dim)
    weighted_summary = model_summary_text(weighted_model)
    weighted_history = weighted_model.fit(
        x_train_scaled,
        y_train,
        validation_split=0.2,
        epochs=WEIGHTED_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=weights,
        verbose=2,
    )
    weighted_history_image = save_history_plot(
        weighted_history,
        "Funcțiile loss și accuracy după adăugarea ponderilor de clasă",
        "05_istoric_ponderi_clase.png",
    )
    weighted_results = evaluate_model(weighted_model, x_test_scaled, y_test)
    weighted_results_image = save_results_plot(
        weighted_results,
        "Figura 5.3. Rezultatele clasificării după ponderarea claselor",
        "06_rezultate_ponderi_clase.png",
    )

    optimized_model = build_optimized_model(input_dim)
    optimized_summary = model_summary_text(optimized_model)
    architecture_image = save_architecture_plot(input_dim)
    optimized_history = optimized_model.fit(
        x_train_scaled,
        y_train,
        validation_split=0.2,
        epochs=OPTIMIZED_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=weights,
        verbose=2,
    )
    optimized_history_image = save_history_plot(
        optimized_history,
        "Funcțiile loss și accuracy pentru modelul RNA îmbunătățit",
        "07_istoric_model_imbunatatit.png",
    )
    optimized_results = evaluate_model(optimized_model, x_test_scaled, y_test)
    optimized_results_image = save_results_plot(
        optimized_results,
        "Rezultatele clasificării cu modelul RNA îmbunătățit",
        "08_rezultate_model_imbunatatit.png",
    )

    outputs = {
        "dataset_shape": list(data.shape),
        "head": data.head(8).to_dict(orient="records"),
        "encoded_head": encoded_data.head(8).to_dict(orient="records"),
        "describe": describe.reset_index(names="coloana").fillna("").to_dict(orient="records"),
        "null_values": {key: int(value) for key, value in null_values.items()},
        "target_distribution": {str(key): int(value) for key, value in target_distribution.items()},
        "encoders": encoders,
        "train_shape": list(x_train.shape),
        "test_shape": list(x_test.shape),
        "class_weights": {str(key): value for key, value in weights.items()},
        "initial_summary": initial_summary,
        "weighted_summary": weighted_summary,
        "optimized_summary": optimized_summary,
        "initial_results": initial_results,
        "weighted_results": weighted_results,
        "optimized_results": optimized_results,
        "metrics_summary": {
            "model_initial": metrics_summary(initial_results),
            "model_ponderi_clase": metrics_summary(weighted_results),
            "model_imbunatatit": metrics_summary(optimized_results),
        },
        "images": {
            "class_distribution": class_distribution_image.name,
            "correlations": correlations_image.name,
            "initial_history": initial_history_image.name,
            "initial_results": initial_results_image.name,
            "weighted_history": weighted_history_image.name,
            "weighted_results": weighted_results_image.name,
            "architecture": architecture_image.name,
            "optimized_history": optimized_history_image.name,
            "optimized_results": optimized_results_image.name,
        },
    }

    (RESULTS_DIR / "rezultate.json").write_text(
        json.dumps(outputs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (RESULTS_DIR / "rezumat_model_initial.txt").write_text(initial_summary, encoding="utf-8")
    (RESULTS_DIR / "rezumat_model_ponderi_clase.txt").write_text(weighted_summary, encoding="utf-8")
    (RESULTS_DIR / "rezumat_model_imbunatatit.txt").write_text(optimized_summary, encoding="utf-8")

    return outputs


if __name__ == "__main__":
    train_and_evaluate()
