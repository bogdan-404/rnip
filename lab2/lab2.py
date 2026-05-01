from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import InceptionV3, ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "DATASET" / "TRAIN"
TEST_DIR = BASE_DIR / "DATASET" / "TEST"
IMAGES_DIR = BASE_DIR / "images"
RESULTS_DIR = BASE_DIR / "results"

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42
EPOCHS_CNN = 3
EPOCHS_TRANSFER = 3


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dirs() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def pasul1_info() -> dict[str, object]:
    info: dict[str, object] = {
        "train": str(TRAIN_DIR),
        "test": str(TEST_DIR),
        "clase_train": sorted(p.name for p in TRAIN_DIR.iterdir() if p.is_dir()),
    }
    counts_train = {}
    for cls in info["clase_train"]:
        n = len(list((TRAIN_DIR / cls).glob("*.jpg")))
        counts_train[cls] = n
    info["numar_imagini_train_pe_clasa"] = counts_train
    counts_test = {}
    for cls in sorted(p.name for p in TEST_DIR.iterdir() if p.is_dir()):
        counts_test[cls] = len(list((TEST_DIR / cls).glob("*.jpg")))
    info["numar_imagini_test_pe_clasa"] = counts_test
    return info


def collect_paths_labels(folder: Path) -> tuple[list[str], list[int], dict[str, int]]:
    classes = sorted(d.name for d in folder.iterdir() if d.is_dir())
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    paths: list[str] = []
    labels: list[int] = []
    for cls in classes:
        for p in (folder / cls).glob("*.jpg"):
            paths.append(str(p))
            labels.append(class_to_idx[cls])
    return paths, labels, class_to_idx


def pasul2_primele_10(paths: list[str], labels: list[int], class_names: list[str]) -> Path:
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.ravel()
    for i in range(10):
        img = tf.io.read_file(paths[i])
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.cast(img, tf.uint8).numpy()
        axes[i].imshow(img)
        axes[i].set_title(f"Clasa {labels[i]} ({class_names[labels[i]]})")
        axes[i].axis("off")
    fig.suptitle("Primele 10 imagini (inainte de amestecare)")
    plt.tight_layout()
    out = IMAGES_DIR / "01_primele_10_imagini.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def pasul3_amesteca(paths: list[str], labels: list[int]) -> tuple[list[str], list[int]]:
    pairs = list(zip(paths, labels))
    random.shuffle(pairs)
    p2, l2 = zip(*pairs)
    return list(p2), list(l2)


def pasul6_dataset_antrenament() -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset="training",
        seed=SEED,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)
    return train_ds, val_ds, test_ds


def pasul7_exemplu_per_clasa(train_ds: tf.data.Dataset, class_names: list[str]) -> Path:
    found: dict[int, np.ndarray] = {}
    for batch_x, batch_y in train_ds:
        for i in range(batch_x.shape[0]):
            lab = int(batch_y[i].numpy())
            if lab not in found:
                img = batch_x[i].numpy()
                if img.max() <= 1.0:
                    img = (img * 255.0).clip(0, 255).astype("uint8")
                else:
                    img = img.astype("uint8")
                found[lab] = img
            if len(found) >= 2:
                break
        if len(found) >= 2:
            break
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, (lab, img) in zip(axes, sorted(found.items())):
        ax.imshow(img)
        ax.set_title(f"Exemplu clasa {lab} ({class_names[lab]})")
        ax.axis("off")
    fig.suptitle("Cate un exemplu din fiecare clasa (antrenament)")
    plt.tight_layout()
    out = IMAGES_DIR / "02_exemplu_fiecare_clasa.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def build_augmentation() -> Sequential:
    return Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(height_factor=(-0.3, -0.2), width_factor=(-0.3, -0.2)),
        ],
        name="augmentare_date",
    )


def pasul8_model_cnn() -> Sequential:
    aug = build_augmentation()
    model = Sequential(name="cnn_lucrare")
    model.add(layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(aug)
    model.add(layers.Rescaling(1.0 / 255))
    model.add(layers.Conv2D(filters=32, kernel_size=(7, 7), padding="same"))
    model.add(layers.Activation("relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), padding="valid"))
    model.add(layers.Activation("relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding="valid"))
    model.add(layers.Activation("relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding="valid"))
    model.add(layers.Activation("relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding="valid"))
    model.add(layers.Activation("relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.GlobalMaxPooling2D())
    model.add(layers.Dense(units=256))
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(units=1))
    model.add(layers.Activation("sigmoid"))
    return model


def build_resnet_transfer() -> tf.keras.Model:
    aug = build_augmentation()
    base = ResNet50(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = aug(inputs)
    x = layers.Lambda(lambda img: tf.keras.applications.resnet50.preprocess_input(img))(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="resnet50_transfer")


def build_inception_transfer() -> tf.keras.Model:
    aug = build_augmentation()
    base = InceptionV3(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = aug(inputs)
    x = layers.Lambda(lambda img: tf.keras.applications.inception_v3.preprocess_input(img))(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="inceptionv3_transfer")


def plot_history(history: tf.keras.callbacks.History, title: str, fname: str) -> Path:
    path = IMAGES_DIR / fname
    h = history.history
    epochs = range(1, len(h["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, h["loss"], marker="o", label="Pierdere antrenare")
    axes[0].plot(epochs, h["val_loss"], marker="o", label="Pierdere validare")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoca")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, h["accuracy"], marker="o", label="Acuratete antrenare")
    axes[1].plot(epochs, h["val_accuracy"], marker="o", label="Acuratete validare")
    axes[1].set_title("Acuratete")
    axes[1].set_xlabel("Epoca")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def evaluate_binary(model: tf.keras.Model, test_ds: tf.data.Dataset) -> dict[str, float]:
    loss, acc = model.evaluate(test_ds, verbose=0)
    return {"loss": float(loss), "accuracy": float(acc)}


def transfer_only() -> None:
    set_seed()
    ensure_dirs()
    train_ds, val_ds, test_ds = pasul6_dataset_antrenament()
    epochs_transfer = min(2, EPOCHS_TRANSFER)
    path = RESULTS_DIR / "rezultate.json"
    if path.exists():
        rezultate = json.loads(path.read_text(encoding="utf-8"))
    else:
        rezultate = {}

    resnet = build_resnet_transfer()
    resnet.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    hist_res = resnet.fit(
        train_ds, validation_data=val_ds, epochs=epochs_transfer, verbose=1
    )
    plot_history(hist_res, "Antrenare ResNet50 (transfer)", "04_istoric_resnet50.png")
    rezultate["resnet50_test"] = evaluate_binary(resnet, test_ds)
    print("Evaluare ResNet50 pe test:", rezultate["resnet50_test"])

    inception = build_inception_transfer()
    inception.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    hist_inc = inception.fit(
        train_ds, validation_data=val_ds, epochs=epochs_transfer, verbose=1
    )
    plot_history(hist_inc, "Antrenare InceptionV3 (transfer)", "05_istoric_inceptionv3.png")
    rezultate["inceptionv3_test"] = evaluate_binary(inception, test_ds)
    print("Evaluare InceptionV3 pe test:", rezultate["inceptionv3_test"])

    path.write_text(json.dumps(rezultate, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Actualizat", path)


def main() -> None:
    set_seed()
    ensure_dirs()

    info = pasul1_info()
    (RESULTS_DIR / "info_dataset.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    paths, labels, class_to_idx = collect_paths_labels(TRAIN_DIR)
    class_names = sorted(class_to_idx, key=lambda k: class_to_idx[k])
    print("Pasul 1 — informatii dataset:", json.dumps(info, ensure_ascii=False, indent=2))
    print("Etichete numerice (binar):", class_to_idx)

    out10 = pasul2_primele_10(paths, labels, class_names)
    print("Pasul 2 — salvat:", out10)

    paths_s, labels_s = pasul3_amesteca(paths, labels)
    print("Pasul 3 — amestecat, primul path:", paths_s[0])

    train_ds, val_ds, test_ds = pasul6_dataset_antrenament()
    print("Pasul 6 — dataset antrenament / validare / test pregatite")

    out_ex = pasul7_exemplu_per_clasa(train_ds, class_names)
    print("Pasul 7 — salvat:", out_ex)

    cnn = pasul8_model_cnn()
    print("Pasul 8–9 — CNN creat. Rezumat model:")
    cnn.summary()

    cnn.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    hist_cnn = cnn.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_CNN, verbose=1)
    plot_history(
        hist_cnn,
        "Antrenare CNN (de la zero)",
        "03_istoric_cnn.png",
    )
    metrics_cnn = evaluate_binary(cnn, test_ds)
    print("Pasul 11 — evaluare CNN pe test:", metrics_cnn)

    resnet = build_resnet_transfer()
    print("Model ResNet50 (transfer). Rezumat:")
    resnet.summary()
    resnet.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    hist_res = resnet.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_TRANSFER, verbose=1)
    plot_history(hist_res, "Antrenare ResNet50 (transfer)", "04_istoric_resnet50.png")
    metrics_res = evaluate_binary(resnet, test_ds)
    print("Evaluare ResNet50 pe test:", metrics_res)

    inception = build_inception_transfer()
    print("Model InceptionV3 (transfer). Rezumat:")
    inception.summary()
    inception.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    hist_inc = inception.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS_TRANSFER, verbose=1
    )
    plot_history(hist_inc, "Antrenare InceptionV3 (transfer)", "05_istoric_inceptionv3.png")
    metrics_inc = evaluate_binary(inception, test_ds)
    print("Evaluare InceptionV3 pe test:", metrics_inc)

    rezultate = {
        "dataset": info,
        "etichete": class_to_idx,
        "cnn_test": metrics_cnn,
        "resnet50_test": metrics_res,
        "inceptionv3_test": metrics_inc,
        "imagini": {
            "primele_10": out10.name,
            "exemplu_clase": out_ex.name,
            "istoric_cnn": "03_istoric_cnn.png",
            "istoric_resnet": "04_istoric_resnet50.png",
            "istoric_inception": "05_istoric_inceptionv3.png",
        },
    }
    (RESULTS_DIR / "rezultate.json").write_text(
        json.dumps(rezultate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("Rezultate salvate in", RESULTS_DIR / "rezultate.json")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--transfer-only":
        transfer_only()
    else:
        main()
