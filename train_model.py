"""Entraînement reproductible de MobileNetV2 sur Intel Image Classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "mobilenetv2_intel.keras"
LABELS_PATH = ARTIFACT_DIR / "class_names.json"


def discover_train_dir(root: Path) -> Path | None:
    """Trouve le répertoire de classes sans scanner tout le dataset récursivement."""
    root = root.resolve()

    wrappers = {"seg_train", "seg_test", "seg_pred", "train", "test", "validation", "val"}
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def is_class_directory(path: Path) -> bool:
        if not path.is_dir():
            return False
        children = [child for child in path.iterdir() if child.is_dir()]
        if len(children) < 2 or any(child.name.lower() in wrappers for child in children):
            return False
        image_classes = sum(
            any(file.is_file() and file.suffix.lower() in image_suffixes for file in child.iterdir())
            for child in children
        )
        return image_classes >= 2

    candidates = [
        root / "seg_train" / "seg_train",
        root / "seg_train",
        root / "train" / "train",
        root / "train",
        root,
    ]
    for candidate in candidates:
        if is_class_directory(candidate):
            return candidate
    return None


def train_model(
    train_dir: Path,
    max_epochs: int = 15,
    on_epoch: Callable[[int, int, float], None] | None = None,
    on_batch: Callable[[str, int, int, int, int, float], None] | None = None,
) -> dict:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(123)
    image_size = (224, 224)
    batch_size = 32
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int",
    )
    class_names = train_ds.class_names
    steps_per_epoch = int(tf.data.experimental.cardinality(train_ds).numpy())
    validation_steps = int(tf.data.experimental.cardinality(val_ds).numpy())
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*image_size, 3), include_top=False, weights="imagenet"
    )
    backbone.trainable = False
    inputs = tf.keras.Input(shape=(*image_size, 3))
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_intel")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    warmup_path = ARTIFACT_DIR / "_warmup_best.keras"
    fine_tune_path = ARTIFACT_DIR / "_fine_tune_best.keras"

    class ProgressCallback(tf.keras.callbacks.Callback):
        def __init__(self, offset: int, total: int):
            super().__init__()
            self.offset, self.total = offset, total

        def on_epoch_begin(self, epoch, logs=None):
            self.current_epoch = epoch

        def on_train_batch_end(self, batch, logs=None):
            self.report_batch("Entraînement", batch, steps_per_epoch, batch + 1)

        def on_test_batch_end(self, batch, logs=None):
            completed_steps = steps_per_epoch + batch + 1
            self.report_batch("Validation", batch, validation_steps, completed_steps)

        def report_batch(self, phase: str, batch: int, phase_steps: int, completed_steps: int):
            update_interval = max(1, phase_steps // 50)
            if on_batch and ((batch + 1) % update_interval == 0 or batch + 1 == phase_steps):
                epoch = self.offset + self.current_epoch + 1
                total_steps = steps_per_epoch + validation_steps
                on_batch(
                    phase,
                    epoch,
                    self.total,
                    batch + 1,
                    phase_steps,
                    completed_steps / total_steps,
                )

        def on_epoch_end(self, epoch, logs=None):
            if on_epoch:
                on_epoch(self.offset + epoch + 1, self.total, float((logs or {}).get("val_accuracy", 0)))

    def callbacks(checkpoint_path: Path, offset: int, total: int):
        return [
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path, monitor="val_accuracy", mode="max", save_best_only=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", mode="max", patience=4, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7
            ),
            ProgressCallback(offset, total),
        ]

    total = max(3, int(max_epochs))
    warmup_epochs = min(4, max(2, total // 3))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=warmup_epochs,
        callbacks=callbacks(warmup_path, 0, total),
        verbose=0,
    )
    # Fine-tune only the last backbone layers; keep BatchNorm statistics frozen.
    backbone.trainable = True
    for layer in backbone.layers[:-30]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=warmup_epochs,
        epochs=total,
        callbacks=callbacks(fine_tune_path, warmup_epochs, total),
        verbose=0,
    )

    warmup_model = tf.keras.models.load_model(warmup_path)
    warmup_loss, warmup_accuracy = warmup_model.evaluate(val_ds, verbose=0)
    fine_tune_model = tf.keras.models.load_model(fine_tune_path)
    fine_tune_loss, fine_tune_accuracy = fine_tune_model.evaluate(val_ds, verbose=0)
    best_model, val_loss, val_accuracy = (
        (fine_tune_model, fine_tune_loss, fine_tune_accuracy)
        if fine_tune_accuracy >= warmup_accuracy
        else (warmup_model, warmup_loss, warmup_accuracy)
    )
    best_model.save(MODEL_PATH)
    LABELS_PATH.write_text(json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8")
    warmup_path.unlink(missing_ok=True)
    fine_tune_path.unlink(missing_ok=True)
    return {"val_loss": float(val_loss), "val_accuracy": float(val_accuracy), "classes": class_names}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Dossier du dataset ou dossier seg_train")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    directory = discover_train_dir(args.dataset)
    if directory is None:
        parser.error("Dossier seg_train/train introuvable dans le chemin indiqué.")
    result = train_model(directory, max_epochs=args.epochs)
    print(json.dumps(result, ensure_ascii=False, indent=2))
