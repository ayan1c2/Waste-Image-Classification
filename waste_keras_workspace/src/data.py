from __future__ import annotations
import random
from pathlib import Path
import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _count_images(root: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def discover_dataset_root(data_dir: str | Path) -> Path:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    # If root already contains >=2 class subdirectories with images, use it.
    class_dirs = [d for d in root.iterdir() if d.is_dir() and _count_images(d) > 0]
    if len(class_dirs) >= 2:
        return root
    # Otherwise find the shallowest directory containing >=2 image-bearing subdirectories.
    candidates = []
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        kids = [k for k in d.iterdir() if k.is_dir() and _count_images(k) > 0]
        if len(kids) >= 2:
            candidates.append((len(d.parts), d))
    if not candidates:
        raise ValueError(
            "Could not detect class folders. Expected data_dir/class_name/image.jpg or a nested equivalent."
        )
    return sorted(candidates, key=lambda x: x[0])[0][1]


def make_datasets(data_dir, image_size=(224, 224), batch_size=32,
                  validation_split=0.15, test_split=0.15, seed=42):
    if validation_split + test_split >= 1.0:
        raise ValueError("validation_split + test_split must be < 1")
    root = discover_dataset_root(data_dir)

    # First create train and holdout. image_dataset_from_directory can only do a 2-way split,
    # so the holdout is split deterministically into validation and test batches afterward.
    holdout = validation_split + test_split
    train_ds = tf.keras.utils.image_dataset_from_directory(
        root, validation_split=holdout, subset="training", seed=seed,
        image_size=image_size, batch_size=batch_size, label_mode="categorical", shuffle=True
    )
    holdout_ds = tf.keras.utils.image_dataset_from_directory(
        root, validation_split=holdout, subset="validation", seed=seed,
        image_size=image_size, batch_size=batch_size, label_mode="categorical", shuffle=False
    )
    class_names = train_ds.class_names

    n_holdout_batches = tf.data.experimental.cardinality(holdout_ds).numpy()
    val_fraction_of_holdout = validation_split / holdout
    n_val = max(1, int(round(n_holdout_batches * val_fraction_of_holdout)))
    val_ds = holdout_ds.take(n_val)
    test_ds = holdout_ds.skip(n_val)

    train_ds = train_ds.cache().shuffle(1000, seed=seed).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)
    return train_ds, val_ds, test_ds, class_names, root


def augmentation_layer() -> tf.keras.Sequential:
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
        tf.keras.layers.RandomContrast(0.10),
    ], name="augmentation")
