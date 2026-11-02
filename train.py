
from pathlib import Path
import random

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from model import build_unet, combined_loss


# Project settings
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 512
BATCH_SIZE = 2
EPOCHS = 3

# Start with 600 samples to limit training time.
# Increase this after checking the initial results.
MAX_SAMPLES = 600

DATA_DIR = Path("dataset") / "processed"
IMAGE_DIR = DATA_DIR / "images"
MASK_DIR = DATA_DIR / "masks"

MODEL_DIR = Path("models")

# Save to a NEW file; preserve the existing model.
MODEL_PATH = MODEL_DIR / "lane_unet_v2.keras"

SEED = 42


def load_image(image_path, mask_path):
    """Load an RGB road image and its binary lane mask."""

    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(
        image, [IMAGE_HEIGHT, IMAGE_WIDTH]
    )
    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.io.decode_png(mask, channels=1)
    mask = tf.image.resize(
        mask,
        [IMAGE_HEIGHT, IMAGE_WIDTH],
        method="nearest"
    )

    mask = tf.cast(mask > 0, tf.float32)

    return image, mask


def dice_score(y_true, y_pred):
    """Measure overlap between predicted and actual lane pixels."""

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true * y_pred)

    return (
        2.0 * intersection + 1.0
    ) / (
        tf.reduce_sum(y_true) +
        tf.reduce_sum(y_pred) + 1.0
    )


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    if not IMAGE_DIR.exists() or not MASK_DIR.exists():
        print("ERROR: Processed dataset folders are missing.")
        return

    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))

    mask_lookup = {
        path.stem: path
        for path in MASK_DIR.glob("*.png")
    }

    pairs = [
        (str(image_path), str(mask_lookup[image_path.stem]))
        for image_path in image_paths
        if image_path.stem in mask_lookup
    ]

    if len(pairs) < 2:
        print("ERROR: Not enough matching image-mask pairs.")
        return

    random.shuffle(pairs)
    pairs = pairs[:MAX_SAMPLES]

    train_pairs, val_pairs = train_test_split(
        pairs,
        test_size=0.2,
        random_state=SEED
    )

    def make_dataset(samples, training=False):
        image_files = [pair[0] for pair in samples]
        mask_files = [pair[1] for pair in samples]

        dataset = tf.data.Dataset.from_tensor_slices(
            (image_files, mask_files)
        )

        dataset = dataset.map(
            load_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        if training:
            dataset = dataset.shuffle(
                buffer_size=min(len(samples), 500),
                seed=SEED
            )

        return dataset.batch(BATCH_SIZE).prefetch(
            tf.data.AUTOTUNE
        )

    train_dataset = make_dataset(
        train_pairs, training=True
    )
    val_dataset = make_dataset(val_pairs)

    print(f"Matched pairs: {len(pairs)}")
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")
    print(f"Epochs: {EPOCHS}")
    print(f"New model will be saved to: {MODEL_PATH}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model = build_unet(
        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    )

    # Explicitly ensure the intended loss and metrics are used.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0001
        ),
        loss=combined_loss,
        metrics=["binary_accuracy", dice_score]
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print("\nTraining finished.")

    for metric_name, values in history.history.items():
        print(
            f"{metric_name}: "
            f"last={values[-1]:.4f}, "
            f"best={min(values) if 'loss' in metric_name else max(values):.4f}"
        )

    if MODEL_PATH.exists():
        print(f"Best checkpoint saved to: {MODEL_PATH}")
    else:
        print("No checkpoint was saved. Check the training output.")


if __name__ == "__main__":
    main()
