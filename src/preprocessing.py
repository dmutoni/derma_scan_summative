# src/preprocessing.py

import tensorflow as tf
from pathlib import Path

# -------------------------------------
# DIRECTORY DEFINITIONS
# -------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
NEW_DATA_DIR = DATA_DIR / "new_data"   # holds uploaded training images


# -------------------------------------
# BASE DATASET LOADING
# -------------------------------------
def load_train_test_datasets(img_size=(256, 256), batch_size=32):
    """
    Loads the pre-split train/ and test/ folders
    Returns:
        train_ds, test_ds, class_names
    """

    print("📌 Loading train dataset from:", TRAIN_DIR)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
    )

    print("📌 Loading test dataset from:", TEST_DIR)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )

    class_names = train_ds.class_names
    print("📌 Classes detected:", class_names)

    # Normalize (MobileNet expects values 0-1)
    train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
    test_ds = test_ds.map(lambda x, y: (x / 255.0, y))

    # Cache + prefetch for performance
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds, class_names


# -------------------------------------
# NEW DATASET LOADING (RETRAINING)
# -------------------------------------
def load_new_data_dataset(img_size=(256, 256), batch_size=32):
    """
    Loads new images uploaded to data/new_data/
    These images will be used to retrain the model.
    
    Returns:
        new_ds  OR  None if no data exists
    """

    if not NEW_DATA_DIR.exists():
        print("⚠️ No new_data/ directory found.")
        return None

    # Check if directory has any files
    if not any(NEW_DATA_DIR.glob("*/*")):
        print("⚠️ No new images to retrain on.")
        return None

    print("📌 Loading NEW retraining data from:", NEW_DATA_DIR)

    new_ds = tf.keras.utils.image_dataset_from_directory(
        NEW_DATA_DIR,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
    )

    new_ds = new_ds.map(lambda x, y: (x / 255.0, y))
    new_ds = new_ds.cache().prefetch(tf.data.AUTOTUNE)

    return new_ds
