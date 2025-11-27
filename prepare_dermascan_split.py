import os
import shutil
import random
from pathlib import Path


BASE = Path("data")
RAW = BASE / "IMG_CLASSES"
TRAIN = BASE / "train"
TEST = BASE / "test"

# Clean class name mapping
CLEAN_NAMES = {
    "1. Eczema 1677": "eczema",
    "3. Atopic Dermatitis - 1.25k": "eczema",

    "9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k": "fungal",

    "2. Melanoma 15.75k": "melanoma",

    "4. Basal Cell Carcinoma (BCC) 3323": "basal_cell_carcinoma",

    "5. Melanocytic Nevi (NV) - 7970": "melanocytic_nevi",

    "6. Benign Keratosis-like Lesions (BKL) 2624": "benign_keratosis",

    "7. Psoriasis pictures Lichen Planus and related diseases - 2k": "psoriasis",

    "8. Seborrheic Keratoses and other Benign Tumors - 1.8k": "seborrheic_keratosis",

    "10. Warts Molluscum and other Viral Infections - 2103": "warts",
}

# acne + normal already exist
EXISTING = {"acne", "normal"}

# ==================================


def create_folders(class_name):
    """Create train/test folders for each clean class."""
    (TRAIN / class_name).mkdir(parents=True, exist_ok=True)
    (TEST / class_name).mkdir(parents=True, exist_ok=True)


def split_and_copy(images, class_name):
    """Split images 80/20 into train/test and copy them."""
    random.shuffle(images)
    split = int(0.8 * len(images))

    train_imgs = images[:split]
    test_imgs = images[split:]

    for img in train_imgs:
        shutil.copy(img, TRAIN / class_name)

    for img in test_imgs:
        shutil.copy(img, TEST / class_name)

    print(f"✔ {class_name}: {len(train_imgs)} → train, {len(test_imgs)} → test")


def main():
    # Create folders for all clean classes
    clean_classes = set(CLEAN_NAMES.values()) | EXISTING
    for cls in clean_classes:
        create_folders(cls)

    # Process raw folders
    for raw_folder in os.listdir(RAW):
        raw_path = RAW / raw_folder
        if not raw_path.is_dir():
            continue

        # Skip anything not in our mapping
        if raw_folder not in CLEAN_NAMES:
            print(f"⚠ Skipping folder (not mapped): {raw_folder}")
            continue

        class_name = CLEAN_NAMES[raw_folder]

        # Collect images
        images = [
            raw_path / f
            for f in os.listdir(raw_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(images) == 0:
            print(f"⚠ No images found in {raw_folder}")
            continue

        split_and_copy(images, class_name)

    print("\n🎉 DONE! All disease folders have been split and cleaned into train/test.")


if __name__ == "__main__":
    main()
