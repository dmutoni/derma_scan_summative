from pathlib import Path
import sys

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def rename_images(root_dir: Path):
    root_dir = root_dir.resolve()
    if not root_dir.exists():
        print(f"❌ Root directory does not exist: {root_dir}")
        return

    print(f"📁 Root directory: {root_dir}")

    # Go through each subfolder (class folder: acne, basal_cell_carcinoma, etc.)
    for class_dir in sorted(root_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        folder_name = class_dir.name
        prefix = folder_name.replace(" ", "_").replace("-", "_").lower()

        print(f"\n➡ Processing folder: {folder_name}  (prefix='{prefix}_')")

        for img_path in sorted(class_dir.iterdir()):
            if not img_path.is_file():
                continue

            ext = img_path.suffix.lower()
            if ext not in VALID_EXTS:
                continue

            # Skip if already has the prefix
            if img_path.name.startswith(prefix + "_"):
                print(f"   ⏭ Already prefixed: {img_path.name}")
                continue

            new_name = f"{prefix}_{img_path.name}"
            new_path = img_path.with_name(new_name)

            try:
                img_path.rename(new_path)
                print(f"   ✔ {img_path.name}  →  {new_name}")
            except Exception as e:
                print(f"   ❌ Failed to rename {img_path.name}: {e}")

    print("\n🎉 Done renaming images.")


if __name__ == "__main__":
    # ---- HOW TO CALL IT ----
    # Example for your project:
    #   python rename_images.py data/test
    #
    # Or for train:
    #   python rename_images.py data/train

    if len(sys.argv) < 2:
        print("Usage: python rename_images.py <root_folder>")
        print("Example: python rename_images.py data/test")
        sys.exit(1)

    target = Path(sys.argv[1])
    rename_images(target)
