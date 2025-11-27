from locust import HttpUser, task, between
from pathlib import Path
import random
import os

# ------------------------------------------
# Resolve correct path to the test images
# ------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TEST_IMAGES_DIR = BASE_DIR / "data" / "test_images"

# Convert to string for compatibility with os.listdir
TEST_IMAGES_DIR = TEST_IMAGES_DIR.resolve()

print(f"[Locust] Loading images from: {TEST_IMAGES_DIR}")

# ------------------------------------------
# Load all images in the folder
# ------------------------------------------
if not TEST_IMAGES_DIR.exists():
    print(f"[Locust ERROR] Test images directory not found: {TEST_IMAGES_DIR}")
    TEST_IMAGES = []
else:
    TEST_IMAGES = [
        str(TEST_IMAGES_DIR / img)
        for img in os.listdir(TEST_IMAGES_DIR)
        if img.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

print(f"[Locust] Loaded {len(TEST_IMAGES)} test images")


# ========================================================================
# LOCUST USER
# ========================================================================
class DermaScanUser(HttpUser):
    wait_time = between(1, 3)  # seconds between requests

    @task
    def predict_random_image(self):
        """Send a random test image to /predict endpoint."""
        if not TEST_IMAGES:
            print("[Locust] No test images found. Skipping request.")
            return

        # Pick a random image
        img_path = random.choice(TEST_IMAGES)

        try:
            with open(img_path, "rb") as f:
                files = {"file": (Path(img_path).name, f, "image/jpeg")}

                response = self.client.post(
                    "/predict",
                    files=files,
                    timeout=30
                )

                try:
                    data = response.json()
                except Exception:
                    data = {"error": "Invalid JSON response"}

                print(f"[Locust] Sent {img_path} → Response: {data}")

        except Exception as e:
            print(f"[Locust ERROR] Could not send image {img_path}: {e}")
