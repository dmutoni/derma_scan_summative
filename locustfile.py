# locustfile.py

from locust import HttpUser, task, between
import os
import random

# Path to a folder containing test images for load testing
TEST_IMAGES_DIR = "test_images"   


class DermaScanUser(HttpUser):
    """
    Simulates a user sending repeated prediction requests to the API.
    """
    wait_time = between(1, 3)  # wait 1–3 sec between requests

    @task
    def predict_image(self):
        """
        Sends a POST request to /predict with a real image file.
        """

        # Pick a random image from the test_images folder
        images = os.listdir(TEST_IMAGES_DIR)
        if not images:
            print("⚠️ No images found in test_images/. Add some JPG files.")
            return

        img_name = random.choice(images)
        img_path = os.path.join(TEST_IMAGES_DIR, img_name)

        with open(img_path, "rb") as img:
            files = {"file": (img_name, img, "image/jpeg")}

            # Send request to your FastAPI server
            self.client.post("/predict", files=files)
