# src/prediction.py

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from pathlib import Path
import io
from PIL import Image

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "dermascan_base.h5"

# ---------------------------
# LOAD MODEL ONCE
# ---------------------------
def get_model():
    """
    Loads the trained model once and returns it.
    Called by API at startup.
    """
    print(f"Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    return model


# ---------------------------
# IMAGE PREPROCESSING
# ---------------------------
def preprocess_image(img: Image.Image):
    """
    Preprocess a PIL Image object:
    - Normalize
    - Expand batch dimension
    """
    img = img.resize((256, 256))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ---------------------------
# PREDICT FROM RAW BYTES
# ---------------------------
def predict_from_bytes(model, img_bytes: bytes, class_names: list):
    """
    Predicts from image bytes (uploaded file).
    
    Returns:
    {
        "class_name": "...",
        "confidence": 0.94
    }
    """

    # Load image
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Preprocess
    img_array = preprocess_image(img)

    # Predict
    preds = model.predict(img_array)
    confidence = float(np.max(preds))
    class_index = int(np.argmax(preds))
    class_name = class_names[class_index]

    return {
        "class_name": class_name,
        "confidence": confidence
    }
