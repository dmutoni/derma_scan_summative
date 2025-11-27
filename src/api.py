# src/api.py

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pathlib import Path
import shutil
import time
import io

import numpy as np
from PIL import Image

from .preprocessing import (
    load_train_test_datasets,
    load_new_data_dataset,
    NEW_DATA_DIR,
)
from .prediction import get_model
from .model import fine_tune


BASE_DIR = Path(__file__).resolve().parents[1]

app = FastAPI(title="DermaScan API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()

# ---------------------------
# Load datasets & model ONCE
# ---------------------------
train_ds, test_ds, CLASS_NAMES = load_train_test_datasets()
model = get_model()     # loads dermascan_base.h5


# ---------------------------
# API ROOT
# ---------------------------
@app.get("/")
def root():
    return {"status": "DermaScan API is running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "num_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
    }


# ---------------------------
# 1️⃣ PREDICT ENDPOINT
# ---------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts a single image file and returns:
    - class_name
    - confidence
    """

    try:
        # Read image
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Preprocess
        img_resized = img.resize((256, 256))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        preds = model.predict(img_array)
        confidence = float(np.max(preds))
        class_index = int(np.argmax(preds))
        class_name = CLASS_NAMES[class_index]

        # Correct JSON format for Streamlit UI
        return {
            "class_name": class_name,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}


# ---------------------------
# 2️⃣ BULK UPLOAD FOR RETRAINING
# ---------------------------
@app.post("/upload-bulk")
async def upload_bulk(files: List[UploadFile] = File(...)):
    """
    Upload multiple images (bulk) for retraining.
    Expected filename format: eczema_123.jpg, acne_4.png, etc.
    The prefix before '_' is used as class name.
    """
    NEW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0

    for f in files:
        name = f.filename

        # Get class from filename prefix
        try:
            class_name = name.split("_")[0].lower()
        except:
            class_name = "unknown"

        target_dir = NEW_DATA_DIR / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

        dest = target_dir / name
        with open(dest, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)

        saved += 1

    return {"status": "saved", "files_saved": saved}


# ---------------------------
# 3️⃣ RETRAIN ENDPOINT
# ---------------------------
@app.post("/retrain")
def retrain():
    """
    Trigger retraining using new images found in data/new_data.
    Appends new dataset to existing training set.
    """

    global model, train_ds, test_ds

    new_ds = load_new_data_dataset()
    if new_ds is None:
        return {"status": "no_new_data"}

    # Combine existing train data with new data
    combined_train = train_ds.concatenate(new_ds)

    # Retrain
    model, history = fine_tune(
        model,
        combined_train,
        test_ds,
        model_path="dermascan_retrained.h5",
        epochs=5,
    )

    # Clear new_data directory after retraining
    for child in NEW_DATA_DIR.glob("*"):
        if child.is_dir():
            shutil.rmtree(child)

    return {
        "status": "retrained",
        "epochs": len(history.history["accuracy"]),
        "final_train_acc": float(history.history["accuracy"][-1]),
        "final_val_acc": float(history.history["val_accuracy"][-1]),
    }
