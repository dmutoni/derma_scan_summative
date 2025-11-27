# src/api.py

import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pathlib import Path
import shutil
import time
import io
import numpy as np
from PIL import Image

from .prediction import get_model
from .preprocessing import NEW_DATA_DIR

# =========================================================
#  ENVIRONMENT CHECK
# =========================================================
# Render sets this env automatically → used to disable training
IS_RENDER = os.getenv("RENDER") == "true"

# =========================================================
#  FASTAPI INITIALIZATION
# =========================================================
app = FastAPI(title="DermaScan API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()

# =========================================================
#  LOAD MODEL (always available)
# =========================================================
model = get_model()

# Hardcoded class list so Render does NOT need dataset folders
CLASS_NAMES = [
    "acne",
    "basal_cell_carcinoma",
    "benign_keratosis",
    "eczema",
    "fungal",
    "melanocytic_nevi",
    "melanoma",
    "normal",
    "psoriasis",
    "seborrheic_keratosis",
    "warts"
]

# =========================================================
#  LOCAL TRAINING DATA (ONLY when not on Render)
# =========================================================
if not IS_RENDER:
    try:
        from .preprocessing import load_train_test_datasets
        train_ds, test_ds, _ = load_train_test_datasets()
        print("🔵 Loaded dataset locally for training.")
    except Exception as e:
        print("⚠️ Local dataset loading failed:", e)
        train_ds, test_ds = None, None
else:
    train_ds = None
    test_ds = None
    print("🟣 Running on Render — dataset loading & retraining disabled.")


# =========================================================
#  ROOT
# =========================================================
@app.get("/")
def root():
    return {"status": "DermaScan API is running", "render_mode": IS_RENDER}


# =========================================================
#  HEALTH CHECK
# =========================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "running_on_render": IS_RENDER,
        "num_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
    }


# =========================================================
#  PREDICT (Always Works)
# =========================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        img_resized = img.resize((256, 256))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        confidence = float(np.max(preds))
        class_index = int(np.argmax(preds))
        class_name = CLASS_NAMES[class_index]

        return {
            "class_name": class_name,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}


# =========================================================
#  UPLOAD BULK (Enabled locally, simulated on Render)
# =========================================================
@app.post("/upload-bulk")
async def upload_bulk(files: List[UploadFile] = File(...)):
    """
    Upload extra images to /data/new_data for retraining.
    On Render: simulated only.
    """

    # if IS_RENDER:
    #     return {"warning": "Upload disabled on Render", "status": "simulated"}

    NEW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0

    for f in files:
        name = f.filename

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


# =========================================================
#  RETRAIN (Fully disabled on Render, fully functional locally)
# =========================================================
@app.post("/retrain")
def retrain():
    """
    Train the model using:
    - base dataset
    - uploaded new data
    Only works locally.
    """

    # if IS_RENDER:
    #     return {
    #         "status": "disabled_on_render",
    #         "message": "Retraining cannot run on Render. Demo this locally."
    #     }

    if train_ds is None:
        return {"status": "error", "message": "No training dataset available"}

    try:
        from .preprocessing import load_new_data_dataset
        from .model import fine_tune

        new_ds = load_new_data_dataset()
        if new_ds is None:
            return {"status": "no_new_data"}

        combined_train = train_ds.concatenate(new_ds)

        updated_model, history = fine_tune(
            model,
            combined_train,
            test_ds,
            model_path="dermascan_retrained.h5",
            epochs=5,
        )

        return {
            "status": "retrained",
            "epochs": len(history.history["accuracy"]),
            "final_train_acc": float(history.history["accuracy"][-1]),
            "final_val_acc": float(history.history["val_accuracy"][-1]),
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
