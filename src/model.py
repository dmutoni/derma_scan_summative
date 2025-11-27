# src/model.py

import tensorflow as tf
from pathlib import Path

# -------------------------------------
# MODEL LOADING
# -------------------------------------

def get_model(model_path: str = "models/dermascan_base.h5"):
    """
    Loads the model from the given .h5 path.
    If running first time, it loads the base pretrained model.
    """
    model_path = Path(__file__).resolve().parents[1] / model_path

    print(f"📌 Loading model from: {model_path}")

    model = tf.keras.models.load_model(model_path)
    model.trainable = True  # ensure layers are trainable for fine-tuning

    return model


# -------------------------------------
# FINE-TUNING LOGIC
# -------------------------------------

def fine_tune(model, train_ds, val_ds, model_path="models/dermascan_retrained.h5", epochs=5):
    """
    Fine-tune the model using combined training data.
    
    Args:
        model: the loaded Keras model
        train_ds: the training dataset
        val_ds: validation dataset
        model_path: where to save retrained model
        epochs: number of epochs to fine-tune

    Returns:
        model, history
    """

    print("🔧 Starting fine-tuning...")

    # Compile using MobileNet-friendly optimizer
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1
    )

    # Save retrained model
    save_path = Path(__file__).resolve().parents[1] / model_path
    print(f"💾 Saving retrained model to: {save_path}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)

    print("✅ Fine-tuning complete")

    return model, history
