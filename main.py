from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import io
import os
import re

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

MODEL_DIR = "models"

# Load labels once
with open("class_names.json", "r") as f:
    labels = json.load(f)

# Cache models so we don't reload from disk every request
MODEL_CACHE = {}


def get_model(epoch: int):
    # Matches training save format: models/model_01.keras
    model_filename = os.path.join(MODEL_DIR, f"model_{epoch:02d}.keras")

    if not os.path.exists(model_filename):
        return None, model_filename

    if epoch not in MODEL_CACHE:
        MODEL_CACHE[epoch] = tf.keras.models.load_model(model_filename)

    return MODEL_CACHE[epoch], model_filename


@app.get("/versions")
def list_versions():
    """
    Optional helper: returns available epoch versions from /models folder.
    """
    if not os.path.exists(MODEL_DIR):
        return {"available_epochs": []}

    files = [f for f in os.listdir(MODEL_DIR) if re.match(r"model_\d{2}\.keras$", f)]
    epochs = sorted([int(f.replace("model_", "").replace(".keras", "")) for f in files])
    return {"available_epochs": epochs}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    epoch: int = Query(1, ge=1, le=99)  # min 1 max 99
):
    # 1) Load model for requested epoch/version
    model, model_filename = get_model(epoch)
    if model is None:
        return {
            "error": f"Version {epoch} not found. Train it first.",
            "file_expected": model_filename
        }

    # 2) Image processing
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))

    # IMPORTANT: NO /255.0 here
    # Because this model already includes preprocess_input inside it.
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)

    # 3) Prediction
    preds = model.predict(img_array, verbose=0)
    conf = float(np.max(preds[0]))
    class_idx = int(np.argmax(preds[0]))

    return {
        "disease": labels[class_idx].replace("_", " "),
        "confidence": f"{conf * 100:.2f}%",
        "version": f"Epoch {epoch}",
        "file_used": model_filename
    }
