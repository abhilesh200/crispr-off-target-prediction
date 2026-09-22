from fastapi import FastAPI, UploadFile
import tensorflow as tf
import numpy as np

app = FastAPI()
model = tf.keras.models.load_model("model/crispr_model.h5", compile=False)

@app.post("/predict")
def predict(data: dict):
    sg = data["sgRNA"]
    off = data["off_seq"]

    # preprocessing here
    prob = float(model.predict([X_seq, X_manual])[0][0])

    return {
        "probability": prob,
        "risk": "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low",
        "heatmap": "heatmap.png"
    }
