import logging
import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT/"models"/"heart_disease_pipeline.joblib"))
LOG_DIR = ROOT/"logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR/"prediction.log", level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Educational binary classification prototype. Not a medical diagnostic tool.",
    version="1.0.0"
)
model = None

class PatientInput(BaseModel):
    age: float = Field(..., ge=1, le=120)
    sex: Literal[0, 1]
    cp: Literal[1, 2, 3, 4]
    trestbps: float = Field(..., ge=40, le=300)
    chol: float = Field(..., ge=50, le=700)
    fbs: Literal[0, 1]
    restecg: Literal[0, 1, 2]
    thalach: float = Field(..., ge=40, le=250)
    exang: Literal[0, 1]
    oldpeak: float = Field(..., ge=-5, le=15)
    slope: Literal[1, 2, 3]
    ca: float = Field(..., ge=0, le=4)
    thal: float = Field(..., ge=3, le=7)

@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        logging.info("Model loaded from %s", MODEL_PATH)
    else:
        logging.error("Model artifact missing: %s", MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(patient: PatientInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Run training first.")
    try:
        row = pd.DataFrame([patient.model_dump()])
        prediction = int(model.predict(row)[0])
        probability = float(model.predict_proba(row)[0, 1])
        # Log outcome and probability only; do not log patient feature values.
        logging.info("Prediction completed: class=%s probability=%.4f", prediction, probability)
        return {
            "prediction": prediction,
            "label": "Disease indicators detected" if prediction == 1 else "No disease indicators detected",
            "probability": round(probability, 4),
            "disclaimer": "Educational prototype only; not a diagnosis or substitute for medical advice."
        }
    except Exception:
        logging.exception("Prediction failed")
        raise HTTPException(status_code=400, detail="Unable to process this input. Check the values and try again.")
