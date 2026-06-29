# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Iris Classifier API")

# Loaded once at startup, reused for every request
model = joblib.load("model.joblib")

CLASS_NAMES = ["setosa", "versicolor", "virginica"]

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: IrisFeatures):
    X = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]])
    pred_class = int(model.predict(X)[0])
    pred_proba = model.predict_proba(X)[0].tolist()

    return {
        "predicted_class": pred_class,
        "predicted_species": CLASS_NAMES[pred_class],
        "probabilities": pred_proba,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)