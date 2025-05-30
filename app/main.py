# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from app.model import GestureModel

app = FastAPI()

model = GestureModel(
    model_path="models/XGBoost_Best_model.pkl",
    scaler_path="models/scaler.pkl",
    encoder_path="models/label_encoder.pkl"
)

class Keypoints(BaseModel):
    data: list  # List of 63 floats

@app.post("/predict")
def predict(keypoints: Keypoints):
    if len(keypoints.data) != 63:
        return {"error": "Expected 63 features"}
    
    direction = model.predict(keypoints.data)
    return {"direction": direction}
