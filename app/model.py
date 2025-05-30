import numpy as np
import joblib
import pandas as pd

class GestureModel:
    def __init__(self, model_path, scaler_path, encoder_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.encoder = joblib.load(encoder_path)
        self.feature_names = [
            f"{axis}{i}" for i in range(1, 22) for axis in ("x", "y", "z")
        ]

    def predict(self, keypoints):
        # keypoints: List of 63 floats
        X = pd.DataFrame([keypoints], columns=self.feature_names)
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict(X_scaled)
        label = self.encoder.inverse_transform(pred)[0]

        direction_map = {
            'like': 'up',
            'dislike': 'down',
            'fist': 'left',
            'stop': 'right'
        }
        return direction_map.get(label, 'unknown')
