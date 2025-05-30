from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.main import app
client = TestClient(app)

def test_predict_valid_input():
    data = {
        "data": [0.1] * 63  # dummy input
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert "direction" in response.json()

def test_predict_invalid_input():
    data = {
        "data": [0.1] * 10  # too short
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert response.json()["error"] == "Expected 63 features"
