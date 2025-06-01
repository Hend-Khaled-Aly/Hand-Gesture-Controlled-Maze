# 🧠 Hand Gesture Recognition API – MLOps Final Project

This is the production API that serves a trained hand gesture recognition model used for controlling a maze game via hand signs. It connects to a frontend app that displays the maze and interprets predictions into movement.

---

## 🔍 Introduction

This backend project is part of a full MLOps pipeline that includes:

- Loading and serving a trained machine learning model
- REST API interface using FastAPI
- Containerization with Docker
- Monitoring with Prometheus + Grafana
- Deployment via DockerHub/Railway
- Frontend integration for a complete user experience

👉 [Frontend Repo]([https://github.com/IshraqAhmedJamaluddin/MLOPs-Final-Project](https://github.com/Hend-Khaled-Aly/Frontend_Hand-Gesture-Controlled-Maze))

---

## 📁 Repository Structure
-  app/
  * main.py -> FastAPI entry (CORS enabled)
  * model.py -> Model loading and prediction
- models/
  * XGBoost_Best_model.pkl
  * label_encoder.pkl
  * scaler.pkl
- tests/
  * test_api.py -> Unit tests
- prometheus/ -> Prometheus config
  * prometheus.yml
- .github/workflows/
  * deploy.yml
- docker-compose.yml -> system monitoring
- Dockerfile
- docker-compose.yml
- requirements.txt
- README.md

## Monitoring Metrics
collected and visualized the following metrics using Prometheus + Grafana:
1. Model-related — python_gc_collections_total
Tracks Python garbage collection events.
🔍 Why? Frequent collections can indicate memory pressure from large model objects during prediction.
2. Data-related — http_requests_total
Counts total HTTP requests received.
🔍 Why? Useful to analyze traffic volume, user interaction frequency, and detect frontend issues.
3. Server-related — process_resident_memory_bytes
Monitors current memory usage by the API container.
🔍 Why? Critical for spotting memory leaks or inefficiencies in model loading.

