# Hand Gesture Classification Using MediaPipe Landmarks and HaGRID Dataset

## 📖 Project Overview

This project focuses on classifying hand gestures using landmark data extracted from the HaGRID (Hand Gesture Recognition Image Dataset) with **MediaPipe**. The goal is to build a machine learning model capable of recognizing 18 distinct hand gestures based on the 3D coordinates (x, y, z) of 21 hand landmarks.

These predicted gestures will enable navigation through a maze game using hand signs, providing a natural and interactive interface.

---

## 📂 Dataset Description

- **Dataset:** HaGRID (Hand Gesture Recognition Image Dataset)  
- **Number of Classes:** 18  
- **Gesture Labels:** Call, Dislike, Fist, Four, Like, Mute, Ok, One, Palm, Peace, Peace Inverted, Rock, Stop, Stop Inverted, Three, Three2, Two Up, Two Up Inverted.  
- **Input Features:** Each gesture is represented by 21 hand landmarks, each with x, y, z coordinates, extracted using MediaPipe.  
- **Data Format:** CSV file containing landmark coordinates alongside their gesture labels.

---

## 🧪 Model Training and Experiment Tracking with MLflow

Multiple models were trained and evaluated using the landmark data. To track and manage these experiments efficiently, **MLflow** was used to log parameters, metrics, and model artifacts.

### Models Evaluated

- **XGBoost**  
- **Support Vector Machine (SVM)**  
- **Random Forest**

### MLflow Tracking

- Every training run logs accuracy, precision, recall, F1-score, and training time.  
- Experiment artifacts include serialized models and preprocessing objects (scalers, encoders).  
- Allows easy comparison and reproducibility of results.

---

## 📊 Model Comparison

| Model          | Accuracy | Precision | Recall  | F1-Score | Training Time  |
|----------------|----------|-----------|---------|----------|----------------|
| **XGBoost**        | 0.9748   | 0.9751    | 0.9749  | 0.9750   | 20.0 seconds   |
| **SVM**            | 0.9616   | 0.9620    | 0.9620  | 0.9620   | 20.2 seconds   |
| **Random Forest**   | 0.9529   | 0.9596    | 0.9593  | —        | 1.7 minutes    |

---

## ✅ Selected Model for Production

The **XGBoost** model was selected for deployment based on:

- Highest accuracy (~97.5%) and balanced classification performance.  
- Fast training and inference time.  
- Lightweight model size suitable for real-time usage.

---

## 🚀 Loading and Using the Selected Model

The chosen model is saved as an MLflow artifact, and i moved it to production stage

---
Project File Structure

├── hand_landmarks_data.csv      # CSV dataset with hand landmarks and labels
├── XGBoost_Best_model.pkl       # Serialized XGBoost model for production use
├── label_encoder.pkl            # Label encoder for gesture classes
├── scaler.pkl                   # Scaler for feature normalization
├── train.py                     # Script to train and log models with MLflow
├── mlruns/                      # Directory containing MLflow experiment logs
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
