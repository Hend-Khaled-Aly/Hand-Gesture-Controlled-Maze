import pandas as pd
import numpy as np
import pickle
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set MLflow experiment
mlflow.set_experiment("hand_gesture_experiment")

def preprocess_hand_landmarks(df):
    wrist_x, wrist_y = df["x1"], df["y1"]
    mid_x, mid_y = df["x13"], df["y13"]
    df_processed = df.copy()

    for i in range(1, 22):
        df_processed[f"x{i}"] = (df[f"x{i}"] - wrist_x) / (abs(mid_x) + abs(mid_y))
        df_processed[f"y{i}"] = (df[f"y{i}"] - wrist_y) / (abs(mid_x) + abs(mid_y))

    return df_processed

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1": f1_score(y_test, y_pred, average='weighted')
    }

def train_and_log_models():
    df = pd.read_csv("hand_landmarks_data.csv")
    df = preprocess_hand_landmarks(df)

    X = df.drop(columns=['label'])
    y = df['label']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    joblib.dump(label_encoder, "label_encoder.pkl")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "scaler.pkl")

    model = XGBClassifier(
        n_estimators=154, learning_rate=0.2989, max_depth=4,
        min_child_weight=4, gamma=0.0236, subsample=0.8994,
        colsample_bytree=0.8870, random_state=42)


    with mlflow.start_run(run_name="XGBoost"):        
        model.fit(X_train_scaled, y_train)
        metrics = evaluate_model(model, X_test_scaled, y_test)
        mlflow.sklearn.log_model(model, "model")

        for key, val in metrics.items():
            mlflow.log_metric(key, val)
        print(f"XGBoost - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")


   
    with open("XGBoost_Best_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"Saved XGBoost model with F1 score: {metrics['f1']:.4f}")


if __name__ == "__main__":
    train_and_log_models()