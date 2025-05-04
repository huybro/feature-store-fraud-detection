import requests
import pandas as pd
import numpy as np
import sys
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.training import CONFIG, train_model
import torch
def preprocess_data(df):
    print("⚙️ Preprocessing data...")

    # Drop non-feature columns
    df = df.drop(columns=["cc_num", "feature_timestamp"], errors="ignore")

    # Handle categorical encoding
    categorical_cols = ["category", "gender", "day_of_week"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Fill any missing values
    df = df.fillna(0)

    # Balance the data (5:1 ratio)
    if "is_fraud" in df.columns:
        fraud_df = df[df["is_fraud"] == 1]
        non_fraud_df = df[df["is_fraud"] == 0].sample(n=len(fraud_df) * 5, random_state=42)
        df = pd.concat([fraud_df, non_fraud_df]).sample(frac=1, random_state=42)
        print(f"✅ Balanced dataset: {len(fraud_df)} fraud + {len(non_fraud_df)} non-fraud")

    # Split X and y
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def run_training_experiment(api_url):
    print(f"🔍 Fetching training data from: {api_url}")
    fetch_start = time.time()

    response = requests.get(api_url)
    fetch_end = time.time()
    fetch_duration = fetch_end - fetch_start
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.status_code}")

    data = response.json()
    df = pd.DataFrame(data)

    print(f"✅ Fetched {len(df)} records in {fetch_duration:.2f} seconds.")
    train_times = []
    accs, precisions, recalls, f1s = [], [], [], []

    for run in range(3):
        print(f"\n🚀 Training Run {run+1}/3")

        # Sample 75% of the dataset
        sampled_df = df.sample(frac=0.75, random_state=42 + run)
        print(f"🔄 Sampled {len(sampled_df)} rows for run {run+1}")

        # Preprocess data
        X, y = preprocess_data(sampled_df)

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42 + run)

        # Train model
        start = time.time()
        model = train_model(X_train, y_train, X_test, y_test, CONFIG)
        duration = time.time() - start
        train_times.append(duration)

        # Evaluate
        y_pred = (model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy() > 0.5).astype(int)
        y_true = y_test.to_numpy()

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        print(f"✅ Run {run+1} — Time: {duration:.2f}s | Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # Average results
    print("\n📊 === Average over 3 runs ===")
    print(f"🕒 Avg Training Time: {np.mean(train_times):.2f} sec")
    print(f"🎯 Avg Accuracy: {np.mean(accs):.4f}")
    print(f"📌 Avg Precision: {np.mean(precisions):.4f}")
    print(f"📌 Avg Recall: {np.mean(recalls):.4f}")
    print(f"📌 Avg F1 Score: {np.mean(f1s):.4f}")

if __name__ == "__main__":
    API_URL = "http://localhost:8000/api/v1/features/bulk"  
    run_training_experiment(API_URL)