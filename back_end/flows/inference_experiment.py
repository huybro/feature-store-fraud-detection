import requests
import pandas as pd
import numpy as np
import sys
import os
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.training import CONFIG, DeepFraudNet  # Load your model class & CONFIG

def preprocess_inference_data(df):
    print("⚙️ Preprocessing inference data...")

    # Drop non-feature columns if they exist
    drop_cols = ["cc_num", "feature_timestamp", "last_update"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

    # Handle categorical encoding
    categorical_cols = ["category", "gender", "day_of_week"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    df = df.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    return X_scaled

def run_inference(api_url, model_path):
    print(f"🔍 Fetching inference data from: {api_url}")

    # Record fetch start time
    fetch_start = time.time()
    response = requests.get(api_url)
    fetch_duration = time.time() - fetch_start

    if response.status_code != 200:
        raise Exception(f"Failed to fetch inference data: {response.status_code}")

    data = response.json()
    df = pd.DataFrame(data)
    print(f"✅ Retrieved {len(df)} records for inference in {fetch_duration:.2f} seconds.")

    if len(df) == 0:
        print("⚠️ No data to infer. Exiting.")
        return

    # Preprocess
    X = preprocess_inference_data(df)
    print(f"🔢 Inference dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Load trained model
    input_dim = X.shape[1]
    model = DeepFraudNet(input_dim, CONFIG.get("dropout_rate", 0.5))
    print(f"📦 Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Run inference 3 times and take the average
    inference_times = []

    for run in range(3):
        print(f"\n🚀 Inference Run {run + 1}/3...")
        infer_start = time.time()
        y_probs = []
        batch_size = CONFIG.get("batch_size", 512)

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32)
                outputs = model(batch).numpy()
                y_probs.extend(outputs)

        infer_duration = time.time() - infer_start
        inference_times.append(infer_duration)
        print(f"✅ Run {run + 1} completed in {infer_duration:.2f} seconds.")

    avg_infer_time = np.mean(inference_times)
    print(f"\n📊 === Inference Timing Summary ===")
    for i, t in enumerate(inference_times, 1):
        print(f" - Run {i}: {t:.2f} sec")
    print(f"🕒 Average Inference Time over 3 runs: {avg_infer_time:.2f} sec")

    # Post-process predictions (from last run)
    y_preds = [1 if p > 0.5 else 0 for p in y_probs]
    print(f"🔍 Completed predictions for {len(y_preds)} rows.")

    # Optionally: Save predictions to CSV
    df_result = df.copy()
    df_result["fraud_probability"] = y_probs
    df_result["is_fraud_prediction"] = y_preds
    output_csv = "./inference_results.csv"
    df_result.to_csv(output_csv, index=False)
    print(f"💾 Inference results saved to: {output_csv}")

    # Final timings
    print("\n⏱️ Timing summary:")
    print(f" - Data retrieval time: {fetch_duration:.2f} sec")
    print(f" - Average inference time: {avg_infer_time:.2f} sec")

if __name__ == "__main__":
    BULK_API_URL = "http://localhost:8000/api/v1/redis/transactions/bulk?limit=100000"
    MODEL_PATH = "./fraud_model.pth"  # 🔥 Set your trained model path here
    run_inference(BULK_API_URL, MODEL_PATH)