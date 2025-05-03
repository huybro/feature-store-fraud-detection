import requests
import pandas as pd
import numpy as np
import time
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.training import CONFIG, train_model, FraudDataset, DeepFraudNet
import torch

def preprocess_data(df):
    df = df.drop(columns=["cc_num", "feature_timestamp"], errors="ignore")
    categorical = ["category", "gender", "day_of_week"]
    for col in categorical:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    df = df.fillna(0)
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"] if "is_fraud" in df.columns else pd.Series([0] * len(df))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def run_training_experiment(data_path):
    print(f"🔍 Loading full processed dataset from: {data_path}")

    # Load once outside loop
    full_df = pd.read_csv(data_path)

    train_times = []
    accs, precisions, recalls, f1s = [], [], [], []

    for run in range(3):
        print(f"\n🚀 Training Run {run+1}/3")

        # Step 1: Sample 75% each time
        print("🔄 Sampling 75% of the dataset for this run...")
        sampled_df = full_df.sample(frac=0.75, random_state=42 + run)

        # Step 2: Save to temp CSV for preprocessing
        temp_path = data_path.replace(".csv", f"_train_75_run{run+1}.csv")
        sampled_df.to_csv(temp_path, index=False)
        print(f"✅ Sampled data saved to: {temp_path}")

        # Step 3: Preprocess
        X, y = preprocess_data(sampled_df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42 + run)

        # Step 4: Train
        start = time.time()
        model = train_model(X_train, y_train, X_test, y_test, CONFIG)
        duration = time.time() - start
        train_times.append(duration)

        # Evaluate without plotting
        y_true, y_pred = [], []
        model.eval()
        with torch.no_grad():
            outputs = model(torch.tensor(X_test, dtype=torch.float32))
            y_probs = outputs.numpy()
            y_pred = [1 if p > 0.5 else 0 for p in y_probs]
            y_true = y_test.to_numpy()

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        print(f"✅ Training time: {duration:.2f} sec | Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # Final average
    print("\n📊 === Average over 3 runs ===")
    print(f"🕒 Avg Training Time: {np.mean(train_times):.2f} sec")
    print(f"🎯 Avg Accuracy: {np.mean(accs):.4f}")
    print(f"📌 Avg Precision: {np.mean(precisions):.4f}")
    print(f"📌 Avg Recall: {np.mean(recalls):.4f}")
    print(f"📌 Avg F1 Score: {np.mean(f1s):.4f}")

    return model  # Return the last trained model for inference!

def run_inference_experiment(df, model):
    X, y = preprocess_data(df)
    print(f"🔢 Inference dataset: {X.shape[0]} samples, {X.shape[1]} features")

    model.eval()
    print("🚀 Running inference on 100,000 rows...")
    start = time.time()
    y_probs = []
    batch_size = CONFIG["batch_size"]
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32)
            outputs = model(batch).numpy()
            y_probs.extend(outputs)

    duration = time.time() - start
    print(f"✅ Inference complete in {duration:.2f} seconds")

    y_preds = [1 if p > 0.5 else 0 for p in y_probs]

    if "is_fraud" in df.columns:
        y_true = y.to_numpy()
        acc = accuracy_score(y_true, y_preds)
        precision = precision_score(y_true, y_preds, zero_division=0)
        recall = recall_score(y_true, y_preds, zero_division=0)
        f1 = f1_score(y_true, y_preds, zero_division=0)

        print(f"📊 Inference Metrics → Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    else:
        print("ℹ️ No ground truth labels available for metrics.")

if __name__ == "__main__":
    # Train Phase
    """print("🔍 Crawling training data via API...")
    crawl_start_time = time.time()
    print(f"⏱ Started crawling at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(crawl_start_time))}")

    # Training crawl
    records = []
    for cc_num in range(0, 983):
        response = requests.get(f"http://localhost:8000/api/v1/features/by-ccnum/{cc_num}")
        if response.status_code == 200:
            records.extend(response.json())

    crawl_end_time = time.time()
    duration = crawl_end_time - crawl_start_time

    df_crawled = pd.DataFrame(records)
    print(f"✅ Crawled {len(df_crawled)} records.")
    print(f"🏁 Finished crawling at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(crawl_end_time))}")
    print(f"⏲️ Total crawling time: {duration:.2f} seconds\n")

    # Save crawled data to CSV
    output_path = "./crawled_features.csv"
    df_crawled.to_csv(output_path, index=False)
    print(f"💾 Saved to {output_path}")

    # Run training benchmark
    trained_model = run_training_experiment(output_path)"""

    # Inference Phase
    print("\n🔍 Crawling 100,000 rows for inference via /redis/transactions/bulk...")
    inference_crawl_start = time.time()
    print(f"⏱ Started inference crawl at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(inference_crawl_start))}")

    BULK_URL = "http://localhost:8000/api/v1/redis/transactions/bulk?limit=100000"
    response = requests.get(BULK_URL)

    if response.status_code == 200:
        data = response.json()
        df_bulk = pd.DataFrame(data)

        inference_crawl_end = time.time()
        crawl_duration = inference_crawl_end - inference_crawl_start

        print(f"✅ Crawled {len(df_bulk)} records for inference.")
        print(f"🏁 Finished inference crawl at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(inference_crawl_end))}")
        print(f"⏲️ Total inference crawling time: {crawl_duration:.2f} seconds\n")

        # Save to CSV (optional)
        bulk_output_path = "./crawled_bulk_features.csv"
        df_bulk.to_csv(bulk_output_path, index=False)
        print(f"💾 Saved bulk data to {bulk_output_path}")

        # Run inference benchmark
        run_inference_experiment(df_bulk, trained_model)
    else:
        print(f"❌ Failed to fetch inference data: {response.status_code}")