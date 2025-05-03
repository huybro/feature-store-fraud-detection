import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.training import CONFIG, train_model, evaluate_model, FraudDataset, DataLoader, DeepFraudNet
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder

def process_credit_card_data(input_csv_path, output_csv_path):
    # Load data
    df = pd.read_csv(input_csv_path, parse_dates=["dob", "trans_date_trans_time"])

    # Ensure proper datetime parsing
    df["dob"] = pd.to_datetime(df["dob"])
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["unix_time"] = df["trans_date_trans_time"].astype(np.int64) // 10**9

    # Feature: hour of day, day of week
    df["hour_of_day"] = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek + 1  # Spark's dayofweek starts from 1

    # Feature: age at transaction
    df["age_at_txn"] = (df["trans_date_trans_time"] - df["dob"]).dt.total_seconds() / (60 * 60 * 24 * 365.25)

    # Feature: distance to merchant (Haversine formula in miles)
    def haversine(row):
        lat1, lon1, lat2, lon2 = map(np.radians, [row["lat"], row["long"], row["merch_lat"], row["merch_long"]])
        dlat = lat1 - lat2
        dlon = lon1 - lon2
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        return 2 * 3963 * np.arcsin(np.sqrt(a))

    df["distance_to_merchant"] = df.apply(haversine, axis=1)

    # Sort for time window calculations
    df.sort_values(by=["cc_num", "unix_time"], inplace=True)

    # Feature: txn_count_last_10_min
    df["txn_count_last_10_min"] = df.groupby("cc_num")["unix_time"].transform(
        lambda x: x.rolling(window=len(x), min_periods=1)
                    .apply(lambda s: ((x.iloc[-1] - s) <= 600).sum(), raw=False))

    # Feature: avg_amt_last_1_hour
    def rolling_avg_amt(group):
        result = []
        for i in range(len(group)):
            current_time = group.iloc[i]["unix_time"]
            window_df = group[(current_time - group["unix_time"]) <= 3600]
            result.append(window_df["amt"].mean())
        return pd.Series(result, index=group.index)

    df["avg_amt_last_1_hour"] = df.groupby("cc_num", group_keys=False).apply(rolling_avg_amt)

    # Select relevant columns
    output_cols = ["cc_num", "amt", "hour_of_day", "day_of_week", "age_at_txn", "distance_to_merchant",
                   "txn_count_last_10_min", "avg_amt_last_1_hour", "category", "gender", "city_pop",
                   "trans_date_trans_time", "is_fraud"]

    df = df[output_cols]

    # Save output
    df.to_csv(output_csv_path, index=False)

def preprocess_data(path_pattern):
    df = pd.read_csv(path_pattern)
    df = df.drop(columns=["cc_num", "trans_date_trans_time"])
    categorical = ["category", "gender", "day_of_week"]
    for col in categorical:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    fraud_df = df[df["is_fraud"] == 1]
    non_fraud_df = df[df["is_fraud"] == 0].sample(n=len(fraud_df) * 5, random_state=42)
    df = pd.concat([fraud_df, non_fraud_df]).sample(frac=1, random_state=42)

    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y 

def run_training_experiment(data_path):
    print(f"🔍 Loading full processed dataset from: {data_path}")

    full_df = pd.read_csv(data_path)

    train_times = []
    accuracies, precisions, recalls, f1s = [], [], [], []

    for run in range(3):
        print(f"\n🚀 Training Run {run+1}/3")

        # Step 1: Sample 75% each time
        print("🔄 Sampling 75% of the dataset for this run...")
        sampled_df = full_df.sample(frac=0.75, random_state=42 + run)

        # Step 2: Save to temp CSV for preprocess_data
        temp_path = data_path.replace(".csv", f"_train_75_run{run+1}.csv")
        sampled_df.to_csv(temp_path, index=False)
        print(f"✅ Sampled data saved to: {temp_path}")

        # Step 3: Preprocess sampled data
        X, y = preprocess_data(temp_path)

        # Step 4: Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42 + run)

        start = time.time()
        model = train_model(X_train, y_train, X_test, y_test, CONFIG)
        duration = time.time() - start
        train_times.append(duration)

        print(f"✅ Training time for run {run+1}: {duration:.2f} sec")

        # Evaluate on test set
        with torch.no_grad():
            y_probs = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
        y_preds = [1 if p > 0.5 else 0 for p in y_probs]
        y_true = np.array(y_test)

        acc = accuracy_score(y_true, y_preds)
        precision = precision_score(y_true, y_preds, zero_division=0)
        recall = recall_score(y_true, y_preds, zero_division=0)
        f1 = f1_score(y_true, y_preds)

        accuracies.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    avg_train_time = sum(train_times) / 3
    print(f"\n📊 === Averaged Training Results ===")
    print(f"🕒 Avg Training Time : {avg_train_time:.2f} sec")
    print(f"🎯 Avg Accuracy      : {sum(accuracies)/3:.4f}")
    print(f"📌 Avg Precision     : {sum(precisions)/3:.4f}")
    print(f"📌 Avg Recall        : {sum(recalls)/3:.4f}")
    print(f"📌 Avg F1 Score      : {sum(f1s)/3:.4f}")

    return {
        "average_training_time": avg_train_time,
        "average_accuracy": sum(accuracies) / 3,
        "average_precision": sum(precisions) / 3,
        "average_recall": sum(recalls) / 3,
        "average_f1": sum(f1s) / 3
    }


def run_testing_experiment(input_csv_path):
    print(f"🔍 Starting testing experiment using raw file: {input_csv_path}")

    total_times = []

    for run in range(3):
        print(f"\n🧪 === Testing Run {run+1}/3 ===")

        df = pd.read_csv(input_csv_path)
        sampled_df = df.sample(n=min(100_000, len(df)), random_state=42 + run)

        base_dir = os.path.dirname(input_csv_path)
        sample_input_csv = os.path.join(base_dir, f"sampled_input_run{run+1}.csv")
        sample_output_csv = os.path.join(base_dir, f"output_features_python_sampled_run{run+1}.csv")
        sampled_df.to_csv(sample_input_csv, index=False)

        print("⚙️ Processing sampled dataset...")
        run_start_time = time.time()
        process_credit_card_data(sample_input_csv, sample_output_csv)
        X, y = preprocess_data(sample_output_csv)

        sample_dataset = FraudDataset(X, pd.Series([0] * len(X)))
        sample_loader = DataLoader(sample_dataset, batch_size=CONFIG["batch_size"])

        model = DeepFraudNet(X.shape[1], CONFIG["dropout_rate"])
        model.load_state_dict(torch.load("fraud_model.pth"))
        model.eval()

        with torch.no_grad():
            for X_batch, _ in sample_loader:
                _ = model(X_batch)

        run_duration = time.time() - run_start_time
        total_times.append(run_duration)
        print(f"🕒 Total Time (processing + inference): {run_duration:.2f} sec")

    avg_total_time = sum(total_times) / 3
    print(f"\n📉 === Avg Total Inference Time over 3 Runs: {avg_total_time:.2f} sec")
    return {"average_total_time": avg_total_time}


# Paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Full dataset paths
input_csv = os.path.join(base_dir, 'data', 'credit_card_transactions.csv')
output_csv = os.path.join(base_dir, 'data', 'output_features_python.csv')

# print("\n⚙️ Processing full dataset...")
# start_full = time.time()
# process_credit_card_data(input_csv, output_csv)
# end_full = time.time()
# print(f"✅ Full dataset processing completed in {end_full - start_full:.2f} seconds") #451.29 seconds
# print(f"➡️ Output saved to: {output_csv}")

# run_training_experiment(output_csv) 
# 📊 === Averaged Training Results ===
# 🕒 Avg Training Time : 76.34 sec
# 🎯 Avg Accuracy      : 0.9740
# 📌 Avg Precision     : 0.9600
# 📌 Avg Recall        : 0.8822
# 📌 Avg F1 Score      : 0.9194
# run_testing_experiment(input_csv) #25.61 sec