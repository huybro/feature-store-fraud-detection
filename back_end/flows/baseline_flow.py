import pandas as pd
import numpy as np
import os
from datetime import datetime
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

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

def measure_average_testing_time(model, X_test, runs=3, sample_size=100_000):
    X_sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=42)
    times = []
    for _ in range(runs):
        start = time.time()
        _ = model.predict(X_sample)
        end = time.time()
        times.append(end - start)
    avg_test_time = sum(times) / runs
    print(f"\nAverage Testing Time over {runs} runs on {len(X_sample)} rows: {avg_test_time:.4f} sec")
    return avg_test_time

def run_experiment(data_path):
    df = pd.read_csv(data_path)
    X = df.drop(columns=['is_fraud', 'trans_date_trans_time'])
    y = df['is_fraud']

    times = []
    accuracies, precisions, recalls, f1s = [], [], [], []
    test_times = []

    for run in range(3):
        print(f"\n=== Run {run+1}/3 ===")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, shuffle=True, stratify=y
        )

        start = time.time()

        # === Define and train your model here ===
        # model = YourModel()
        # model.fit(X_train, y_train)
        # y_pred = model.predict(X_test)
        y_pred = [0] * len(y_test)  # Placeholder
        model = None  # Placeholder
        # === End model definition ===

        end = time.time()
        times.append(end - start)

        # Metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))

        # If you used a real model, measure test time
        if model is not None:
            test_time = measure_average_testing_time(model, X_test)
            test_times.append(test_time)

    avg_metrics = {
        "average_training_time": sum(times) / 3,
        "average_accuracy": sum(accuracies) / 3,
        "average_precision": sum(precisions) / 3,
        "average_recall": sum(recalls) / 3,
        "average_f1": sum(f1s) / 3,
        "average_testing_time": sum(test_times) / len(test_times) if test_times else None
    }

    print("\n=== Averaged Results over 3 Runs ===")
    for k, v in avg_metrics.items():
        if v is None:
            print(f"{k.replace('_', ' ').title()}: N/A (no model)")
        elif 'time' in k:
            print(f"{k.replace('_', ' ').title()}: {v:.4f} sec")
        else:
            print(f"{k.replace('_', ' ').title()}: {v:.4f}")

    return avg_metrics

# Define file paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
input_csv = os.path.join(base_dir, 'data', 'credit_card_transactions.csv')
output_csv = os.path.join(base_dir, 'data', 'output_features_python.csv')

# Measure processing time
start = time.time()
process_credit_card_data(input_csv, output_csv)
end = time.time()

processing_duration = end - start
print(f"\nData processing completed in {processing_duration:.2f} seconds")

# Run the experiment
metrics = run_experiment(output_csv)