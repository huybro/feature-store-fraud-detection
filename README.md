# 🛡️ Feature Store for Fraud Detection

This project implements a **feature store** designed for fraud detection using **credit card transaction data**. The system enables efficient **feature computation, storage, and serving** for both **batch** and **real-time fraud detection models**.

## 🚀 Key Features

- ✅ **Real-time feature computation:**  
  Processes streaming transaction data using **Kafka + Flink + Redis** to compute dynamic features (e.g., transaction counts, average amounts).

- ✅ **Batch feature computation:**  
  Historical data is processed and stored in **PostgreSQL** for offline analysis and model training.

- ✅ **Feature serving API:**  
  A **FastAPI backend** provides endpoints to:
  - Retrieve features for a credit card number / dates.
  - Sync features from Redis to PostgreSQL.
  - Bulk export features for training/inference.

- ✅ **Fraud Detection Model:**  
  - Deep learning model (`DeepFraudNet`) implemented using **PyTorch**.
  - Supports **training experiments** and **benchmarking inference performance**.
"""
## 🏗️ Architecture

```mermaid
graph TD
    A[Kafka Transaction Stream] --> B[Real-time Feature Computation (PyFlink)]
    G[Batch Feature Computation (Spark)] --> D[Offline Store (PostgreSQL)]
    B --> C[Online Store (Redis)]
    B --> D
    E[FastAPI Server]
    C --> E
    D --> E
    E --> F[Model Training / Inference Flow]
    
## 📂 Project Structure

```
feature_store_fraud_detection/
├── back_end/
│   ├── app/                   # FastAPI application (routes, schemas, services)
│   ├── model/                 # Deep learning model (DeepFraudNet) & training code
│   ├── flows/                 # Scripts for training, inference, and feature syncing
│   ├── feature_store/         # Redis & PostgreSQL integration logic
│   └── prisma/                # Prisma client & database schema
├── docker/                    # Docker setup & docker-compose configs
├── data/                      # Raw data & processed feature files
└── README.md                  # Project documentation

```

## 🛠️ Setup & Installation

1️⃣ **Clone the repo:**

```bash
git clone https://github.com/yourusername/feature_store_fraud_detection.git
cd feature_store_fraud_detection
```

2️⃣ **Set up the environment:**

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

**🔄Using Poetry:**

```bash
pip install poetry
poetry lock
poetry install --no-root
```

_This sets up the environment and installs all dependencies. You can then run the FastAPI server as usual:_

```bash
uvicorn back_end.main:app --reload
```

---

3️⃣ **Run services (Kafka, Redis, PostgreSQL):**

We recommend using Docker Compose:

```bash
cd back_end/docker
docker-compose up -d
```

4️⃣ **Start the FastAPI server:**

```bash
uvicorn back_end.main:app --reload
```

## 📈 Feature API Endpoints

| Method | Endpoint                                                       | Description                                                                |
|--------|----------------------------------------------------------------|----------------------------------------------------------------------------|
| GET    | `/api/v1/features/bulk`                                        | Retrieve all feature rows from the PostgreSQL feature store.               |
| GET    | `/api/v1/features/by-ccnum/{cc_num}`                           | Retrieve features for a specific credit card number.                       |
| GET    | `/api/v1/features/by-date?start=YYYY-MM-DD&end=YYYY-MM-DD`     | Retrieve features within a specific date range.                            |
| POST   | `/api/v1/features/batch`                                       | Insert a batch of feature rows into the PostgreSQL store.                  |
| POST   | `/api/v1/features/sync`                                        | Sync static feature columns from PostgreSQL into Redis stats hashes.       |
| DELETE | `/api/v1/features/clear`                                       | Delete all features from the PostgreSQL feature store.                     |
| GET    | `/api/v1/redis/transactions/bulk?limit=100000`                | Bulk export transaction stats (with latest `amt`) from Redis.              |
| GET    | `/api/v1/redis/transactions/{cc_num}`                          | Retrieve latest transaction stats and recent transactions for a card.      |

## 🧪 Running Experiments

- **Training experiment:**

```bash
python back_end/flows/featurestore_flow.py
```

- **Inference experiment:**

```bash
python back_end/flows/inference_experiment.py
```

- **Baseline experiment:**

```bash
python back_end/flows/baseline_flow.py
```

## 💡 Example Features

- `txn_count_last_10_min`
- `avg_amt_last_1_hour`
- `distance_to_merchant`
- `hour_of_day`, `day_of_week`
- `category`, `gender`
- `city_pop`

## 🧪 Experiment Results

### Experiment 1: Training Runtime & Performance

**Objective:** Measure average training time and performance metrics using the feature store vs. baseline.

| Setup           | Avg Training Time (sec) | Accuracy | Precision | Recall | F1 Score |
|-----------------|-------------------------|----------|-----------|--------|----------|
| Baseline        | 527.71                  | 0.9645   | 0.9042    | 0.8822 | 0.8927   |
| Feature Store   | 118.38                  | 0.9714   | 0.9666    | 0.8564 | 0.9080   |

---

### Experiment 2: Inference Runtime Benchmark

**Objective:** Measure average inference latency retrieving features via the feature store.

| Setup           | Data Retrieval Time (sec) | Inference Time (sec) |
|-----------------|---------------------------|----------------------|
| Feature Store   | 0.97                      | 0.02                 |
|   Baseline      |                         25.92 

## 🚩 Future Improvements

- [ ] Add CI/CD pipeline
- [ ] Monitoring feature freshness & latency
- [ ] Add more complex fraud features

---
