# Feature Store for Fraud Detection

This project implements a feature store for fraud detection using credit card transaction data. The feature store enables efficient feature computation, storage, and serving for both batch and real-time fraud detection models.

## Project Structure

```
feature-store-fraud-detection/
├── data/                            # Raw and processed datasets
├── notebooks/                       # Jupyter notebooks for exploration
├── scripts/                         # CLI jobs / one-off scripts
├── feature_store/                  # Core feature store logic
├── api/                            # FastAPI app for feature serving
├── tests/                          # Unit & integration tests
├── docker/                         # Docker-related files
└── docs/                           # Documentation
```