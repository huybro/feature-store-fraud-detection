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

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up the database:
```bash
# Using Docker Compose
docker-compose up -d
```

4. Run the API:
```bash
uvicorn api.main:app --reload
```

## Features

- Batch feature computation using PySpark
- Online feature serving via FastAPI
- PostgreSQL for offline feature storage
- Redis for online feature storage
- Authentication and authorization
- Comprehensive test suite

## Documentation

Detailed documentation can be found in the `docs/` directory, including:
- Architecture overview
- API documentation
- Development guidelines
- Milestone planning 