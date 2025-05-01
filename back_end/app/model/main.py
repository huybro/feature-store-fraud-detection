import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

import glob

csv_files = glob.glob("./data/part-*.csv")  # Adjust to your actual directory
df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
print(f"Combined {len(csv_files)} files into one DataFrame with {len(df)} rows.")

# Drop high-cardinality or irrelevant columns
df = df.drop(columns=["cc_num", "trans_date_trans_time"])

# Label encode categorical features
categorical = ["category", "gender", "day_of_week"]
for col in categorical:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Handle imbalance: downsample non-fraud (or use class weights later)
fraud_df = df[df["is_fraud"] == 1]
non_fraud_df = df[df["is_fraud"] == 0].sample(n=len(fraud_df)*5, random_state=42)
df = pd.concat([fraud_df, non_fraud_df]).sample(frac=1, random_state=42)

# Separate features/labels
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Dataset class
class FraudDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(FraudDataset(X_train, y_train), batch_size=1024, shuffle=True)
test_loader = DataLoader(FraudDataset(X_test, y_test), batch_size=1024)

# Model
class FraudNet(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze()

model = FraudNet(X_train.shape[1])
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training
for epoch in range(10):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss {total_loss:.4f}")

# Evaluation
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        pred = model(X_batch)
        y_pred.extend(pred.numpy())
        y_true.extend(y_batch.numpy())

# Binary predictions
y_pred_label = [1 if p > 0.5 else 0 for p in y_pred]

from sklearn.metrics import accuracy_score, f1_score
print("Accuracy:", accuracy_score(y_true, y_pred_label))
print("F1 Score:", f1_score(y_true, y_pred_label))
