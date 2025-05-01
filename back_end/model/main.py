import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- Configuration ---
CONFIG = {
    "batch_size": 512,
    "epochs": 300,
    "learning_rate": 1e-3,
    "patience": 3,
    "lr_reduce_factor": 0.5,
    "weight_decay": 1e-4,
    "dropout_rate": 0.3,
}

# --- Load Data ---
print(f"Working directory: {os.getcwd()}")
files = glob.glob("../data/output_features/*.csv")
print(f"Found {len(files)} files.")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print(f"Total rows loaded: {len(df)}")

# --- Preprocessing ---
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

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Dataset ---
class FraudDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(FraudDataset(X_train, y_train), batch_size=CONFIG["batch_size"], shuffle=True)
test_loader = DataLoader(FraudDataset(X_test, y_test), batch_size=CONFIG["batch_size"])

# --- Model ---
class DeepFraudNet(nn.Module):
    def __init__(self, in_features, dropout_rate):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).squeeze()

model = DeepFraudNet(X_train.shape[1], CONFIG["dropout_rate"])
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CONFIG["lr_reduce_factor"], patience=CONFIG["patience"])

# --- Training ---
train_losses = []
val_aucs = []
val_f1s = []
best_loss = float('inf')
early_stop_counter = 0

for epoch in range(CONFIG["epochs"]):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader)
    scheduler.step(avg_loss)
    train_losses.append(avg_loss)

    # Validation AUC/F1 (on test set)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            y_pred.extend(outputs.numpy())
            y_true.extend(y_batch.numpy())
    auc_score = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, [1 if p > 0.5 else 0 for p in y_pred])
    val_aucs.append(auc_score)
    val_f1s.append(f1)

    print(f"Epoch {epoch+1}/{CONFIG['epochs']}, Loss: {avg_loss:.4f}, AUC: {auc_score:.4f}, F1: {f1:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), "fraud_model.pth")
        print("✅ Model improved and saved.")
    else:
        early_stop_counter += 1
        if early_stop_counter >= CONFIG["patience"] + 30:
            print("🛑 Early stopping triggered.")
            break

# --- Final Evaluation ---
model.load_state_dict(torch.load("fraud_model.pth"))
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        y_pred.extend(outputs.numpy())
        y_true.extend(y_batch.numpy())

y_pred_label = [1 if p > 0.5 else 0 for p in y_pred]

print("✅ Final Evaluation Metrics")
print("Accuracy:", accuracy_score(y_true, y_pred_label))
print("F1 Score:", f1_score(y_true, y_pred_label))
print("\nClassification Report:\n", classification_report(y_true, y_pred_label, digits=4))

# --- Plots ---
# Loss plot
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss per Epoch")
plt.grid(True)
plt.legend()
plt.show()

# AUC plot
plt.figure()
plt.plot(val_aucs, label="Validation AUC")
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.title("Validation AUC per Epoch")
plt.grid(True)
plt.legend()
plt.show()

# F1 plot
plt.figure()
plt.plot(val_f1s, label="Validation F1 Score")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("Validation F1 Score per Epoch")
plt.grid(True)
plt.legend()
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_true, y_pred)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_pred):.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.grid(True)
plt.legend()
plt.show()

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_true, y_pred)
plt.figure()
plt.plot(recall, precision, label=f"PR AUC = {auc(recall, precision):.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid(True)
plt.legend()
plt.show()
