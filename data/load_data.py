import pandas as pd
import numpy as np


# Load the dataset
splits = {'train': 'credit_card_transaction_train.csv', 'test': 'credit_card_transaction_test.csv'}
df = pd.read_csv("hf://datasets/pointe77/credit-card-transaction/" + splits["train"])

# Preprocessing steps
df = df.drop(columns=['Unnamed: 0'])
df['merch_zipcode'] = df['merch_zipcode'].fillna(-1)

# Save the preprocessed DataFrame to a CSV file
output_file = 'data/credit_card_transactions.csv'
df.to_csv(output_file, index=False)
print(f"Preprocessed data saved to {output_file}")

