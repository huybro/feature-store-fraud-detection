import pandas as pd
import numpy as np


# Load the dataset
splits = {'train': 'credit_card_transaction_train.csv', 'test': 'credit_card_transaction_test.csv'}
df = pd.read_csv("hf://datasets/pointe77/credit-card-transaction/" + splits["train"])

# Preprocessing steps
df = df.drop(columns=['Unnamed: 0'])
df['merch_zipcode'] = df['merch_zipcode'].fillna(-1)

is_sorted = df['cc_num'].is_monotonic_increasing
print(f"Is cc_num sorted in ascending order? {is_sorted}")

# Print summary statistics (mean, std, min, max, etc.)
print("\n📊 Summary Statistics:\n")
print(df[['cc_num', 'amt', 'lat', 'long', 'merch_lat', 'merch_long']].describe(include='all'))

df['cc_num'] = pd.factorize(df['cc_num'])[0]
output_file = './back_end/data/credit_card_transactions.csv'
df.to_csv(output_file, index=False)
print(f"Preprocessed data saved to {output_file}")

