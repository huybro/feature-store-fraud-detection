import pandas as pd
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from tqdm import tqdm
import os
import glob
# === CONFIG ===
BATCH_SIZE = 250
MAX_WORKERS = 2
MAX_RETRIES = 3
ENDPOINT_URL = "http://localhost:8000/api/v1/features/batch"

# === Load your preprocessed DataFrame ===
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
output_path = os.path.join(base_dir, 'data', 'output_features')
part_files = glob.glob(os.path.join(output_path, "part-*.csv"))
if not part_files:
    raise FileNotFoundError("No output part file found in output_features directory.")
df_pd = pd.concat([pd.read_csv(f) for f in part_files], ignore_index=True)
df_pd = df_pd.fillna(0)
records = df_pd.to_dict(orient="records")

# === Chunking Helper ===
def chunk_data(data: List[dict], batch_size: int):
    for i in range(0, len(data), batch_size):
        yield i, data[i:i + batch_size]

# === Send a single chunk with retry logic ===
def send_chunk(chunk: List[dict], start_index: int):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            res = requests.post(ENDPOINT_URL, json=chunk, timeout=50)
            if res.status_code == 200:
                print(f"✅ Batch {start_index}-{start_index+len(chunk)} inserted")
                return True
            else:
                print(f"❌ Error {res.status_code} on batch {start_index}: {res.text}")
        except Exception as e:
            print(f"⚠️ Attempt {attempt} failed on batch {start_index}: {e}")
        time.sleep(1)  # Wait before retrying
    print(f"❌ Failed to insert batch {start_index} after {MAX_RETRIES} attempts")
    return False

# === Parallel Upload Function ===
def post_in_chunks_parallel(records: List[dict], batch_size: int, workers: int):
    chunks = list(chunk_data(records, batch_size))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(send_chunk, chunk, start)
            for start, chunk in chunks
        ]


# === Run the uploader ===
if __name__ == "__main__":
    print(f"📦 Starting upload of {len(records)} records in batches of {BATCH_SIZE}...")
    post_in_chunks_parallel(records, batch_size=BATCH_SIZE, workers=MAX_WORKERS)
    print("🎉 Upload complete.")