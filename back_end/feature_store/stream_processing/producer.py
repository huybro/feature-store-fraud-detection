from kafka import KafkaProducer
import json
import time
import random
from datetime import datetime
import numpy as np
import uuid

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

CC_NUM_POOL = [
    f"{random.choice(['5018', '5020', '5038', '6304'])}{''.join(str(random.randint(0, 9)) for _ in range(10))}"
    for _ in range(50)
]

def get_random_cc_num():
    # Bias: some repeat, some less frequent
    return random.choices(CC_NUM_POOL, k=1)[0]

def generate_fake_transaction():
    raw_data = {
        "txn_id": f"txn_{uuid.uuid4().int >> 96}",  # shorter UUID-derived int
        "cc_num": get_random_cc_num(),
        "amount": round(np.random.exponential(scale=70), 2),
        "lat": round(np.random.normal(loc=38.5, scale=5.1), 6),
        "long": round(np.random.normal(loc=-90.2, scale=13.7), 6),
        "merch_lat": round(np.random.normal(loc=38.5, scale=5.1), 6),
        "merch_long": round(np.random.normal(loc=-90.2, scale=13.7), 6),
        "timestamp": datetime.now().isoformat()
    }
    return {k: str(v) for k, v in raw_data.items()}

if __name__ == "__main__":
    print("🚀 Sending fake transactions to Kafka topic 'transactions'...")
    while True:
        txn = generate_fake_transaction()
        print(f"Generated: {txn}")

        try:
            producer.send('transactions', txn).get(timeout=10)
            print("✔ Sent to Kafka")
        except Exception as e:
            print(f"❌ Failed to send: {e}")

        time.sleep(1)