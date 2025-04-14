from kafka import KafkaProducer
import json
import time
import random
from datetime import datetime

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def generate_fake_transaction():
    raw_data = {
        "id": 0,
        "transaction_id": random.randint(100000, 999999),
        "amount": round(random.uniform(10, 500), 2),
        "merch_long": random.randint(-130,130),
        "merch_lat": random.randint(-130,130),
        "long": random.randint(-130,130),
        "lat": random.randint(-130,130),
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