from pyflink.datastream.functions import MapFunction
import redis
import datetime

class RedisWriter(MapFunction):
    def __init__(self, host='localhost', port=6379, db=0, ttl=3600):
        self.host = host
        self.port = port
        self.db = db
        self.ttl = ttl
        self.redis_client = None

    def open(self, runtime_context):
        self.redis_client = redis.Redis(host=self.host, port=self.port, db=self.db)
        print(f"Connected to Redis at {self.host}:{self.port}")

    def map(self, value):
        try:
            txn_id = value.get('txn_id')
            key_prefix = f"txn:{value['cc_num']}"
            transaction_key = f"{key_prefix}:data:{txn_id}"
            stats_key = f"{key_prefix}:stats"

            self.redis_client.hset(transaction_key, mapping=value)
            self.redis_client.expire(transaction_key, self.ttl)

            stats = {
                'txn_count_last_10_min': value.get('txn_count_last_10_min', '0'),
                'avg_amt_last_1_hour': value.get('avg_amt_last_1_hour', '0'),
                'distance_to_merchant': value.get('distance_to_merchant', '0'),
                'last_update': datetime.datetime.now().isoformat()
            }
            self.redis_client.hset(stats_key, mapping=stats)

            if 'timestamp' in value:
                timestamp = int(datetime.datetime.fromisoformat(value['timestamp']).timestamp())
                self.redis_client.zadd(f"{key_prefix}:timeline", {txn_id: timestamp})
                self.redis_client.expire(f"{key_prefix}:timeline", self.ttl)

            print(f"Stored transaction {txn_id} in Redis for card {value['cc_num']}")
        except Exception as e:
            print(f"Redis write error: {e}")

        return value