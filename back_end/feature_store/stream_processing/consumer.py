from pyflink.datastream import StreamExecutionEnvironment, TimeCharacteristic, KeyedStream
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common import Time, WatermarkStrategy, Types, Duration
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer
from pyflink.datastream.functions import ProcessWindowFunction, RuntimeContext, CoProcessFunction, SinkFunction, MapFunction
from pyflink.datastream.window import TumblingEventTimeWindows, SlidingEventTimeWindows
from pyflink.datastream.state import ValueStateDescriptor
from pyflink.common.watermark_strategy import TimestampAssigner
from pyflink.datastream.window import SlidingProcessingTimeWindows
from math import radians, sin, cos, sqrt, atan2
from typing import Iterable
import datetime
import redis
import json
import os

class TransactionTimestampAssigner(TimestampAssigner):
    def extract_timestamp(self, element, record_timestamp):
        timestamp = int(datetime.datetime.fromisoformat(element['timestamp']).timestamp() * 1000)
        return timestamp

class TxnCountLast10Min(ProcessWindowFunction):
    def process(self, key, context: ProcessWindowFunction.Context, elements: Iterable[dict]):
        elements_list = list(elements)
        count = len(elements_list)
        yield({
            'cc_num': key,
            'txn_count_last_10_min': str(count)
        })

class AvgAmtLast1Hour(ProcessWindowFunction):
    def process(self, key, context: ProcessWindowFunction.Context, elements: Iterable[dict]):
        elements_list = list(elements)
        amounts = [float(e['amount']) for e in elements_list]
        avg = sum(amounts) / len(amounts)
        yield({
            'cc_num': key,
            'avg_amt_last_1_hour': str(avg)
        })

class CombineTxnAndAvg(CoProcessFunction):
    def open(self, runtime_context: RuntimeContext):
        self.txn_count_state = runtime_context.get_state(ValueStateDescriptor("txn_count", Types.MAP(Types.STRING(), Types.STRING())))
        self.avg_amt_state = runtime_context.get_state(ValueStateDescriptor("avg_amt", Types.MAP(Types.STRING(), Types.STRING())))

    def process_element1(self, value, ctx: CoProcessFunction.Context):
        self.txn_count_state.update(value)
        avg_amt = self.avg_amt_state.value()
        if avg_amt:
            combined = {**value, **avg_amt}
            yield combined

    def process_element2(self, value, ctx: CoProcessFunction.Context):
        self.avg_amt_state.update(value)
        txn_count = self.txn_count_state.value()
        if txn_count:
            combined = {**txn_count, **value}
            yield combined

class FinalJoiner(CoProcessFunction):
    def open(self, runtime_context: RuntimeContext):
        self.distance_state = runtime_context.get_state(ValueStateDescriptor("distance", Types.MAP(Types.STRING(), Types.STRING())))
        self.stats_state = runtime_context.get_state(ValueStateDescriptor("stats", Types.MAP(Types.STRING(), Types.STRING())))

    def process_element1(self, distance, ctx: CoProcessFunction.Context):
        self.distance_state.update(distance)
        stats = self.stats_state.value()
        if stats:
            yield {**distance, **stats}

    def process_element2(self, stats, ctx: CoProcessFunction.Context):
        self.stats_state.update(stats)
        distance = self.distance_state.value()
        if distance:
            yield {**distance, **stats}

class RedisWriter(MapFunction):
    """Map function that writes data to Redis and passes it through."""
    
    def __init__(self, host='localhost', port=6379, db=0, ttl=3600):
        self.host = host
        self.port = port
        self.db = db
        self.ttl = ttl
        self.redis_client = None
    
    def open(self, runtime_context):
        """Initialize Redis client when the function is opened."""
        self.redis_client = redis.Redis(host=self.host, port=self.port, db=self.db)
        print(f"Connected to Redis at {self.host}:{self.port}")
    
    def map(self, value):
        """Write value to Redis and return it unchanged."""
        try:
            if not self.redis_client:
                self.redis_client = redis.Redis(host=self.host, port=self.port, db=self.db)
            
            # Get transaction ID or generate one if not present
            txn_id = value.get('transaction_id', f"txn_{datetime.datetime.now().timestamp()}")
            
            # Use cc_num as key prefix
            key_prefix = f"txn:{value['cc_num']}"
            
            # Store full transaction data as a hash
            transaction_key = f"{key_prefix}:data:{txn_id}"
            self.redis_client.hset(transaction_key, mapping=value)
            self.redis_client.expire(transaction_key, self.ttl)
            
            # Store stats in a separate key for quick access
            stats_key = f"{key_prefix}:stats"
            stats = {
                'txn_count_last_10_min': value.get('txn_count_last_10_min', '0'),
                'avg_amt_last_1_hour': value.get('avg_amt_last_1_hour', '0'),
                'distance_to_merchant': value.get('distance_to_merchant', '0'),
                'last_update': datetime.datetime.now().isoformat()
            }
            self.redis_client.hset(stats_key, mapping=stats)
            
            # Add to a sorted set for time-based queries (score is timestamp)
            if 'timestamp' in value:
                timestamp = int(datetime.datetime.fromisoformat(value['timestamp']).timestamp())
                self.redis_client.zadd(f"{key_prefix}:timeline", {txn_id: timestamp})
                self.redis_client.expire(f"{key_prefix}:timeline", self.ttl)
                
            print(f"Stored transaction {txn_id} in Redis for card {value['cc_num']}")
        except Exception as e:
            print(f"Redis write error: {e}")
        
        # Return the input value unchanged
        return value

def haversine(lat1, lon1, lat2, lon2):
    R = 3963
  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

env = StreamExecutionEnvironment.get_execution_environment()
env.set_stream_time_characteristic(TimeCharacteristic.EventTime)

# Path to JAR files
jars_dir = "./jars" 

# Add the Kafka connector JAR with proper file:// protocol
jar_path = f"file://{os.path.abspath(os.path.join(jars_dir, 'flink-connector-kafka-3.3.0-1.20.jar'))}" 
env.add_jars(jar_path)

kafka_clients_jar = f"file://{os.path.abspath(os.path.join(jars_dir, 'kafka-clients-7.9.0-ccs.jar'))}"
env.add_jars(kafka_clients_jar)

# Create a Kafka source
properties = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'pyflink-consumer'
}

stream = env.add_source(
    FlinkKafkaConsumer(
        topics='transactions',
        deserialization_schema=SimpleStringSchema(),
        properties=properties
    )
)

# Parse JSON
parsed = stream.map(lambda raw: json.loads(raw), output_type=Types.MAP(Types.STRING(), Types.STRING()))

# Assign timestamps and watermarks
parsed = parsed.assign_timestamps_and_watermarks(
    WatermarkStrategy
        .for_bounded_out_of_orderness(Duration.of_seconds(5))   
        .with_timestamp_assigner(TransactionTimestampAssigner())
)

# Calculate distance between buyer and merchant + float conversion for amount
distance_included = parsed.map(
    lambda x : {
        **x,
        'distance_to_merchant': str(haversine(x['lat'], x['long'], x['merch_lat'], x['merch_long']))
    },
    output_type=Types.MAP(Types.STRING(), Types.STRING())
)

# Count transactions in the last 10 minutes (sliding every 1 minute) (currently testing with a smaller window of 2 minutes)
txn_count = distance_included \
    .key_by(lambda t: t['cc_num']) \
    .window(SlidingProcessingTimeWindows.of(Time.minutes(2), Time.seconds(20))) \
    .process(TxnCountLast10Min(),output_type=Types.MAP(Types.STRING(), Types.STRING()))

# Average amount in the last 1 hour (sliding every 1 minute) (currently testing with a smaller window of 5 minutes)
avg_amount = distance_included \
    .key_by(lambda t: t['cc_num']) \
    .window(SlidingProcessingTimeWindows.of(Time.minutes(5), Time.minutes(1))) \
    .process(AvgAmtLast1Hour(),output_type=Types.MAP(Types.STRING(), Types.STRING()))

# distance_included.print()
# txn_count.print()
# avg_amount.print()

# Key each stream by cc_num for join compatibility
txn_count_keyed = txn_count.key_by(lambda x: x['cc_num'], key_type=Types.STRING())
avg_amount_keyed = avg_amount.key_by(lambda x: x['cc_num'], key_type=Types.STRING())

# Connect txn count and avg amount and combine
combined_txn_avg = txn_count_keyed.connect(avg_amount_keyed).process(
    CombineTxnAndAvg(),
    output_type=Types.MAP(Types.STRING(), Types.STRING())
)

distance_keyed = distance_included.key_by(lambda x: x['cc_num'], key_type=Types.STRING())
combined_txn_avg_keyed = combined_txn_avg.key_by(lambda x: x['cc_num'], key_type=Types.STRING())

# Combine distance with the joined stats
final_stream = distance_keyed.connect(combined_txn_avg_keyed).process(
    FinalJoiner(),
    output_type=Types.MAP(Types.STRING(), Types.STRING())
)

# Print final merged result
final_stream.print()

# Store on Redis
final_stream \
    .map(RedisWriter(host='localhost', port=6379, db=0, ttl=86400), 
         output_type=Types.MAP(Types.STRING(), Types.STRING())) \
    .print("Stored in Redis: ")

# Execute the stream processing pipeline
env.execute("PyFlink Kafka Stream")