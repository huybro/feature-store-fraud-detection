from pyflink.datastream import StreamExecutionEnvironment, TimeCharacteristic
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common import Time, WatermarkStrategy, Types, Duration
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer
from pyflink.datastream.functions import ProcessWindowFunction, RuntimeContext
from pyflink.datastream.window import TumblingEventTimeWindows, SlidingEventTimeWindows
from pyflink.datastream.state import ValueStateDescriptor
from pyflink.common.watermark_strategy import TimestampAssigner
from pyflink.datastream.window import SlidingProcessingTimeWindows
from math import radians, sin, cos, sqrt, atan2
from typing import Iterable
import datetime
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
jars_dir = "/Users/huybro/Desktop/feature_store_fraud_detection/back_end/feature_store/stream_processing/jars" 

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

# Count transactions in the last 10 minutes (sliding every 1 minute) (currently testing with a smaller window of 10 seconds)
txn_count = distance_included \
    .key_by(lambda t: t['cc_num']) \
    .window(SlidingProcessingTimeWindows.of(Time.seconds(10), Time.seconds(1))) \
    .process(TxnCountLast10Min(),output_type=Types.MAP(Types.STRING(), Types.STRING()))


# Average amount in the last 1 hour (sliding every 1 minute) (currently testing with a smaller window of 10 seconds)
avg_amount = distance_included \
    .key_by(lambda t: t['cc_num']) \
    .window(SlidingProcessingTimeWindows.of(Time.seconds(10), Time.seconds(1))) \
    .process(AvgAmtLast1Hour(),output_type=Types.MAP(Types.STRING(), Types.STRING()))

distance_included.print()
txn_count.print()
avg_amount.print()

# Execute the stream processing pipeline
env.execute("PyFlink Kafka Stream")