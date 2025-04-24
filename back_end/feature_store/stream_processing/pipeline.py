from pyflink.datastream import StreamExecutionEnvironment, TimeCharacteristic
from pyflink.common import Time, Duration, Types
from pyflink.common.serialization import SimpleStringSchema
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer
from pyflink.common.watermark_strategy import WatermarkStrategy, TimestampAssigner
from pyflink.datastream.window import SlidingProcessingTimeWindows
import json, os
from stream_features import TxnCountLast10Min, AvgAmtLast1Hour, CombineTxnAndAvg, FinalJoiner
from redis_writer import RedisWriter
import datetime
from math import radians, sin, cos, sqrt, atan2

class TransactionTimestampAssigner(TimestampAssigner):
    def extract_timestamp(self, element, record_timestamp):
        return int(datetime.datetime.fromisoformat(element['timestamp']).timestamp() * 1000)

def haversine(lat1, lon1, lat2, lon2):
    R = 3963
    lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

env = StreamExecutionEnvironment.get_execution_environment()
env.set_stream_time_characteristic(TimeCharacteristic.EventTime)

jars_dir = "./jars"
print(f"file://{os.path.abspath(os.path.join(jars_dir, 'flink-connector-kafka-3.3.0-1.20.jar'))}")
env.add_jars(f"file://{os.path.abspath(os.path.join(jars_dir, 'flink-connector-kafka-3.3.0-1.20.jar'))}")
env.add_jars(f"file://{os.path.abspath(os.path.join(jars_dir, 'kafka-clients-7.9.0-ccs.jar'))}")

properties = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'pyflink-consumer'
}

stream = env.add_source(FlinkKafkaConsumer(
    topics='transactions',
    deserialization_schema=SimpleStringSchema(),
    properties=properties
))

parsed = stream.map(lambda raw: json.loads(raw), output_type=Types.MAP(Types.STRING(), Types.STRING()))

parsed = parsed.assign_timestamps_and_watermarks(
    WatermarkStrategy.for_bounded_out_of_orderness(Duration.of_seconds(5))
    .with_timestamp_assigner(TransactionTimestampAssigner())
)

distance_included = parsed.map(
    lambda x: {
        **x,
        'distance_to_merchant': str(haversine(x['lat'], x['long'], x['merch_lat'], x['merch_long']))
    },
    output_type=Types.MAP(Types.STRING(), Types.STRING())
)

txn_count = distance_included \
    .key_by(lambda t: t['cc_num']) \
    .window(SlidingProcessingTimeWindows.of(Time.minutes(2), Time.seconds(20))) \
    .process(TxnCountLast10Min(), output_type=Types.MAP(Types.STRING(), Types.STRING()))

avg_amount = distance_included \
    .key_by(lambda t: t['cc_num']) \
    .window(SlidingProcessingTimeWindows.of(Time.minutes(5), Time.minutes(1))) \
    .process(AvgAmtLast1Hour(), output_type=Types.MAP(Types.STRING(), Types.STRING()))

combined_txn_avg = txn_count.key_by(lambda x: x['cc_num']) \
    .connect(avg_amount.key_by(lambda x: x['cc_num'])) \
    .process(CombineTxnAndAvg(), output_type=Types.MAP(Types.STRING(), Types.STRING()))

final_stream = distance_included.key_by(lambda x: x['cc_num']) \
    .connect(combined_txn_avg.key_by(lambda x: x['cc_num'])) \
    .process(FinalJoiner(), output_type=Types.MAP(Types.STRING(), Types.STRING()))

final_stream \
    .map(RedisWriter(host='localhost', port=6379, db=0, ttl=86400),
         output_type=Types.MAP(Types.STRING(), Types.STRING())) \
    .print("Stored in Redis: ")

env.execute("PyFlink Kafka Stream")