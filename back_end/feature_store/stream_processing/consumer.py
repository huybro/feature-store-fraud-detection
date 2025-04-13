from pyflink.datastream import StreamExecutionEnvironment
from pyflink.common.serialization import DeserializationSchema, SimpleStringSchema
from pyflink.common.typeinfo import Types
from pyflink.common import WatermarkStrategy
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer

import json
import os

env = StreamExecutionEnvironment.get_execution_environment()

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
parsed.print()

# Execute the stream processing pipeline
env.execute("PyFlink Kafka Stream")