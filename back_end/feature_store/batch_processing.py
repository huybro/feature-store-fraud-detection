from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, hour, dayofweek, avg, count,
    radians, asin, sin, sqrt, cos,
    to_timestamp, unix_timestamp
)
from pyspark.sql.window import Window
import pandas as pd
import requests
import os
import glob

# Initialize Spark session
spark = SparkSession.builder \
    .appName("CreditCardTransactionsBatchProcessing") \
    .getOrCreate()

# Load the CSV data into a DataFrame
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
credits_card_transaction_path = os.path.join(base_dir, 'data', 'credit_card_transactions.csv')
df = spark.read.csv(credits_card_transaction_path, header=True, inferSchema=True)

# Preprocess timestamps
df = df.withColumn("dob", to_timestamp(col("dob"))) \
       .withColumn("trans_date_trans_time", to_timestamp(col("trans_date_trans_time"))) \
       .withColumn("unix_time", unix_timestamp(col("trans_date_trans_time")))

# Compute features
df = df.withColumn("hour_of_day", hour(col("trans_date_trans_time"))) \
       .withColumn("day_of_week", dayofweek(col("trans_date_trans_time"))) \
       .withColumn("age_at_txn", ((col("trans_date_trans_time").cast("long") - col("dob").cast("long")) / (60 * 60 * 24 * 365.25))) \
       .withColumn("dlon", radians(col("long")) - radians(col("merch_long"))) \
       .withColumn("dlat", radians(col("lat")) - radians(col("merch_lat"))) \
       .withColumn("distance_to_merchant", asin(sqrt(
           sin(col("dlat") / 2) ** 2 +
           cos(radians(col("lat"))) * cos(radians(col("merch_lat"))) * sin(col("dlon") / 2) ** 2
       )) * 2 * 3963) \
       .drop("dlon", "dlat")

# Define time windows
window_spec_10_min = Window.partitionBy("cc_num").orderBy("unix_time").rangeBetween(-600, 0)
window_spec_1_hour = Window.partitionBy("cc_num").orderBy("unix_time").rangeBetween(-3600, 0)

df = df.withColumn("txn_count_last_10_min", count("*").over(window_spec_10_min)) \
       .withColumn("avg_amt_last_1_hour", avg("amt").over(window_spec_1_hour))

df = df.select("cc_num", "amt", "hour_of_day", "day_of_week", "age_at_txn", "distance_to_merchant",
               "txn_count_last_10_min", "avg_amt_last_1_hour", "category", "gender", "city_pop", "trans_date_trans_time","is_fraud")

# Save to CSV (writes to directory with part files)
output_path = os.path.join(base_dir, 'data', 'output_features')
df.write.csv(output_path, header=True, mode="overwrite")

spark.stop()


