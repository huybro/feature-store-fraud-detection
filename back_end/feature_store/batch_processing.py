from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek, when, avg, count, radians, asin, sin, sqrt, cos
from pyspark.sql.window import Window
from pyspark.sql.functions import to_timestamp

# Initialize Spark session
spark = SparkSession.builder \
    .appName("CreditCardTransactionsBatchProcessing") \
    .getOrCreate()

# Load the CSV data into a DataFrame
df = spark.read.csv("data/credit_card_transactions.csv", header=True, inferSchema=True)


df = df.withColumn("dob", to_timestamp(col("dob"))) \
       .withColumn("trans_date_trans_time", to_timestamp(col("trans_date_trans_time")))

# Compute the required features
df = df.withColumn("hour_of_day", hour(col("trans_date_trans_time"))) \
        .withColumn("day_of_week", dayofweek(col("trans_date_trans_time"))) \
        .withColumn("age_at_txn",((col("trans_date_trans_time").cast("long") - col("dob").cast("long")) / (60 * 60 * 24 * 365.25)))\
        .withColumn("dlon", radians(col("long")) - radians(col("merch_long"))) \
        .withColumn("dlat", radians(col("lat")) - radians(col("merch_lat"))) \
        .withColumn("distance_to_merchant", asin(sqrt(
                                                sin(col("dlat") / 2) ** 2 + cos(radians(col("lat")))
                                                * cos(radians(col("merch_lat"))) * sin(col("dlon") / 2) ** 2
                                                )
                                            ) * 2 * 3963) \
        .drop("dlon", "dlat")\
        
#Define window for 10_min and 1_Hour
window_spec_10_min = Window.partitionBy("cc_num").orderBy("unix_time").rangeBetween(-600, 0)
window_spec_1_hour = Window.partitionBy("cc_num").orderBy("unix_time").rangeBetween(-3600, 0)

df = df.withColumn("txn_count_last_10_min", count("*").over(window_spec_10_min)) \
       .withColumn("avg_amt_last_1_hour", avg("amt").over(window_spec_1_hour))
 
# Select the required columns
df = df.select("cc_num","amt", "hour_of_day", "day_of_week", "age_at_txn", "distance_to_merchant", "txn_count_last_10_min", "avg_amt_last_1_hour",
               "category", "gender", "city_pop", "is_fraud")

# Show the transformed data
df.show()

# Stop the Spark session
spark.stop()
