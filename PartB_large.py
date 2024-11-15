from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list, array, concat_ws, array_join, transform, size, isnan, when, count
from pyspark.sql.types import StringType
from pyspark.ml.fpm import PrefixSpan
import logging
import time
import os
from datetime import datetime
import pandas as pd
from load_data import adjust_coordinates

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

start_time = time.time()

# Initialize SparkSession with increased memory and configurations
spark = SparkSession.builder \
    .appName("SequentialPatternMining") \
    .config("spark.driver.memory", "8g") \
    .config("spark.driver.memoryOverhead", "4g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.memoryOverhead", "4g") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "4g") \
    .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
    .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC") \
    .getOrCreate()

# Set log level to DEBUG
# spark.sparkContext.setLogLevel("DEBUG")

# Enable checkpointing
spark.sparkContext.setCheckpointDir('/tmp/spark-checkpoints')

city = "kotae"

data_path = "/home/nalin/master/Y4S1/SC4020/task1_dataset_kotae_first_30_days.csv"

# Step 1: Load the dataset
logger.info(f"Loading data from {data_path}")
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Verify that 'd' column exists
if 'd' not in df.columns:
    logger.error("'d' column not found in the data.")
    spark.stop()
    exit(1)

# Step 2: Filter for the first 30 days
df = df.filter((col('d') >= 0) & (col('d') < 30))
logger.info("Filtered data for the first 30 days")

# Check for nulls
null_counts = df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns])
null_counts.show()

# Handle or drop nulls as appropriate
df = df.dropna(subset=['uid', 'x', 'y'])

# Step 3: Scale x and y coordinates to reflect 500m spatial resolution
df = df.withColumn('x', col('x') * 500)
df = df.withColumn('y', col('y') * 500)

# Step 4: Combine x and y into a single coordinate string
df = df.withColumn('coordinate', concat_ws(',', col('x').cast(StringType()), col('y').cast(StringType())))

# Step 5: Rename columns to match
df = df.withColumnRenamed('uid', 'user_id')

# Step 6: Select necessary columns
df = df.select('user_id', 'coordinate')

# Step 7: Wrap 'coordinate' in an array to create itemsets
df = df.withColumn('itemset', array(col('coordinate')))

# Group events by user_id into sequences of itemsets
df_sequences = df.groupBy('user_id').agg(collect_list('itemset').alias('sequence'))

# Analyze sequence lengths
sequence_lengths = df_sequences.withColumn('length', size(col('sequence')))
sequence_lengths.describe('length').show()

# Filter out long sequences
max_sequence_length = 100  # Adjust as needed
df_sequences_filtered = df_sequences.filter(size(col('sequence')) <= max_sequence_length)

# Cache the data
data_for_prefixspan = df_sequences_filtered.select('sequence').cache()
data_for_prefixspan.count()  # Force caching

# Limit data for testing, UNCOMMENT this to limit if having memory issues! 
# data_sample = data_for_prefixspan.limit(10000).cache()
# data_sample.count()

# Step 8: Run PrefixSpan algorithm with adjusted parameters
minSupport = 0.01  # 1 percent minSupport
maxPatternLength = 10  # can reduce to 5 if having memory issues
logger.info(f"Running PrefixSpan with minSupport = {minSupport}, maxPatternLength = {maxPatternLength}")
ps = PrefixSpan(minSupport=minSupport, maxPatternLength=maxPatternLength)

# Try running PrefixSpan and catch any exceptions
try:
    frequent_sequences = ps.findFrequentSequentialPatterns(data_for_prefixspan)
except Exception as e:
    logger.error("Error running PrefixSpan:", exc_info=True)
    spark.stop()
    exit(1)

# Step 9: Save results
output_dir = f'results_PartB/output_{city}'
os.makedirs(output_dir, exist_ok=True)

# Generate a unique directory name with a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(output_dir, f'frequent_sequences_{timestamp}')

# Convert sequences to a readable format using built-in functions
frequent_sequences_formatted = frequent_sequences.withColumn(
    'sequence_str',
    array_join(
        transform(
            col('sequence'),
            lambda itemset: array_join(itemset, '|')
        ),
        ';'
    )
)

# Select the formatted sequence and frequency
frequent_sequences_formatted = frequent_sequences_formatted.select('sequence_str', 'freq')

# Repartition before writing
frequent_sequences_formatted = frequent_sequences_formatted.repartition(100)

# Save frequent sequences to Parquet
output_parquet_path = output_path + '.parquet'
frequent_sequences_formatted.write.parquet(output_parquet_path, mode='overwrite')

logger.info(f'Saved frequent sequences to {output_parquet_path}')

# Log total runtime
end_time = time.time()
elapsed_time = end_time - start_time
logger.info(f"Total runtime: {elapsed_time:.2f} seconds")

# Stop SparkSession
spark.stop()

output_df = pd.read_parquet(output_parquet_path)
output_df['sequence_str'] = output_df['sequence_str'].apply(adjust_coordinates)
print(output_df)

save_path = f"results_PartB/results_{city}_{timestamp}.csv"
output_df.to_csv(save_path, index=False) 
logger.info(f"Results saved in csv format to {save_path}")