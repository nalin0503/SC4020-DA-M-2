"""
This script assists in splitting larger datasets into smaller subsets, based on the day 'd' and time 't' columns. 
For Kotae city's dataset, it is simply too large for the hardware at hand, hence we split it into 4 subsets, 
based on the first 30 days of data. 
The script allows us additional control on each subset's size in terms of days and times considered, however for the 
final version, we select rows that are in the first 30 days of data and split into four 6-hour time intervals. 
"""
import pandas as pd

# Define file paths
input_csv = "/home/nalin/master/Y4S1/SC4020/task1_dataset_kotae.csv"
output_csv = "/home/nalin/master/Y4S1/SC4020/task1_dataset_kotae_subset_4.csv"

# Define the range of 'd' values to select
d_start = 0
d_end = 29  # Inclusive

# Define the range of 't' values to select
t_start = 36
t_end = 47 # Inclusive

# Set the chunk size (number of rows per chunk)
chunksize = 10 ** 6  # Adjust based on system's memory capacity

# Initialize a variable to track if it's the first chunk
is_first_chunk = True

# Read and process the CSV file in chunks
with pd.read_csv(input_csv, chunksize=chunksize) as reader:
    for chunk in reader:
        # Apply filters to the chunk
        filtered_chunk = chunk[
            (chunk['d'] >= d_start) & (chunk['d'] <= d_end) &
            (chunk['t'] >= t_start) & (chunk['t'] <= t_end)
        ]

        # Write the filtered chunk to the output CSV file
        if not filtered_chunk.empty:
            if is_first_chunk:
                # Write header for the first chunk
                filtered_chunk.to_csv(output_csv, mode='w', index=False)
                is_first_chunk = False
            else:
                # Append without writing  header
                filtered_chunk.to_csv(output_csv, mode='a', index=False, header=False)
