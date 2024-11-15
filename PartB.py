import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import trackintel as ti
from pymining import seqmining
import logging
from datetime import datetime
import os
import warnings 
import ast  # Added for converting strings back to tuples
import time  # Added for timing the script

warnings.filterwarnings("ignore")  # Uncomment to ignore warnings

# TODO
# run kotae subsets

# Choose from "hiroshima", "sapporo", "kumamoto", "kotae"
city = "kotae"
subset = 4 # for 'Kotae' only 

logging.basicConfig(level=logging.INFO)

start_time = time.time()  # Start timing the script

# Step 1: Load the dataset, INPUT your own paths here. 
if city == "kotae": 
    df = pd.read_csv(f"/home/nalin/master/Y4S1/SC4020/task1_dataset_kotae_subset_{subset}.csv")
else:
    df = pd.read_csv(f'/home/nalin/master/Y4S1/SC4020/{city}_challengedata.csv') 

# Or if you want to manually enter the file path: 
# df = pd.read_csv('') 

# Step 2: Filter for the first 30 days (days 0 to 29, indexed.)       
df = df[(df['d'] >= 0) & (df['d'] < 30)].copy()

# Step 3: Scale x and y coordinates to reflect 500m spatial resolution
df['x'] = df['x'] * 500  # Convert x to meters
df['y'] = df['y'] * 500  # Convert y to meters

# Step 4: Create 'tracked_at' datetime column
df['tracked_at'] = pd.to_datetime(df['d'] * 86400 + df['t'] * 1800, unit='s', utc=True)

# Step 5: Combine x and y into geometry
df['geom'] = gpd.points_from_xy(df['x'], df['y'])

# Step 6: Rename columns to match trackintel's expected structure
df = df.rename(columns={'uid': 'user_id'})

# Step 7: Create a GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry='geom')

# Step 8: Set the CRS (assuming meters)
gdf.set_crs(epsg=3857, inplace=True)  # Web Mercator projection

# Step 9: Prepare positionfixes for trackintel
positionfixes = gdf[['user_id', 'tracked_at', 'geom']].copy()
positionfixes.as_positionfixes  # Set accessor

# Step 10: Generate staypoints (even if none are found)
logging.info('Generating staypoints.')
positionfixes, staypoints = ti.preprocessing.positionfixes.generate_staypoints(  
    positionfixes,
    method='sliding',
    dist_threshold=1000,  # Constant on 2 adjacent blocks, 1000m
    time_threshold=60    # 60 minutes (adjust as needed!!)
)

# Note: staypoints currently not utilized / necessary to generate triplegs. 
# At this point, 'positionfixes' will have a 'staypoint_id' column, even if 'staypoints' is empty.

# Step 11: Generate triplegs
logging.info('Generating triplegs.')
positionfixes, triplegs = ti.preprocessing.positionfixes.generate_triplegs(
    positionfixes,
    # staypoints = staypoints,
    method='between_staypoints', # reminder that this method does not need staypoints to generate triplegs
    gap_threshold= 60 # we count a tripleg in a 1 hour period, 30 min intervals may generate a singular position tripleg
)

if triplegs.empty:
    logging.warning('No triplegs generated.')
else:
    logging.info(f'Generated {len(triplegs)} triplegs.')

# Step 12: Extract sequences of (x, y) coordinates from triplegs
tripleg_sequences = []

for idx, row in triplegs.iterrows():
    # Extract the LINESTRING geometry
    linestring = row['geom']

    # Get the list of (x, y) coordinates
    coords = list(linestring.coords)

    # Remove consecutive duplicates
    coords_no_dups = [coords[0]] + [coords[i] for i in range(1, len(coords)) if coords[i] != coords[i - 1]]

    # Handling length of triplegs: split long triplegs into shorter sub-triplegs.
    max_length = 10  # Adjust as needed!!
    for i in range(0, len(coords_no_dups), max_length):
        sub_coords = coords_no_dups[i:i + max_length]
        if len(sub_coords) > 1:
            tripleg_sequences.append({
                'user_id': row['user_id'],
                'sequence': sub_coords
            })
# Convert to DataFrame
tripleg_sequences_df = pd.DataFrame(tripleg_sequences)

# Step 13: Prepare sequences for GSP
user_sequences = dict()

for idx, row in tripleg_sequences_df.iterrows():
    user_id = row['user_id']
    sequence = row['sequence']
    # Convert coordinates to strings to use as items
    sequence_str = [str(coord) for coord in sequence]
    if user_id not in user_sequences:
        user_sequences[user_id] = [] 
    user_sequences[user_id].append(sequence_str)

# Prepare sequences for GSP
gsp_sequences = []

for sequences in user_sequences.values():
    for seq in sequences:
        gsp_sequences.append(seq)

# gsp_sequences is now a sequence of sequences, as required. 

# Step 14: Run the GSP algorithm using pymining

# Set minimum support
# Currently set at 0.1% of length of sequences possible. (Rough gauge set for frequent sequences across all sequences)
min_support = max(1, int(0.001 * len(gsp_sequences))) 
# min_support = 10
logging.info(f"min_support chosen is {min_support}")

logging.info('Running GSP algorithm...')
freq_seqs = seqmining.freq_seq_enum(gsp_sequences, min_support)
freq_seqs = list(freq_seqs) 

# Order output by descending support for better, more logical presentability 
freq_seqs = sorted(freq_seqs, key=lambda x: x[1], reverse=True)

# Step 15: Save results to a unique text file in the 'output' directory
# We save for results with singletons and without singletons (len>1), since 'trips' are logically atleast 2 coordinates. 
# However, the GSP algorithm entails that itemsets of length 1 may also be deemed frequent. 

if city =="kotae":
    output_dir = f'results_PartB/output_{city}_subset_{subset}'
else:
    output_dir = f'results_PartB/output_{city}'
os.makedirs(output_dir, exist_ok=True) # Create the output directory if it doesn't exist.

# Generate a unique filename with a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# File path for all frequent sequences
output_file_path_all = os.path.join(output_dir, f'frequent_sequences_all_{timestamp}.txt')
# File path for sequences of length > 1
output_file_path_filtered = os.path.join(output_dir, f'frequent_sequences_filtered_{timestamp}.txt')

# Write all sequences to the first file
with open(output_file_path_all, 'w') as f_all:
    for seq, support in freq_seqs:
        # Convert each item in seq back to (x, y) coordinates
        seq_coords = [ast.literal_eval(item) for item in seq]
        # Convert coordinates back to original grid cell numbers by dividing by 500
        seq_original_coords = [(int(round(coord[0] / 500)), int(round(coord[1] / 500))) for coord in seq_coords]
        f_all.write(f'Sequence: {seq_original_coords}, Support: {support}\n')

logging.info(f'Saved all frequent sequences to {output_file_path_all}')

# Write only sequences of length > 1 to the second file
with open(output_file_path_filtered, 'w') as f_filtered:
    for seq, support in freq_seqs:
        if len(seq) > 1:  # Only include sequences longer than 1
            # Convert each item in seq back to (x, y) coordinates
            seq_coords = [ast.literal_eval(item) for item in seq]
            # Convert coordinates back to original grid cell numbers by dividing by 500
            seq_original_coords = [(int(round(coord[0] / 500)), int(round(coord[1] / 500))) for coord in seq_coords]
            f_filtered.write(f'Sequence: {seq_original_coords}, Support: {support}\n')

logging.info(f'Saved frequent sequences of length > 1 to {output_file_path_filtered}')

# Log total runtime
end_time = time.time()
elapsed_time = end_time - start_time
logging.info(f"Total runtime: {elapsed_time:.2f} seconds")

# Print paths for confirmation
print(f'All sequences saved to: {output_file_path_all}')
print(f'Filtered sequences (length > 1) saved to: {output_file_path_filtered}')