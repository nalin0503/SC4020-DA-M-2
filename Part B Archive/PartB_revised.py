import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import trackintel as ti
from pymining import seqmining
import logging
from datetime import datetime
import os
import warnings 
warnings.filterwarnings("ignore") # uncomment to ignore warnings

#TODO
#order output by desc min_support
# convert output back into x,y coordinates for presentability.. 
# estimated run time - 20 seconds, after all the adjustments for computational load (could use strftime to time) 

# write report on gdocs explaining choice of params, how we reduced the computational load by splitting triplegs at 10 etc. 
# continued^ - removing consecutive duplicates - so reducing the computational load that way, also correctness of staypoint interpretation 
# aggregate trips for each user, then flatten. 

# min support set at 5% of length of gsp_sequences, the number of unique movement patterns by the users. 
# estimated run time - 20 seconds, after all the adjustments for computational load (could use strftime to time)
# 5 percent minsup works well, imo 

# did i do anything else special here? in terms of processing steps... 80% is preprocessing 20% is applying the model, as per the last lecture

logging.basicConfig(level=logging.INFO)

# Step 1: Load the dataset
df = pd.read_csv('/home/nalin/master/Y4S1/SC4020/hiroshima_challengedata.csv')

# Step 2: Filter for the first 30 days (days 0 to 29, indexed.)
df = df[(df['d'] >= 0) & (df['d'] < 30)].copy()

# Step 3: Scale x and y coordinates to reflect 500m spatial resolution
df['x'] = df['x'] * 500  # Convert x to meters
df['y'] = df['y'] * 500  # Convert y to meters

# Step 4: Create 'tracked_at' datetime column
df['tracked_at'] = pd.to_datetime(df['d'] * 86400 + df['t'] * 1800, unit='s', utc=True)

# Step 5: Combine x and y into geometry
df['geom'] = gpd.points_from_xy(df['x'], df['y'])

# Step 6: rename columns to match trackintel's expected structure
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
    dist_threshold=500,  # 500 meters
    time_threshold=60    # 60 minutes (adjust as needed!!) 
)

# At this point, 'positionfixes' will have a 'staypoint_id' column, even if 'staypoints' is empty.

# Step 11: Generate triplegs
logging.info('Generating triplegs.')
positionfixes, triplegs = ti.preprocessing.positionfixes.generate_triplegs(
    positionfixes,
    method='between_staypoints',
    gap_threshold=30
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
    coords_no_dups = [coords[0]] + [coords[i] for i in range(1, len(coords)) if coords[i] != coords[i-1]]
    
    # Handling length of triplegs: split long triplegs into shorter sub-triplegs.
    max_length = 10  # Adjust as needed!!
    for i in range(0, len(coords_no_dups), max_length):
        sub_coords = coords_no_dups[i:i+max_length]
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

# Prepare sequences for GSP..
gsp_sequences = []

for sequences in user_sequences.values():
    # Flatten the sequences for each user into a single sequence
    user_sequence = []
    for seq in sequences:
        user_sequence.extend(seq)
    gsp_sequences.append(user_sequence)

# print(len(gsp_sequences))
# raise 

# Step 14: Run the GSP algorithm using pymining
# Set minimum support
min_support = max(1, int(0.05 * len(gsp_sequences))) ## CURRENTLY SET AT 5% of gsp_sequences, gives decent # of output. 
logging.info(f"min_support chosen is {min_support}")

logging.info('Running GSP algorithm...')
freq_seqs = seqmining.freq_seq_enum(gsp_sequences, min_support)
freq_seqs = list(freq_seqs)

# Step 15: Save results to a unique text file in an 'output' directory
# Create the output directory if it doesn't exist.
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Generate a unique filename with a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file_path = os.path.join(output_dir, f'frequent_sequences_{timestamp}.txt')

# Write the frequent sequences and their support to a text file
with open(output_file_path, 'w') as f:
    for seq, support in freq_seqs:
        f.write(f'Sequence: {seq}, Support: {support}\n')

logging.info(f'saved frequent sequential patterns to {output_file_path}')


# print('Frequent Sequential Patterns:')
# for seq, support in freq_seqs:
#     print(f'Sequence: {seq}, Support: {support}')
