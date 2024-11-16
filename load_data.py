""""
Utility functions for data management
"""
import pandas as pd

def load_data(file_path, sample_size=None):
    data = pd.read_csv(file_path)
    if sample_size is not None:
        data = data.sample(sample_size, random_state=42)
    # in future, we will add more data preprocessing steps here
    return data

# Divide each coordinate value by 500 in sequence_str column
def adjust_coordinates(sequence_str):
    adjusted_sequence = []
    for coord_set in sequence_str.split(';'):
        coords = coord_set.split('|')
        adjusted_coords = []
        for coord in coords:
            x, y = map(float, coord.split(','))
            adjusted_coords.append(f"{int(x) // 500},{int(y) // 500}")
        adjusted_sequence.append('|'.join(adjusted_coords))
    return ';'.join(adjusted_sequence)
