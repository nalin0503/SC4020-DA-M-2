import pandas as pd

def load_data(file_path, sample_size=None):
    data = pd.read_csv(file_path)
    if sample_size is not None:
        data = data.sample(sample_size, random_state=42)
    # in future, we will add more data preprocessing steps here
    return data