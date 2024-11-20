# Frequent Itemset & Sequential Pattern Mining Project for Mobility Data

## Installation Instructions

To set up the environment, install the required dependencies using the following command:

```bash
pip install fastparquet geopandas numpy pandas pymining pyspark torch torchvision torchaudio trackintel logging warnings datetime ast time queue itertools collections matplotlib
```

## Directory Structure for Part B

- **`results_PartB/`**: Contains all the results generated by the scripts.
  - For `PartB_large.py`, a parquet file is saved in `output_kotae/`, along with a CSV.
  - For other cities, results are directly saved as TXT files in `output_{city}/`.

- **`PartB_archive/`**: Contains research and utility scripts.

- **`PartB_normal.py`**: Script to run frequent sequential pattern mining on moderately sized datasets.

- **`PartB_large.py`**: Script to run frequent sequential pattern mining on large-sized datasets.
  - **Note**: This script uses Apache Spark. Configure the Spark session appropriately based on your system's available RAM.

### Usage

Run the respective scripts depending on the dataset size. Results will be automatically saved in the `results_PartB/` directory.

