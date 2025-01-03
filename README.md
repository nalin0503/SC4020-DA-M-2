# Frequent Itemset & Sequential Pattern Mining Project for Mobility Data

## Installation Instructions

To set up the environment, install the required dependencies using the following command:

```bash
pip install fastparquet geopandas numpy pandas pymining pyspark torch torchvision torchaudio trackintel logging warnings datetime ast time queue itertools collections matplotlib pyarrow
```
## Directory structure for Part A

- **`Part A code/`**: Contains all the code used to generate the results for association mining using Apriori
  - `apriori.ipynb` has everything inside.
 
- **`Part A results/`**: Contains all the results for association mining using Apriori
- **`Part A archive/`**: May be ignored


## Directory Structure for Part B

- **`Part B results/`**: Contains all the results generated by the scripts.
  - For `PartB_large.py`, a parquet file is saved in `output_kotae/`, along with a CSV.
  - For other cities, results are directly saved as TXT files in `output_{city}/`.

- **`Part B archive/`**: Contains research and utility scripts.

- **`PartB_normal.py`**: Script to run frequent sequential pattern mining on moderately sized datasets.

- **`PartB_large.py`**: Script to run frequent sequential pattern mining on large-sized datasets.
  - **Note**: This script uses Apache Spark. Configure the Spark session appropriately based on your system's available RAM.
  
### Usage

Run the respective scripts depending on the dataset size. Results will be automatically saved in the `results_PartB/` directory.

## Directory structure for Part C

### Running the application
Run `cd` into the directory and run `application.py` in order to run our command line application! It basically loads a saved model and does prediction  

- **`Part C code/`**: Is the main folder; all the code used to preprocess the sequence data and train the RNN
  - `PartC_Preprocess.ipynb` preprocesses the sequence data to prepare it for the RNN
  - `phrases2matrix` loads a pre-trained BERT word embeddings as a pytorch tensor (matrix) and saves it into a phrase_embeddings.pt 
  - `RNN.ipynb` is a step-by-step guide on how the RNN is set up
  - `RNN.py` is basically the same as the notebook, but **you may run it to train and save the model**
  
 



