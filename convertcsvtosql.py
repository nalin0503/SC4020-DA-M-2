import sqlite3
import pandas as pd


folder_path = r"C:\Users\All Saints\Desktop\Uni mods\SC4020\Project2\SC4020-DA-M-2\citymovementdata"

# Define file paths
csv_file_names = ["hiroshima_challengedata", "kumamoto_challengedata", "sapporo_challengedata"]
db_name = "full_data.db"

db_path = db_name

with sqlite3.connect(db_path) as conn:
    
    for csv_file_name in csv_file_names:
        
        print(f"Loading {csv_file_name} into {db_name}")
        
        csv_file_path = folder_path + "\\" + csv_file_name + ".csv"
        # Read and load CSV in chunks
        
        chunksize = 10000
        for chunk in pd.read_csv(csv_file_path, chunksize=chunksize):
            chunk.to_sql(name=csv_file_name, con=conn, if_exists="append", index=False)
    
        # check if the data is loaded
        query = f"SELECT * FROM {csv_file_name}"
        df = pd.read_sql(query, conn)
        print(df.head())
        
    

    


