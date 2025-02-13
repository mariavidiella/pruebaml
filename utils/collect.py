"""
file for store useful function to collecting data
"""

import pandas as pd  # type: ignore
import os
from sklearn.model_selection import train_test_split # type: ignore

def collect_and_join(folder, test_split_size = 0.3, random_seed = 42):
    """
    function that collects csv files from a folder and concatanate in a pandas dataframe
    args:
        folder: folder from read csv files
    return:
        pandas DataFrane
    """
    dirs_files = [
        os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')
    ]
    # Concatenate all csv files
    dfs = []

    for file in dirs_files:
        df = pd.read_csv(file)
        city_name1 = file.split('/')[-1]
        city_name = city_name1.split('.csv')[0]
        df.insert(1,"city", city_name)
        dfs.append(df)
    df_combined = pd.concat(dfs, ignore_index= True)
    df_combined = df_combined.sample(frac = 1) 

    # Split train test
    train_df, test_df = train_test_split(
        df_combined, test_size= test_split_size, random_state= random_seed
    )
    
    return df_combined, train_df, test_df
    