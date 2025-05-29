import os
import pandas as pd

def import_data_from_trigno(folder_path):
    """
    Import EMG data from Trigno CSV files in a folder.
    """
    # List CSV files in folder
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    dataframes = []

    for file_name in csv_files:
        file_path = os.path.join(folder_path, file_name)
        # Read CSV skipping first 8 rows, no header, first 32 columns
        df = pd.read_csv(file_path, skiprows=8, header=None, usecols=range(32))
        
        # Keep columns 0-3 and odd columns from 5 onward
        cols_to_keep = list(range(4)) + [i for i in range(5, df.shape[1]) if i % 2 == 1]
        df = df.iloc[:, cols_to_keep]

        dataframes.append(df)

    return dataframes