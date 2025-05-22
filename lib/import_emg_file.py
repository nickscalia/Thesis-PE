import os
import pandas as pd

def import_data_from_trigno(folder_path):
    """
    Imports EMG data from CSV files recorded with the Delsys Trigno system.
    
    Parameters:
        folder_path (str): Path to the folder containing the CSV files.
    
    Returns:
        list of pd.DataFrame: A list of preprocessed DataFrames, one for each CSV file.
    """
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    dataframes = []

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        # Skip initial 8 rows with generic data
        df = pd.read_csv(file_path, skiprows=8, header=None, usecols=range(32))
        
        # Keep columns 0-3 and every second column from index 5 onward (odd indices)
        columns_to_keep = list(range(4)) + [i for i in range(5, df.shape[1]) if i % 2 == 1]
        df = df.iloc[:, columns_to_keep]

        dataframes.append(df)

    return dataframes