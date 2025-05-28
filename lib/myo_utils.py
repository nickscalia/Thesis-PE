import os
import pandas as pd
import numpy as np

def myo_dataframe_edit(df):
    """
    Assigns standard column names to a DataFrame acquired from the Myo system.

    Parameters:
        df (pd.DataFrame): The DataFrame to be renamed.

    Returns:
        pd.DataFrame: The same DataFrame with updated column names.
    """
    column_names = ['Channel_1', 'Channel_2', 'Channel_3', 'Channel_4', 'Channel_5', 'Channel_6', 'Channel_7', 'Channel_8']

    if len(column_names) == df.shape[1]:
        df.columns = column_names

    return df

"""def myo_extract_muscle_emg(dataframes, channel_number):
    Extracts 'Channel_number' from a list of DataFrames, compute 'emg_time' knowning Myo frequency,
    and returns the extracted data as two lists of numpy arrays.
    emg_col = f"Channel_{channel_number}"
    channel_EMG = []
    EMG_Time = []

    for i, df in enumerate(dataframes):
        if emg_col not in df.columns:
            raise ValueError(f"Columns '{emg_col}' not found in DataFrame {i}")
        
        emg_signal = df[emg_col].to_numpy()
        time_signal = df['EMG_Time'].to_numpy()
        
        muscle_EMG.append(emg_signal)
        EMG_Time.append(time_signal)
        
        #plot_emg_signal(emg_signal, time_signal, title=f"Channel_{channel_number} EMG Signal #{i+1}")
    
    return muscle_EMG, EMG_Time"""
