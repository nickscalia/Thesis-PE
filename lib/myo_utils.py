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
    column_names = ['channel_1', 'channel_2', 'channel_3', 'channel_4', 'channel_5', 'channel_6', 'channel_7', 'channel_8']

    if len(column_names) == df.shape[1]:
        df.columns = column_names

    return df

def myo_extract_muscle_emg(dataframes, channel_names, fs):
    """
    Extract specified channel EMG and EMG time from DataFrames.
    """
    muscle_EMG_dict = {channel: [] for channel in channel_names}
    EMG_Time = []

    for i, df in enumerate(dataframes):
        num_samples = len(df)
        time_signal = np.arange(num_samples) / fs
        EMG_Time.append(time_signal)
        
        for channel in channel_names:
            emg_col = f"{channel}"
            if emg_col not in df.columns:
                raise ValueError(f"Column '{emg_col}' not found in DataFrame {i}")
            
            emg_signal = df[emg_col].to_numpy()
            muscle_EMG_dict[channel].append(emg_signal)

    return muscle_EMG_dict, EMG_Time