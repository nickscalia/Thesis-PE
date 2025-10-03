import os
import pandas as pd
import numpy as np

def myo_dataframe_edit(df):
    """
    Assigns standard column names to a DataFrame acquired from the Myo system.
    """
    column_names = ['channel_1', 'channel_2', 'channel_3', 'channel_4', 
                    'channel_5', 'channel_6', 'channel_7', 'channel_8', 
                    'ACC_X', 'ACC_Y', 'ACC_Z', 
                    'GYR_X','GYR_Y', 'GYR_Z', 
                    'roll', 'pitch', 'yaw']

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

def myo_extract_imu(dataframes):
    """
    Extract IMU data (acc, gyr, orientation) and corresponding time from DataFrames.
    """
    imu_columns = [ 'ACC_X', 'ACC_Y', 'ACC_Z', 'GYR_X','GYR_Y', 'GYR_Z',  'roll', 'pitch', 'yaw']
    imu_dict = {col: [] for col in imu_columns}
    IMU_Time = []
    fs_list = []

    for i, df in enumerate(dataframes):
        # Keep only rows where IMU values change
        imu_values = df[imu_columns]
        changed_rows = imu_values.shift() != imu_values
        changed_rows = changed_rows.any(axis=1)
        filtered_df = df[changed_rows].reset_index(drop=True)

        # Calculate fs as (num_kept_samples / total_samples) * 200
        total_samples = len(df)
        kept_samples = len(filtered_df)
        fs = (kept_samples / total_samples) * 200
        fs_list.append(fs)

        # Generate IMU time signal
        time_signal = np.arange(kept_samples) / fs
        IMU_Time.append(time_signal)

        # Fill imu_dict
        for col in imu_columns:
            imu_dict[col].append(filtered_df[col].to_numpy())

    return imu_dict, IMU_Time, fs_list