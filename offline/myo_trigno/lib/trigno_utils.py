import os
import pandas as pd
import numpy as np
from emg_utils import plot_emg_signal

def trigno_dataframe_edit(df):
    """
    Rename columns of a Trigno system DataFrame.
    """
    """column_names = [
        'EMG_Time', 'forearm2_EMG',
        'ACC_Time', 'forearm2_ACC_X_G',
        'forearm2_ACC_Y_G', 'forearm2_ACC_Z_G',

        'forearm1_EMG',
        'forearm1_ACC_X_G', 'forearm1_ACC_Y_G', 'forearm1_ACC_Z_G',

        'biceps_EMG',
        'biceps_ACC_X_G', 'biceps_ACC_Y_G', 'biceps_ACC_Z_G',

        'triceps_EMG',
        'triceps_ACC_X_G', 'triceps_ACC_Y_G', 'triceps_ACC_Z_G',
    ]"""

    column_names = [
        'EMG_Time', 'biceps_EMG',
        'IMU_Time', 'biceps_ACC_X', 'biceps_ACC_Y', 'biceps_ACC_Z',
        'biceps_GYR_X', 'biceps_GYR_Y', 'biceps_GYR_Z',

        'triceps_EMG',
        'triceps_ACC_X', 'triceps_ACC_Y', 'triceps_ACC_Z',
        'triceps_GYR_X', 'triceps_GYR_Y', 'triceps_GYR_Z',
    ]
    
    if len(column_names) == df.shape[1]:
        df.columns = column_names
        
    return df

def trigno_extract_muscle_emg(dataframes, muscle_names):
    """
    Extract specified muscle EMG and EMG time from DataFrames.
    """
    muscle_EMG_dict = {muscle: [] for muscle in muscle_names}
    EMG_Time = []

    for i, df in enumerate(dataframes):
        if 'EMG_Time' not in df.columns:
            raise ValueError(f"'EMG_Time' column not found in DataFrame {i}")
        
        time_signal = df['EMG_Time'].to_numpy()
        EMG_Time.append(time_signal)
        
        for muscle in muscle_names:
            emg_col = f"{muscle}_EMG"
            if emg_col not in df.columns:
                raise ValueError(f"Column '{emg_col}' not found in DataFrame {i}")
            
            emg_signal = df[emg_col].to_numpy()
            muscle_EMG_dict[muscle].append(emg_signal)

    return muscle_EMG_dict, EMG_Time

def trigno_extract_muscle_IMU(dataframes, muscle_names, axes=["X", "Y", "Z"]):
    """
    Extract specified muscle accelerations and angular velocities and IMU time from DataFrames.
    """
    # Initialize nested dictionary for ACC and GYR signals
    muscle_IMU_dict = {
        muscle: {
            f"{sensor}_{axis}": [] for sensor in ['ACC', 'GYR'] for axis in axes
        } for muscle in muscle_names
    }
    IMU_Time = []
    
    for i, df in enumerate(dataframes):
        df = df.copy()  
        df['IMU_Time'] = df['IMU_Time'].astype(str).str.strip()
        df_clean = df[df['IMU_Time'] != '']
        time_signal = df_clean['IMU_Time'].astype(float).to_numpy()
        IMU_Time.append(time_signal)
    
        for muscle in muscle_names:
            for sensor in ['ACC', 'GYR']:
                for axis in axes:
                    col_name = f"{muscle}_{sensor}_{axis}"
                    if col_name not in df_clean.columns:
                        raise ValueError(f"Column '{col_name}' not found in DataFrame {i}")
    
                    df_clean.loc[:, col_name] = df_clean[col_name].astype(str).str.strip()
                    df_valid = df_clean[df_clean[col_name] != '']
                    imu_signal = df_valid[col_name].astype(float).to_numpy()
                    muscle_IMU_dict[muscle][f"{sensor}_{axis}"].append(imu_signal)
    
    return muscle_IMU_dict, IMU_Time