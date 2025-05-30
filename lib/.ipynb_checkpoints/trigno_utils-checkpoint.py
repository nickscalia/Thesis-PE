import os
import pandas as pd
import numpy as np
from emg_utils import plot_emg_signal

def trigno_dataframe_edit(df):
    """
    Rename columns of a Trigno system DataFrame.
    """
    column_names = [
        'EMG_Time', 'forearm2_EMG',
        'ACC_Time', 'forearm2_ACC_X_G',
        'forearm2_ACC_Y_G', 'forearm2_ACC_Z_G',

        'forearm1_EMG',
        'forearm1_ACC_X_G', 'forearm1_ACC_Y_G', 'forearm1_ACC_Z_G',

        'biceps_EMG',
        'biceps_ACC_X_G', 'biceps_ACC_Y_G', 'biceps_ACC_Z_G',

        'triceps_EMG',
        'triceps_ACC_X_G', 'triceps_ACC_Y_G', 'triceps_ACC_Z_G',
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