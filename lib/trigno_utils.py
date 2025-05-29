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

def trigno_extract_muscle_emg(dataframes, muscle_name):
    """
    Extract specified muscle EMG and EMG time from DataFrames.
    """
    emg_col = f"{muscle_name}_EMG"
    muscle_EMG = []
    EMG_Time = []

    for i, df in enumerate(dataframes):
        if emg_col not in df.columns or 'EMG_Time' not in df.columns:
            raise ValueError(f"Columns '{emg_col}' or 'EMG_Time' not found in DataFrame {i}")
        
        emg_signal = df[emg_col].to_numpy()
        time_signal = df['EMG_Time'].to_numpy()
        
        muscle_EMG.append(emg_signal)
        EMG_Time.append(time_signal)
        
        #plot_emg_signal(emg_signal, time_signal, title=f"{muscle_name} EMG Signal #{i+1}")
    
    return muscle_EMG, EMG_Time