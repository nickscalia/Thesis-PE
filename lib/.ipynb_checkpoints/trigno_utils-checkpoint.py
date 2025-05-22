import os
import pandas as pd
import numpy as np
from emg_utils import plot_emg_signal

def trigno_dataframe_edit(df):
    """
    Assigns standard column names to a DataFrame acquired from the Trigno system.

    Parameters:
        df (pd.DataFrame): The DataFrame to be renamed.

    Returns:
        pd.DataFrame: The same DataFrame with updated column names.
    """
    column_names = [
        'EMG_Time', 'Forearm2_EMG',
        'ACC_Time', 'Forearm2_ACC_X_G',
        'Forearm2_ACC_Y_G', 'Forearm2_ACC_Z_G',

        'Forearm1_EMG',
        'Forearm1_ACC_X_G', 'Forearm1_ACC_Y_G', 'Forearm1_ACC_Z_G',

        'Biceps_EMG',
        'Biceps_ACC_X_G', 'Biceps_ACC_Y_G', 'Biceps_ACC_Z_G',

        'Triceps_EMG',
        'Triceps_ACC_X_G', 'Triceps_ACC_Y_G', 'Triceps_ACC_Z_G',
    ]

    if len(column_names) == df.shape[1]:
        df.columns = column_names

    return df

def trigno_extract_muscle_emg(dataframes, muscle_name):
    """
    Extracts 'Muscle_EMG' and 'EMG_Time' from a list of DataFrames,
    plots each Muscle EMG signal,
    and returns the extracted data as two lists of numpy arrays.

    Parameters:
        dataframes (list of pd.DataFrame): List of input DataFrames.
        plot_emg_signal: Function to plot a single EMG signal. 
                         Must accept parameters (signal, title).

    Returns:
        tuple: (Muscle_EMG, EMG_Time)
            Muscle_EMG (list of np.ndarray): List of Muscle EMG signals.
            EMG_Time (list of np.ndarray): List of corresponding time arrays.
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
        
        plot_emg_signal(emg_signal, time_signal, title=f"{muscle_name} EMG Signal #{i+1}")
    
    return muscle_EMG, EMG_Time
