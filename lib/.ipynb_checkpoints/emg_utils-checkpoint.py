import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter

def plot_emg_signal(signal, time, title="Signal", xlabel="Time [s]", ylabel="mV"):
    """
    Plots a signal with customizable axis labels and title.

    Parameters:
        signal (array-like): The signal data to plot.
        title (str): Plot title.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    plt.plot(time, signal)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def bandpass_filter(signal, fs, low_freq=20, high_freq=450, order=4):
    """
    Applies a Butterworth bandpass filter to the input signal, allowing frequencies 
    between low_freq and high_freq Hz to pass and attenuating frequencies outside this range.
    """
    b, a = butter(order, [low_freq, high_freq], fs=fs, btype='band')
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal

def rectification(signal):
    """
    Computes the rectified version of the input signal by taking the absolute value of each sample, 
    effectively converting all negative values to positive.
    """
    return np.abs(signal)

def RMS_moving(signal, fs, time_window=0.2):
    """
    Calculates the moving Root Mean Square (RMS) of the input signal using a sliding window of specified duration (time_window in seconds). 
    The function interpolates RMS values between windows for smoother transitions and pads the result to match the input signal length.
    """
    window_length = int(time_window * fs)
    rms_result = []

    for start_idx in range(0, len(signal) - window_length + 1, window_length):
        segment = np.array(signal[start_idx:start_idx + window_length])
        rms_value = np.sqrt(np.mean(segment ** 2))
        if rms_result:
            interpolated_values = np.linspace(rms_result[-1], rms_value, window_length)
        else:
            interpolated_values = np.linspace(signal[0], rms_value, window_length)
        rms_result.extend(interpolated_values)
        
    # Padding 
    if len(rms_result) < len(signal):
        padding_length = len(signal) - len(rms_result)
        rms_result = np.concatenate((rms_result, np.full(padding_length, rms_result[-1])))

    return rms_result

def MVC_normalization(signal, muscle_name, mvc_csv_path='../data/_MVC/Combined_dataset.csv'):
    """
    Normalize the smoothed EMG signal by the MVC value read from a CSV file.

    Parameters:
        signal (np.ndarray): The EMG signal to normalize.
        muscle_name (str): The muscle name to look up the MVC value (row label in CSV).
        mvc_csv_path (str): Path to the CSV file containing MVC values. Default set to '../data/_MVC/Combined_dataset.csv'.

    Returns:
        np.ndarray: Normalized EMG signal.
    """
    df = pd.read_csv(mvc_csv_path, index_col=0)
    
    if muscle_name not in df.index:
        raise ValueError(f"Muscle name '{muscle_name}' not found in MVC CSV.")
    
    mvc = df.loc[muscle_name, 'MVC']
    
    if mvc == 0:
        raise ValueError(f"MVC value for muscle '{muscle_name}' is zero, cannot normalize.")
    
    normalized_emg = signal / mvc
    
    return normalized_emg

def emg_filters(muscle_emg_raw, emg_time, muscle_name):
    """
    Applies bandpass filtering, rectification, and RMS moving smoothing
    to a list of raw muscle EMG signals, using their corresponding time arrays.

    Parameters:
        muscle_emg_raw (list of np.ndarray): List of raw EMG signals.
        emg_time (list of np.ndarray): List of time vectors corresponding to each EMG signal.

    Returns:
        tuple: 
            - muscle_emg_filtered (list of np.ndarray): Filtered EMG signals.
            - muscle_emg_rectified (list of np.ndarray): Rectified EMG signals.
            - muscle_emg_smoothed (list of np.ndarray): Smoothed EMG signals.
    """
    muscle_emg_filtered = []
    muscle_emg_rectified = []
    muscle_emg_smoothed  = []
    muscle_emg_normalized  = []

    for emg_signal, time_signal in zip(muscle_emg_raw, emg_time):
        dt = np.mean(np.diff(time_signal))
        fs = 1 / dt

        filtered_emg = bandpass_filter(emg_signal, fs)
        rectified_emg = rectification(filtered_emg)
        smoothed_emg = RMS_moving(rectified_emg, fs, time_window=0.2)
        normalized_emg = MVC_normalization(smoothed_emg, muscle_name)

        muscle_emg_filtered.append(filtered_emg)
        muscle_emg_rectified.append(rectified_emg)
        muscle_emg_smoothed.append(smoothed_emg)
        muscle_emg_normalized.append(normalized_emg)

    return muscle_emg_filtered, muscle_emg_rectified, muscle_emg_smoothed, muscle_emg_normalized

def compute_MVC(emg_signal, fs, window_ms=500):
    window_samples = int((window_ms / 1000) * fs)
    step = int(window_samples)
    
    if window_samples > len(emg_signal):
        raise ValueError("Window is larger than signal length")

    max_mean = 0
    for start in range(0, len(emg_signal) - window_samples + 1, step):
        window = emg_signal[start:start + window_samples]
        window_mean = np.mean(window)
        if window_mean > max_mean:
            max_mean = window_mean

    return max_mean