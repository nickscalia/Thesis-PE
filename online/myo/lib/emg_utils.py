import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from libemg.feature_extractor import FeatureExtractor

def remove_overlap(prev, curr, tol):
    """
    Removes the initial part of curr if it is already contained
    as a suffix of prev, within a tolerance tol.
    """
    max_ol = min(len(prev), len(curr))
    for ol in range(max_ol, 0, -1):
        if np.allclose(prev[-ol:], curr[:ol], atol=tol):
            #print("Removing")
            return curr[ol:]
    return curr 


def bandpass_filter(signal, fs, low_freq=20, high_freq=450, order=4):
    """
    Applies a Butterworth bandpass filter to the input signal.
    """
    b, a = butter(order, [low_freq, high_freq], fs=fs, btype='band')
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal

def rectification(signal):
    """
    Computes the rectified signal (absolute value).
    """
    return np.abs(signal)

def RMS_moving(signal, fs, time_window=0.2):
    """
    Calculates moving RMS with interpolation between windows.
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
        
    if len(rms_result) < len(signal):
        padding_length = len(signal) - len(rms_result)
        rms_result = np.concatenate((rms_result, np.full(padding_length, rms_result[-1])))

    return rms_result

def normalization(signal, muscle_name, norm_values):
    """
    Normalizes EMG signal by norm value from a dictionary.
    """
    if muscle_name not in norm_values:
        raise ValueError(f"Muscle name '{muscle_name}' not found in norm dictionary.")   
    norm = norm_values[muscle_name]
    return signal / norm

def extract_emg_features(windows_dict, feature_list=None, feature_group=None):
    """
    Extracts EMG features from windows using libemg.
    """
    fe = FeatureExtractor()
    
    trial_count = len(next(iter(windows_dict.values())))  # Number of trials

    all_features = []

    for trial_idx in range(trial_count):
        feature_dfs = []
        
        for muscle, trials in windows_dict.items():
            windows = trials[trial_idx]
            w = np.atleast_2d(windows)
            if w.ndim == 2:
                w = w[:, np.newaxis, :]  # reshape to (num_windows, 1, samples)

            if feature_list:
                feats = fe.extract_features(feature_list, w)
                df = pd.DataFrame({f"{k}_emg_{muscle}": np.ravel(v) for k,v in feats.items()})
            else:
                feats = fe.extract_feature_group(feature_group, w)
                df = pd.DataFrame({f"{k}_emg_{muscle}": np.ravel(v) for k,v in feats.items()})

            feature_dfs.append(df)
        
        combined_df = pd.concat(feature_dfs, axis=1)
        all_features.append(combined_df)

    return all_features