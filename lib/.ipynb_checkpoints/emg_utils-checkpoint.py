import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
from scipy.signal import butter, lfilter
from libemg.feature_extractor import FeatureExtractor

def plot_emg_signal(signal, time, title="Signal", xlabel="Time [s]", ylabel="mV", breakpoints=None):
    """
    Plots an EMG signal and optionally marks breakpoints.
    """
    plt.plot(time, signal)
    if breakpoints is not None:
        for i, bp in enumerate(breakpoints):
            bp_time = time[bp]
            plt.axvline(x=bp_time, color='red', linestyle='--', label='Change point' if i == 0 else "")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    if breakpoints is not None and len(breakpoints) > 0:
        plt.legend()
    plt.show()

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

def MVC_normalization(signal, muscle_name, mvc_csv_path):
    """
    Normalizes EMG signal by MVC value from CSV file.
    """
    df = pd.read_csv(mvc_csv_path, index_col=0)
    
    if muscle_name not in df.index:
        raise ValueError(f"Muscle name '{muscle_name}' not found in MVC CSV.")
    
    mvc = df.loc[muscle_name, 'MVC']
    
    if mvc == 0:
        raise ValueError(f"MVC value for muscle '{muscle_name}' is zero, cannot normalize.")
    
    normalized_emg = signal / mvc
    
    return normalized_emg

def emg_filters(muscle_emg_raw, emg_time, fs_list, muscle_names=None, mvc_csv_path='../../data/mvc_values/trigno/S01_05_14/combined_dataset.csv'):
    """
    Filters, rectifies, smooths, and optionally normalizes EMG signals.
    """
    # Use keys from input if muscle_names is None (when dealing with MVC signal)
    if muscle_names is None:
        muscle_names = list(muscle_emg_raw.keys())
        apply_normalization = False
    else:
        apply_normalization = True

    # Initialize output dictionaries
    muscle_emg_filtered = {muscle: [] for muscle in muscle_names}
    muscle_emg_rectified = {muscle: [] for muscle in muscle_names}
    muscle_emg_smoothed = {muscle: [] for muscle in muscle_names}
    muscle_emg_normalized = {muscle: [] for muscle in muscle_names} if apply_normalization else None

    for i, (time_signal, fs) in enumerate(zip(emg_time, fs_list)):
        for muscle in muscle_names:
            emg_signal = muscle_emg_raw[muscle][i]

            hf = fs / 2 - 1 if fs < 900 else 450
            filtered_emg = bandpass_filter(emg_signal, fs, high_freq=hf)
            rectified_emg = rectification(filtered_emg)
            smoothed_emg = RMS_moving(rectified_emg, fs)
            muscle_emg_filtered[muscle].append(filtered_emg)
            muscle_emg_rectified[muscle].append(rectified_emg)
            muscle_emg_smoothed[muscle].append(smoothed_emg)

            if apply_normalization:
                normalized_emg = MVC_normalization(smoothed_emg, muscle, mvc_csv_path)
                muscle_emg_normalized[muscle].append(normalized_emg)

    if apply_normalization:
        return muscle_emg_filtered, muscle_emg_rectified, muscle_emg_smoothed, muscle_emg_normalized
    else:
        return muscle_emg_filtered, muscle_emg_rectified, muscle_emg_smoothed

def compute_MVC(emg_signal, fs, window=0.5):
    """
    Computes maximum mean amplitude in sliding windows.
    """
    window_samples = int(window * fs)
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

def extract_emg_windows(normalized_emg_dict, filtered_emg_dict, emg_time_list, fs_list, window_duration=0.2, overlap=0.5):
    """
    Extracts overlapping windows from EMG signals and times.
    """
    normalized_windows_all = {muscle: [] for muscle in normalized_emg_dict}
    filtered_windows_all = {muscle: [] for muscle in filtered_emg_dict}
    time_windows_all = []

    for i, (time_signal, fs) in enumerate(zip(emg_time_list, fs_list)):
        step_duration = window_duration * (1 - overlap)
        window_size = int(window_duration * fs)
        step_size = int(step_duration * fs)

        time_windows = []
        for start in range(0, len(time_signal) - window_size + 1, step_size):
            end = start + window_size
            time_windows.append(time_signal[start:end])
        time_windows_all.append(time_windows)

        for muscle in normalized_emg_dict:
            norm_emg = normalized_emg_dict[muscle][i]
            filt_emg = filtered_emg_dict[muscle][i]

            norm_windows = [norm_emg[start:start+window_size] for start in range(0, len(norm_emg) - window_size + 1, step_size)]
            filt_windows = [filt_emg[start:start+window_size] for start in range(0, len(filt_emg) - window_size + 1, step_size)]

            normalized_windows_all[muscle].append(norm_windows)
            filtered_windows_all[muscle].append(filt_windows)

    return normalized_windows_all, filtered_windows_all, time_windows_all

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
                df = pd.DataFrame({f"{k}_{muscle}": np.ravel(v) for k,v in feats.items()})
            else:
                feats = fe.extract_feature_group(feature_group, w)
                df = pd.DataFrame({f"{k}_{muscle}": np.ravel(v) for k,v in feats.items()})

            feature_dfs.append(df)
        
        combined_df = pd.concat(feature_dfs, axis=1)
        all_features.append(combined_df)

    return all_features