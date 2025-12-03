import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler
from libemg.feature_extractor import FeatureExtractor

def imu_lowpass_filt(data, cutoff=5, fs=100, order=4):
    """
    Apply a zero-phase Butterworth low-pass filter to the input data.
    """
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)  # zero-lag filtering

def extract_imu_features_myo(windows_dict, feature_list=None, feature_group=None):
    """
    Extract Myo features from IMU windows.
    """
    fe = FeatureExtractor()

    channels = [ch for ch in windows_dict.keys() if all(x not in ch.upper() for x in ['ROLL', 'PITCH', 'YAW'])]
    trial_count = len(windows_dict[channels[0]])

    all_features = []

    for trial_idx in range(trial_count):
        feature_dfs = []

        for channel in channels:
            window = windows_dict[channel][trial_idx]  
            w = np.atleast_2d(window)
            if w.ndim == 2:
                w = w[np.newaxis, :, :]  

            if feature_list:
                feats = fe.extract_features(feature_list, w)
                df = pd.DataFrame({f"{k}_myo_{channel.upper()}": np.ravel(v) for k, v in feats.items()})
            else:
                feats = fe.extract_feature_group(feature_group, w)
                df = pd.DataFrame({f"{k}_myo_{channel.upper()}": np.ravel(v) for k, v in feats.items()})

            feature_dfs.append(df)

        combined_df = pd.concat(feature_dfs, axis=1)
        all_features.append(combined_df)

    return all_features

def compute_vm_features_myo(segmented_imu_dict):
    """
    Compute Myo VM (Vector Magnitude) for ACC and GYR.
    """
    vm_features = []
    channels = segmented_imu_dict.keys()
    trial_count = len(next(iter(segmented_imu_dict.values())))

    for trial_idx in range(trial_count):
        trial_dict = {}
        base_names = set(ch.split('_')[0] for ch in channels if all(x not in ch.upper() for x in ['ROLL', 'PITCH', 'YAW']))

        for base in base_names:
            try:
                x = segmented_imu_dict[f'{base}_X'][trial_idx]
                y = segmented_imu_dict[f'{base}_Y'][trial_idx]
                z = segmented_imu_dict[f'{base}_Z'][trial_idx]
            except KeyError:
                continue  

            vm = np.sqrt(x**2 + y**2 + z**2)
            trial_dict[f'VM_myo_{base.upper()}_all'] = np.mean(vm)

        vm_features.append(pd.DataFrame(trial_dict, index=[0]))

    return vm_features