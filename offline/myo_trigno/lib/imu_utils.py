import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler
from libemg.feature_extractor import FeatureExtractor

def plot_imu_signals(muscle_imu, imu_time, muscle_names, axes, y_labels=None, file_label=None, flag=False, breakpoints_times=None):
    """
    Plot accelerometer and gyroscope Trigno data for each muscle.
    """
    if flag == True:
        time = imu_time[0]
        
    if y_labels is None:
        y_labels = {
            'acc': 'Acceleration [G]',
            'gyr': 'Angular velocity [deg/s]'
        }

    for muscle in muscle_names:
        # Compose suffix for title
        suffix = f" {file_label}" if file_label else "File #1"

        # Plot Accelerometer signals
        plt.figure(figsize=(12, 6))
        for axis in axes:
            channel = f"ACC_{axis}"
            signal = np.array(muscle_imu[muscle][channel]).flatten()
            if flag == False: time = imu_time[muscle][channel]
            plt.plot(time, signal, label=f"{axis}-axis")
        
        plt.title(f"Trigno {muscle} Accelerometer - {suffix}")
        plt.xlabel("Time [s]")
        plt.ylabel(y_labels.get('acc', 'Acceleration [G]'))
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        # Plot vertical lines for breakpoints if present
        if breakpoints_times is not None:
            for t in breakpoints_times:
                plt.axvline(x=t, color='r', linestyle='--')
                
        plt.show()

        # Plot Gyroscope signals
        plt.figure(figsize=(12, 6))
        for axis in axes:
            channel = f"GYR_{axis}"
            signal = np.array(muscle_imu[muscle][channel]).flatten()
            if flag == False: time = imu_time[muscle][channel]
            plt.plot(time, signal, label=f"{axis}-axis")
        
        plt.title(f"Trigno {muscle} Gyroscope - {suffix}")
        plt.xlabel("Time [s]")
        plt.ylabel(y_labels.get('gyr', 'Angular velocity [deg/s]'))
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if breakpoints_times is not None:
            for t in breakpoints_times:
                plt.axvline(x=t, color='r', linestyle='--')
                
        plt.show()

def detect_segments_imu(imu_time, muscle_imu_raw, muscle_names, axes, windows_idx, fs_list):
    """
    Segments Trigno IMU signals (accelerometer and gyroscope) based on given window indices.
    Returns segmented signals and corresponding timestamps, organized by muscle and channel.
    """
    # Initialize nested dictionaries
    imu_raw_cut = {}
    imu_times_cut = {}

    for muscle in muscle_names:
        imu_raw_cut[muscle] = {}
        imu_times_cut[muscle] = {}
        for sensor in ['ACC', 'GYR']:
            for axis in axes:
                channel = f"{sensor}_{axis}"
                imu_raw_cut[muscle][channel] = []
                imu_times_cut[muscle][channel] = []

    # Iterate through each acquisition
    for i in range(len(imu_time)):
        windows = windows_idx[i]
        fs = fs_list[i]
        time_vector = np.array(imu_time[i]).flatten()

        for muscle in muscle_names:
            for sensor in ['ACC', 'GYR']:
                for axis in axes:
                    channel = f"{sensor}_{axis}"
                    imu_signal = np.array(muscle_imu_raw[muscle][channel][i]).flatten()

                    # Segment signal and time for each window
                    for start_idx, end_idx in windows:
                        if end_idx <= len(time_vector):
                            signal_cut = imu_signal[start_idx:end_idx]
                            time_cut = time_vector[start_idx:end_idx]
                            imu_raw_cut[muscle][channel].append(signal_cut)
                            imu_times_cut[muscle][channel].append(time_cut)

    return imu_raw_cut, imu_times_cut

def imu_lowpass_filt(data, cutoff=5, fs=100, order=4):
    """
    Apply a zero-phase Butterworth low-pass filter to the input data.
    """
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)  # zero-lag filtering

def process_imu_segments(imu_raw, imu_time, muscle_names, axes, cutoff=5, scale_range=(-1, 1)):
    """
    Process Trigno IMU signals: low-pass filtering and min-max normalization.
    """
    muscle_imu_processed = {}

    for muscle in muscle_names:
        muscle_imu_processed[muscle] = {}

        for sensor in ['ACC', 'GYR']:
            for axis in axes:
                channel = f"{sensor}_{axis}"
                muscle_imu_processed[muscle][channel] = {}

                all_filtered_segments = []
                segment_lengths = []

                for i in range(len(imu_time[muscle][channel])):
                    time = imu_time[muscle][channel][i]
                    raw_data = imu_raw[muscle][channel][i]
                    fs = 1 / np.mean(np.diff(time))
                    filtered = imu_lowpass_filt(raw_data, cutoff=cutoff, fs=fs)

                    all_filtered_segments.append(filtered.reshape(-1, 1))
                    segment_lengths.append(len(filtered))

                # Concatenate all segments and normalize
                all_filtered_concat = np.vstack(all_filtered_segments)
                scaler = MinMaxScaler(feature_range=scale_range)
                all_scaled_concat = scaler.fit_transform(all_filtered_concat)

                # Split back into segments
                start_idx = 0
                for i, seg_len in enumerate(segment_lengths):
                    end_idx = start_idx + seg_len
                    scaled_segment = all_scaled_concat[start_idx:end_idx].flatten()
                    muscle_imu_processed[muscle][channel][i] = scaled_segment
                    start_idx = end_idx

    return muscle_imu_processed

def extract_imu_windows(processed_imu_dict, imu_times_dict, window_duration=0.2, overlap=0.5):
    """
    Extracts Trigno overlapping windows from filtered and normalized IMU signals,
    based on fixed time intervals.
    """
    segmented_imu = {}
    segmented_times = {}

    for muscle in processed_imu_dict:
        segmented_imu[muscle] = {}
        segmented_times[muscle] = {}

        for channel in processed_imu_dict[muscle]:
            segmented_imu[muscle][channel] = []
            segmented_times[muscle][channel] = []

            for signal, time in zip(processed_imu_dict[muscle][channel].values(),
                                    imu_times_dict[muscle][channel]):
                
                time = np.array(time)
                signal = np.array(signal)
                start_time = time[0]
                end_time = time[-1]
                step = window_duration * (1 - overlap)

                current_start = start_time
                while current_start + window_duration <= end_time:
                    current_end = current_start + window_duration
                    indices = np.where((time >= current_start) & (time < current_end))[0]

                    if len(indices) > 0:
                        segmented_imu[muscle][channel].append(signal[indices])
                        segmented_times[muscle][channel].append(time[indices])

                    current_start += step

    return segmented_imu, segmented_times

def extract_imu_features(windows_dict, feature_list=None, feature_group=None):
    """
    Extracts Trigno features from IMU windows. Uses libemg FeatureExtractor. 
    """
    fe = FeatureExtractor()
    
    muscles = list(windows_dict.keys())
    channels = list(next(iter(windows_dict[muscles[0]].keys())))
    trial_count = len(next(iter(windows_dict[muscles[0]].values())))
    
    all_features = []

    for trial_idx in range(trial_count):
        feature_dfs = []
        
        for muscle in windows_dict:
            for channel in windows_dict[muscle]:
                windows = windows_dict[muscle][channel][trial_idx]
                
                w = np.atleast_2d(windows)
                if w.ndim == 2:
                    # Shape: (num_windows, samples)
                    w = w[:, np.newaxis, :]  
                
                if feature_list:
                    feats = fe.extract_features(feature_list, w)
                    df = pd.DataFrame({f"{k}_{muscle}_{channel}": np.ravel(v) for k, v in feats.items()})
                else:
                    feats = fe.extract_feature_group(feature_group, w)
                    df = pd.DataFrame({f"{k}_{muscle}_{channel}": np.ravel(v) for k, v in feats.items()})
                
                feature_dfs.append(df)

        combined_df = pd.concat(feature_dfs, axis=1)
        all_features.append(combined_df)

    return all_features

def compute_vm_features(segmented_imu_trigno):
    """
    Compute VM (Vector Magnitude) for ACC and GYR for both biceps and triceps
    based on the corrected segmented_imu_trigno structure.
    """
    vm_features = []
    num_trials = len(segmented_imu_trigno['biceps']['ACC_X'])

    for trial_idx in range(num_trials):
        trial_dict = {}

        for muscle in ['biceps', 'triceps']:
            acc_x = segmented_imu_trigno[muscle]['ACC_X'][trial_idx]
            acc_y = segmented_imu_trigno[muscle]['ACC_Y'][trial_idx]
            acc_z = segmented_imu_trigno[muscle]['ACC_Z'][trial_idx]
            acc_vm = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
            trial_dict[f'VM_{muscle}_ACC_all'] = np.mean(acc_vm)

            gyr_x = segmented_imu_trigno[muscle]['GYR_X'][trial_idx]
            gyr_y = segmented_imu_trigno[muscle]['GYR_Y'][trial_idx]
            gyr_z = segmented_imu_trigno[muscle]['GYR_Z'][trial_idx]
            gyr_vm = np.sqrt(gyr_x**2 + gyr_y**2 + gyr_z**2)
            trial_dict[f'VM_{muscle}_GYR_all'] = np.mean(gyr_vm)

        vm_features.append(pd.DataFrame(trial_dict, index=[0]))

    return vm_features

def align_imu_signals_trigno(imu_time_trigno, muscle_imu_trigno, muscle_names, axes, diff):
    """
    Aligns Trigno IMU signals with Myo based on timing offsets
    """
    for i in range(len(imu_time_trigno)):
        # Shift time vector
        mask = (imu_time_trigno[i] - diff[i]) >= 0
        imu_time_trigno[i] = imu_time_trigno[i][mask] - diff[i]

        for muscle in muscle_names:
            for sensor in ['ACC', 'GYR']:
                for axis in axes:
                    channel = f"{sensor}_{axis}"
                    muscle_imu_trigno[muscle][channel][i] = muscle_imu_trigno[muscle][channel][i][mask]

    return imu_time_trigno, muscle_imu_trigno

def plot_imu_myo(muscle_imu_myo, imu_time_myo):
    """
    Plot each Myo IMU signal in a separate figure.
    """
    imu_signals = [
        'ACC_X', 'ACC_Y', 'ACC_Z', 
        'GYR_X','GYR_Y', 'GYR_Z',
        'roll', 'pitch', 'yaw'
    ]

    ylabel_map = {
        'ACC_X': 'Acceleration [g]',
        'ACC_Y': 'Acceleration [g]',
        'ACC_Z': 'Acceleration [g]',
        'GYR_X': 'Angular Velocity [deg/s]',
        'GYR_Y': 'Angular Velocity [deg/s]',
        'GYR_Z': 'Angular Velocity [deg/s]',
        'roll': 'Orientation [rad]',
        'pitch': 'Orientation [rad]',
        'yaw': 'Orientation [rad]',
    }

    num_recordings = len(imu_time_myo)

    for signal in imu_signals:
        for rec_idx in range(num_recordings):
            signal_data = muscle_imu_myo[signal][rec_idx]
            time_data = imu_time_myo[rec_idx]

            # Convert string values (e.g. '-nan(ind)') to np.nan
            if signal_data.dtype == object:
                signal_data = np.array([
                    float(x) if str(x).replace('.', '', 1).replace('-', '', 1).isdigit() or 'e' in str(x).lower()
                    else np.nan
                    for x in signal_data
                ])

            plt.figure(figsize=(12, 6))
            plt.plot(time_data, signal_data, label=f'{signal}')
            plt.title(f'Myo {signal} - File #{rec_idx + 1}')
            plt.xlabel('Time [s]')
            plt.ylabel(ylabel_map[signal])
            plt.grid(True)
            plt.tight_layout()
            plt.show()

def plot_imu_segmented_myo(imu_raw_cut, imu_times_cut, flag=True, breakpoints_times=None):
    """
    Plot segmented IMU signals from Myo armband.
    """
    ylabel_map = {
        'ACC_X': 'Acceleration [g]',
        'ACC_Y': 'Acceleration [g]',
        'ACC_Z': 'Acceleration [g]',
        'GYR_X': 'Angular Velocity [deg/s]',
        'GYR_Y': 'Angular Velocity [deg/s]',
        'GYR_Z': 'Angular Velocity [deg/s]',
        'roll': 'Orientation [rad]',
        'pitch': 'Orientation [rad]',
        'yaw': 'Orientation [rad]'
    }

    groups = {
        'Acceleration': ['ACC_X', 'ACC_Y', 'ACC_Z'],
        'Gyroscope': ['GYR_X', 'GYR_Y', 'GYR_Z']
    }

    first_acc_signal = groups['Acceleration'][0]
    n_segments = len(imu_raw_cut[first_acc_signal])

    # Loop through each segment
    for segment_idx in range(n_segments):
        for group_name, signals in groups.items():
            plt.figure(figsize=(12, 6))
            any_data = False

            for signal in signals:
                data_segments = imu_raw_cut.get(signal, [])
                time_segments = imu_times_cut.get(signal, [])

                if segment_idx < len(data_segments) and segment_idx < len(time_segments):
                    segment = data_segments[segment_idx]
                    time_segment = time_segments[segment_idx]

                    plt.plot(time_segment, segment, label=signal)
                    any_data = True

            if any_data:
                # Add breakpoint lines if available
                if breakpoints_times is not None:
                    for t in breakpoints_times[segment_idx]:
                        plt.axvline(x=t, color='r', linestyle='--')

                plt.title(f'Myo - {group_name} - Segment #{segment_idx + 1}')
                plt.xlabel('Time [s]')
                if flag:
                    plt.ylabel(ylabel_map.get(signals[0], 'Signal Value'))
                else:
                    plt.ylabel('Normalized Value [a.u.]')  # a.u. = arbitrary units
                plt.legend(loc='upper right', fontsize='small')
                plt.grid(True)
                plt.tight_layout()
                plt.show()
                
def detect_segments_imu_myo(imu_time_myo, imu_dict_myo, windows_idx, fs_list):
    """
    Segments Myo IMU signals based on provided window indices.
    """
    imu_signals = [
        'ACC_X', 'ACC_Y', 'ACC_Z', 
        'GYR_X','GYR_Y', 'GYR_Z',
        'roll', 'pitch', 'yaw'
    ]

    imu_raw_cut = {sig: [] for sig in imu_signals}
    imu_times_cut = {sig: [] for sig in imu_signals}

    for i in range(len(imu_time_myo)):
        time_vector = np.array(imu_time_myo[i]).flatten()
        windows = windows_idx[i]

        for signal in imu_signals:
            signal_data = np.array(imu_dict_myo[signal][i]).flatten()

            # Convert from object or bad strings if needed
            if signal_data.dtype == object:
                signal_data = np.array([
                    float(x) if str(x).replace('.', '', 1).replace('-', '', 1).isdigit() or 'e' in str(x).lower()
                    else np.nan
                    for x in signal_data
                ])

            for start_idx, end_idx in windows:
                if end_idx <= len(signal_data):
                    segment = signal_data[start_idx:end_idx]
                    time_seg = time_vector[start_idx:end_idx]

                    imu_raw_cut[signal].append(segment)
                    imu_times_cut[signal].append(time_seg)

    return imu_raw_cut, imu_times_cut

def process_imu_myo(imu_raw_cut, imu_times_cut, cutoff=5, scale_range=(-1, 1)):
    """
    Process IMU Myo segmented signals (flat structure without muscle level).
    """
    imu_processed = {}

    for channel in imu_raw_cut.keys():
        imu_processed[channel] = {}

        all_filtered_segments = []
        segment_lengths = []

        for i in range(len(imu_raw_cut[channel])):
            time = imu_times_cut[channel][i]
            raw_data = imu_raw_cut[channel][i]

            if np.any(np.isnan(raw_data)):
                nans = np.isnan(raw_data)
                not_nans = ~nans
                raw_data[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(not_nans), raw_data[not_nans])
                
            fs = 1 / np.mean(np.diff(time))


            filtered = imu_lowpass_filt(raw_data, cutoff=cutoff, fs=fs)
            all_filtered_segments.append(filtered.reshape(-1, 1))
            segment_lengths.append(len(filtered))

        all_filtered_concat = np.vstack(all_filtered_segments)
        scaler = MinMaxScaler(feature_range=scale_range)
        all_scaled_concat = scaler.fit_transform(all_filtered_concat)

        start_idx = 0
        for i, seg_len in enumerate(segment_lengths):
            end_idx = start_idx + seg_len
            scaled_segment = all_scaled_concat[start_idx:end_idx].flatten()
            imu_processed[channel][i] = scaled_segment
            start_idx = end_idx

    return imu_processed

def extract_imu_windows_myo(processed_imu_dict, imu_times_dict, window_duration=0.2, overlap=0.5):
    """
    Extract overlapping windows from filtered and normalized IMU signals.
    """
    segmented_imu = {}
    segmented_times = {}

    for channel in processed_imu_dict:
        segmented_imu[channel] = []
        segmented_times[channel] = []

        for segment_idx in range(len(processed_imu_dict[channel])):
            signal = np.array(processed_imu_dict[channel][segment_idx])
            time = np.array(imu_times_dict[channel][segment_idx])

            start_time = time[0]
            end_time = time[-1]
            step = window_duration * (1 - overlap)

            current_start = start_time
            while current_start + window_duration <= end_time:
                current_end = current_start + window_duration
                indices = np.where((time >= current_start) & (time < current_end))[0]

                if len(indices) > 0:
                    segmented_imu[channel].append(signal[indices])
                    segmented_times[channel].append(time[indices])

                current_start += step

    return segmented_imu, segmented_times

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