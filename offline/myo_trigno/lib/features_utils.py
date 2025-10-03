import numpy as np
import pandas as pd
import ruptures as rpt
from emg_utils import plot_emg_signal

def combine_multiple_features_lists(*features_dict_lists):
    """
    Combine multiple lists of feature dictionaries by merging,
    and concatenating them into a single DataFrame with window indices.
    """
    dfs = []
    reordered_cols = []

    n_lists = len(features_dict_lists)  # Number of feature lists received

    for dicts_at_idx in zip(*features_dict_lists):  # Iterate over dicts by window index
        combined_features = {}

        for feature_dict in dicts_at_idx:
            flat = {k: np.array(v) for k, v in feature_dict.items()}
            combined_features.update(flat)  # Merge features from all dicts at this index

        df = pd.DataFrame(combined_features)  # Create DataFrame for combined features
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)  # Concatenate all window DataFrames

    # Extract unique muscle suffixes from columns names
    suffixes = sorted(set('_'.join(col.split('_')[-3:]) for col in combined_df))

    for suffix in suffixes:
        # Add all columns ending with the suffix
        suffix_cols = [col for col in combined_df if col.endswith(f"_{suffix}")]
        reordered_cols.extend(sorted(suffix_cols))  # Optional: sort alphabetically inside group

    combined_df = combined_df[reordered_cols]

    return combined_df

def detect_segments(normalized_signals, time_vectors, intensity_level, n_lifts, n=None, plot=True):
    """
    Detect breakpoints on segmented EMG signals based on intensity thresholds.
    Divides the signal into n_lifts parts, predicts a fixed number of breakpoints per segment,
    then selects the two breakpoints closest to the segment's left and right edges.
    """  
    thresholds = {"light": 0.02, "medium": 0.03, "heavy": 0.05, "all": 0.02}
    threshold = thresholds[intensity_level] # Remove final part of the signal below threshold

    n_bkps_per_segment = 5  # Number of breakpoints predicted per segment
    
    all_breakpoints_list = []

    for idx, (signal, time) in enumerate(zip(normalized_signals, time_vectors)):
        active_indices = np.where(signal > threshold)[0]
        valid_length = active_indices[-1] + 1 if len(active_indices) > 0 else 0

        segment_points = [i * valid_length // n_lifts for i in range(n_lifts)] + [valid_length]

        selected_breakpoints = []

        for i in range(n_lifts):
            start = segment_points[i]
            end = segment_points[i + 1]
            segment = signal[start:end]

            if len(segment) == 0:
                # Empty segment, skip
                continue

            # Predict breakpoints in segment (excluding the last breakpoint which is the segment end)
            bkps = rpt.Binseg(model="l2").fit(segment).predict(n_bkps=n_bkps_per_segment)[:-1]
            bkps = [b + start for b in bkps]  # Adjust to full signal indices

            if not bkps:
                # No breakpoints found, skip segment
                continue

            # Find breakpoint closest to left edge
            left_bkp = min(bkps, key=lambda b: abs(b - start))
            # Find breakpoint closest to right edge
            right_bkp = min(bkps, key=lambda b: abs(b - end))

            selected_breakpoints.extend([left_bkp, right_bkp])

        # Sort and remove duplicates if any
        all_bkps = sorted(set(selected_breakpoints))
        all_breakpoints_list.append(all_bkps)

        if plot:
            if n==None:
                n=idx
            plot_emg_signal(signal, time, title=f"Signal {n+1} - Change Point Detection", ylabel='bit', breakpoints=all_bkps)

    return all_breakpoints_list

def assign_emg_labels(all_breakpoints_list, windowed_signals, sampling_rates, window_duration, overlap, intensity_level, n_lifts):
    """
    Assigns labels to windowed EMG data based on detected breakpoints.
    """
    def assign_labels_by_indices(total_windows, segment_start_indices, segment_labels):
        labels_assigned = [''] * total_windows
        for i in range(len(segment_start_indices)):
            start = segment_start_indices[i]
            end = segment_start_indices[i + 1] if i + 1 < len(segment_start_indices) else total_windows
            for w in range(start, end):
                labels_assigned[w] = segment_labels[i]
        return labels_assigned

    # Labels alternate between rest and activity based on intensity
    segment_labels = ['no weight' if i % 2 == 0 else intensity_level for i in range(2 * n_lifts + 1)]

    all_assigned_labels = []

    for i, breakpoints in enumerate(all_breakpoints_list):
        fs = sampling_rates[i]
        win_size_samples = int(window_duration * fs)
        step_size = int(win_size_samples * (1 - overlap))
        total_windows = len(windowed_signals[i])  # Number of windows for this signal
        
        # Convert breakpoint sample indices to window indices
        window_starts = [0] + breakpoints[:]
        window_starts_idx = [idx // step_size for idx in window_starts if idx // step_size < total_windows]
        
        assigned = assign_labels_by_indices(total_windows, window_starts_idx, segment_labels)
        all_assigned_labels.append(assigned)

    # Flatten all labels into a single list
    flattened_labels = [label for sublist in all_assigned_labels for label in sublist]
    return flattened_labels

def assign_emg_labels_v2(all_breakpoints, n_windows_per_file, fs_list, w_d, ov, intensity_level, n_lifts): 
    """
    Assigns labels to windowed EMG data based on detected breakpoints.
    """
    all_labels = []
    for i, breakpoints in enumerate(all_breakpoints):
        segment_labels = ['no weight' if j % 2 == 0 else intensity_level[i % len(intensity_level)] for j in range(2 * n_lifts + 1)]
        fs = fs_list[i]
        win_size_samples = int(w_d * fs)
        step_size = int(win_size_samples * (1 - ov))
        
        # breakpoint in window indices
        window_bkps = [0] + [b // step_size for b in breakpoints] + [n_windows_per_file]
        
        labels_for_file = [''] * n_windows_per_file
        
        for j in range(len(window_bkps) - 1):
            start_w = window_bkps[j]
            end_w = window_bkps[j+1]
            label = segment_labels[j] if j < len(segment_labels) else 'unknown'
            for w in range(start_w, end_w):
                labels_for_file[w] = label
        
        all_labels.extend(labels_for_file)
    
    return all_labels

def detect_segments_v2(emg_time, muscle_emg_normalized, muscle_emg_filtered,
                             channel_names, channel_ref, windows_idx,
                             intensity_level, n_lifts, fs_list, flag = False):
    """
    Detect breakpoints on segmented EMG signals based on intensity thresholds.
    Return also segmented EMG signals.
    """
    # Initialize dictionaries for each channel
    emg_normalized_cut = {ch: [] for ch in channel_names}
    emg_filtered_cut = {ch: [] for ch in channel_names}
    emg_times_cut = {ch: [] for ch in channel_names}
    all_breakpoints = []
    j = 0  # global segment counter

    # Iterate through each acquisition
    for i in range(len(emg_time)):
        windows = windows_idx[i]
        n = i
        fs = fs_list[i]
        # Iterate through each channel
        for channel in channel_names:
            time_vector = np.array(emg_time[i]).flatten()
            emg_signal = np.array(muscle_emg_normalized[channel][i]).flatten()
            emg_signal2 = np.array(muscle_emg_filtered[channel][i]).flatten()

            # Iterate through each window (in sample indices)
            for start_idx, end_idx in windows:
                if end_idx <= len(time_vector):
                    # Extract segment of signal and time
                    signal_cut = emg_signal[start_idx:end_idx]
                    signal2_cut = emg_signal2[start_idx:end_idx]
                    time_cut = time_vector[start_idx:end_idx]

                    # Append to corresponding dictionaries
                    emg_normalized_cut[channel].append(signal_cut)
                    emg_filtered_cut[channel].append(signal2_cut)
                    emg_times_cut[channel].append(time_cut)

                    # Run segmentation only for reference channel
                    if channel == channel_ref and flag == True:
                        # Choose intensity value cyclically if it's a list
                        intensity = intensity_level[j % len(intensity_level)] if isinstance(intensity_level, list) else intensity_level

                        # Segment the signal
                        all_bkps_list = detect_segments(
                            [signal_cut], [time_cut], intensity, n_lifts, n, plot=flag)

                        all_breakpoints.append(all_bkps_list[0])
                        j += 1  # increment segment counter
                else:
                    print(f"Warning: Skipping window {start_idx/fs:.2f}-{end_idx/fs:.2f}s for channel {channel}, signal {i+1} (too short)")

    return emg_normalized_cut, emg_filtered_cut, emg_times_cut, all_breakpoints

def compute_offset_trigno_myo (emg_time_trigno, emg_time_myo, fs_list_trigno, windows_trigno, intensity_level, 
                              muscle_emg_smoothed_trigno, muscle_ref, breakpoints_myo, n_lifts):
    """
    Compute the average time offset between Trigno and Myo EMG systems for synchronization.
    """
    diff = []

    for j in range(len(emg_time_trigno)):
        fs = fs_list_trigno[j]
        all_diffs = []

        for i, (start_sec, end_sec) in enumerate(windows_trigno):
            intensity = intensity_level[i]
            start = int(fs * start_sec)
            end = int(fs * end_sec)

            # Extract EMG signal and time window for the current repetition
            emg_signals = [trial[start:end] for trial in muscle_emg_smoothed_trigno[muscle_ref]]
            time_vectors = [t[start:end] for t in emg_time_trigno]

            # Detect segments (e.g., lift start and end times) in Trigno signals
            bkps_trigno = detect_segments(emg_signals, time_vectors, intensity, n_lifts, plot=False)

            # Get corresponding segment start times from Myo and Trigno
            st_myo = emg_time_myo[0][breakpoints_myo[i]]
            st_trigno = emg_time_trigno[0][bkps_trigno[0]]
            diffs = st_trigno - st_myo +5

            all_diffs.append(diffs[::2].mean()) # Store mean of every segment start

        diff_j = np.mean(all_diffs)
        diff.append(diff_j)
        print(f"\nOffset #{j+1} Trigno - Myo: {diff_j:.4f} s")

    return diff