import numpy as np
import pandas as pd
import ruptures as rpt
from emg_utils import plot_emg_signal

def combine_multiple_features_lists(*features_dict_lists):
    """
    Combine multiple lists of feature dictionaries by flattening, merging,
    and concatenating them into a single DataFrame with window indices.
    """
    dfs = []

    n_lists = len(features_dict_lists)  # Number of feature lists received

    for dicts_at_idx in zip(*features_dict_lists):  # Iterate over dicts by window index
        combined_features = {}

        for feature_dict in dicts_at_idx:
            # Flatten arrays and round values for numerical stability
            flat = {k: np.round(np.array(v).ravel(), 10) for k, v in feature_dict.items()}
            combined_features.update(flat)  # Merge features from all dicts at this index

        df = pd.DataFrame(combined_features)  # Create DataFrame for combined features
        df.insert(0, 'window_idx', range(len(df)))  # Add window index column

        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)  # Concatenate all window DataFrames
    return combined_df


def detect_segments(normalized_signals, time_vectors, intensity_level, n_lifts, plot=True):
    """
    Detect breakpoints on segmented EMG signals based on intensity thresholds.
    Divides the signal into n_lifts parts, predicts a fixed number of breakpoints per segment,
    then selects the two breakpoints closest to the segment's left and right edges.
    """
    
    thresholds = {"light": 0.02, "medium": 0.03, "heavy": 0.05}
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
            plot_emg_signal(signal, time, title=f"Signal {idx+1} - Change Point Detection", ylabel='%MVC', breakpoints=all_bkps)

    return all_breakpoints_list



def assign_emg_labels(all_breakpoints_list, windowed_signals, sampling_rates, window_duration, overlap, intensity_level):
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
    segment_labels = ['no weight', intensity_level, 'no weight', intensity_level, 'no weight', intensity_level, 'no weight']
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