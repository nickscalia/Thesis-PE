# Nicolas Scalia
# Payload Estimation Master Thesis

## LIBRARIES
import time
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from libemg.streamers import myo_streamer
from libemg.data_handler import OnlineDataHandler
from joblib import load
sys.path.append('../../../lib')
from emg_utils import (bandpass_filter,rectification, RMS_moving, MVC_normalization,
                        extract_emg_features)


# FUNCTIONS
def remove_overlap(prev, curr, tol=1):
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

def main():
    
    # External Variables    
    mvc = '../../../data/mvc_values/myo/S01_05_26/combined_dataset.csv'
    model_path = r"C:\Users\nicol\Thesis\my_codes\data\models\myo\S01_05_26\GB\set_5.pkl"
    scaler_path = r"C:\Users\nicol\Thesis\my_codes\data\models\myo\S01_05_26\scaler_5.pkl"
    model = load(model_path)
    scaler = load(scaler_path)
    
    # Channels of interest
    channel_names = [
        'channel_1', 'channel_2', 'channel_3', 'channel_4',
        'channel_5', 'channel_6', 'channel_7', 'channel_8']
    fs = 200  # Myo nominal frequency in Hz
    window_size = 600  # Local buffer (3 seconds)
    window_feature_size = 40 # Dimension of window size onto which the feature is computed
    feature_calc_interval = 50 #????
    tolerance = 0 # Tolerance for buffer overlap 
    
    # Extracted Features Definition
    """features_list_norm = ['MAV', 'WL', 'VAR', 'SAMPEN', 'WAMP']  # Features on normalized signal
    features_list_filt = ['ZC', 'SSC', 'KURT', 'SKEW']  # Features on filtered signal
    features_list_freq = ['MNF']  # Frequency-domain features"""
    features_list_norm = ['MAV', 'WL']  # Features on normalized signal
    features_list_filt = ['ZC','SSC']  # Features on filtered signal 
    # Create list of ordered features (same order as scaler and model)
    selected_feature_names = []
    for ch in sorted(channel_names):  
        feats = sorted(features_list_norm + features_list_filt)
        selected_feature_names.extend([f"{feat}_{ch}" for feat in feats])
    
    # Initialize buffer variables
    emg_ch = np.zeros((window_size, 8))       
    emg_norm = np.zeros((window_size, 8))
    emg_filt = np.zeros((window_size, 8))
    previous_block = np.array([])
    sample_counter = 0

    # Live Plot Settings
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=1, label='EMG Channel 6')
    ax.set_title("Live EMG Channel 6 Normalized")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(0, 1.5)
    ax.legend()

    # Connect to Myo Armband
    streamer, shared_memory = myo_streamer(emg=True, imu=True, filtered=False)
    odh = OnlineDataHandler(shared_memory_items=shared_memory)
    odh.reset() #?????
    
    try:
        while True:
            # Take last 20 unfiltered samples from shared buffer
            data, _ = odh.get_data(N=20, filter=False) 

            if 'emg' in data and len(data['emg']) > 0:
                # Take only emg samples
                curr = np.array(data['emg']) 
                # Check overlap with previous iteration
                curr_ch6 = curr[:, 6]
                curr_new_ch6 = remove_overlap(previous_block, curr_ch6, tol=tolerance)
                if curr_new_ch6.size == 0:
                    continue
                previous_block = curr_ch6.copy()
                n = len(curr_new_ch6)
                curr_new = curr[-n:, :]
                sample_counter += n
                
                # Add only new data to the local buffer
                emg_ch = np.roll(emg_ch, -n, axis=0)
                emg_ch[-n:, :] = curr_new
                # Process emg_ch: bandpass, rectification, smoothing and normalization      
                for ch in range(8):
                    emg_filtered = bandpass_filter(emg_ch[:, ch], fs, high_freq=99)
                    emg_rectified = rectification(emg_filtered)
                    emg_rms = RMS_moving(emg_rectified, fs)
                    emg_norm_filtered = MVC_normalization(emg_rms, channel_names[ch], mvc)
        
                    # Store filtered and normalized values to local buffer
                    emg_filt[:, ch] = np.roll(emg_filt[:, ch], -n)
                    emg_filt[-n:, ch] = emg_filtered[-n:]
                    emg_norm[:, ch] = np.roll(emg_norm[:, ch], -n)
                    emg_norm[-n:, ch] = emg_norm_filtered[-n:]

                # Extract features and predict weight class
                if sample_counter > window_feature_size and (sample_counter % feature_calc_interval) < n: #???
                    # Create dictionaries
                    normalized_win = {channel_names[ch]: [emg_norm[-window_feature_size:, ch].reshape(1, -1)] for ch in range(8)}
                    filtered_win = {channel_names[ch]: [emg_filt[-window_feature_size:, ch].reshape(1, -1)] for ch in range(8)}
                
                    # Extract features
                    time_features_norm = extract_emg_features(normalized_win, features_list_norm)
                    time_features_filt = extract_emg_features(filtered_win, features_list_filt)
                    df_norm = time_features_norm[0]
                    df_filt = time_features_filt[0]
                    X_features = pd.concat([df_norm, df_filt], axis=1)
                    X_features = X_features[selected_feature_names]
                    
                    # Scale features
                    X_scaled = scaler.transform(X_features)
                    X_scaled = pd.DataFrame(X_scaled, columns=selected_feature_names)
                    
                    # Model Prediction
                    prediction = model.run(X_scaled)
                    print(prediction)
                 
                # Update Live Plot   
                t0 = (sample_counter - window_size) / fs
                t1 = sample_counter / fs
                time_vals = np.linspace(t0, t1, window_size)
                line.set_data(time_vals, emg_norm[:, 6])
                ax.set_xlim(t0, t1)
                plt.pause(0.001)            

            time.sleep(0.01) # Wait for new data on the buffer

    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()