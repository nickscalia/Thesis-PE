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
from sklearn.preprocessing import MinMaxScaler
sys.path.append('C:/Users/nicol/Thesis/my_codes/lib')
from emg_utils import bandpass_filter,rectification, RMS_moving, extract_emg_features
from imu_utils import imu_lowpass_filt, extract_imu_features_myo, compute_vm_features_myo

def MVC_normalization(signal, muscle_name, mvc_values):
    """
    Normalizes EMG signal by MVC value from a dictionary.
    """
    if muscle_name not in mvc_values:
        raise ValueError(f"Muscle name '{muscle_name}' not found in MVC dictionary.")   
    mvc = mvc_values[muscle_name]
    return signal / mvc

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
#    mvc = '../../../data/stream/mvc/test_1.csv'
#    pca = load('../../../data/stream/models/pca/pca_emg.pkl')
    model = load('../../../data/stream/models/emg_imu/52/GB/model.pkl')
    scaler_new = load('../../../data/stream/models/emg_imu/52/GB/scaler.pkl')    

    # Channels of interest
    channel_names = ['channel_1', 'channel_2', 'channel_3', 'channel_4',
                     'channel_5', 'channel_6', 'channel_7', 'channel_8']
    fs = 200  # Myo nominal frequency in Hz
    window_size = 600  # Local buffer (3 seconds)
    window_feature_size = 40 # Dimension of window size onto which the feature is computed
    tolerance = 1 # Tolerance for buffer overlap
    calibration = 15
    smooth_threshold = 3
    
    # Extracted Features Definition
    features_list_norm = ['MAV', 'WL']  # Features on normalized signal
    features_list_filt = ['ZC', 'SSC']  # Features on filtered signal 

    # Create list of ordered features (same order as scaler and model)
    selected_feature_names = []
    for ch in sorted(channel_names):  
        feats = sorted(features_list_norm + features_list_filt)
        selected_feature_names.extend([f"{feat}_emg_{ch}" for feat in feats])
        
    
    imu_scalers = []
    fs_imu = 50 
    features_list_imu = ['MAV', 'SKEW', 'VAR']
    imu_names = ['ACC_X', 'ACC_Y', 'ACC_Z',
                 'GYR_X', 'GYR_Y', 'GYR_Z']
 
    for i,ch in enumerate(sorted(imu_names)):
        feats = sorted(features_list_imu)
        selected_feature_names.extend([f"{feat}_myo_{ch}" for feat in feats])
        if i == 2:  
            selected_feature_names.append("VM_myo_ACC_all")           
    selected_feature_names.append("VM_myo_GYR_all")
    
    
    elapsed_times = [] # Stores processing times
    all_predictions = [] # Stores predicted classes
    all_probabilities = [] # Stores class probabilities
    last_printed_class = None # Tracks last printed class
    label_map = {0: "no weight", 1: "light", 2: "medium", 3: "heavy"} # Maps class IDs to labels

    # Initialize buffer variables
    emg_ch = np.zeros((window_size, 8))       
    emg_norm = np.zeros((window_size, 8))
    emg_filt = np.zeros((window_size, 8))
    emg_smoot = np.zeros((window_size, 8))
    mvc_values = {ch: 0 for ch in channel_names}
    previous_block_emg = np.array([])
    sample_counter = 0
    first_else = True

    
    imu_ch = np.zeros((window_size, 6))       
    imu_filt = np.zeros((window_size, 6))     
    imu_filt_calib = None
    
    
    # Live Plot Settings
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=1, label='EMG Channel 5')
    ax.set_title("Live Smoothed EMG Channel 5")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("bit")
    ax.set_ylim(0, 100)
    ax.legend()

    # Connect to Myo Armband
    streamer, shared_memory = myo_streamer(emg=True, imu=True, filtered=False)
    odh = OnlineDataHandler(shared_memory_items=shared_memory)
    odh.reset()
    
    try:
        total_start = time.time() 
        while True:
            start_time = time.time()
            data, _ = odh.get_data(N=10, filter=False)  # Take last 10 unfiltered samples from shared buffer
            elapsed = time.time() - total_start
            
            if 'emg' in data and len(data['emg']) > 0:
                curr_emg = np.array(data['emg']) # Take only emg samples
                
                # Check overlap with previous iteration
                curr_ch5 = curr_emg[:, 4]
                curr_new_ch5 = remove_overlap(previous_block_emg, curr_ch5, tol=tolerance)
                if curr_new_ch5.size == 0:
                    continue
                previous_block_emg = curr_ch5.copy()
                n = len(curr_new_ch5)
                
                # Add only new data to the local buffer
                curr_new = curr_emg[-n:, :]
                sample_counter += n
                emg_ch = np.roll(emg_ch, -n, axis=0) 
                emg_ch[-n:, :] = curr_new
                    
                if elapsed <= calibration:
                    # Preprocessing of emg signals      
                    for ch, ch_name in enumerate(channel_names):
                        emg_filtered = bandpass_filter(emg_ch[:, ch], fs, high_freq=fs/2 -1)
                        emg_rectified = rectification(emg_filtered)
                        emg_rms = RMS_moving(emg_rectified, fs)
                        
                        window_med = np.median(emg_rms)
                        smoothness = np.mean(np.abs(np.diff(emg_rms)))

                        if window_med > mvc_values[ch_name] and smoothness < smooth_threshold:
                            mvc_values[ch_name] = window_med
            
                        # Store smoothed values to local buffer
                        emg_smoot[:, ch] = np.roll(emg_smoot[:, ch], -n)
                        emg_smoot[-n:, ch] = emg_rms[-n:]
                    
                    # Update Live Plot   
                    t0 = (sample_counter - window_size) / fs
                    t1 = sample_counter / fs
                    time_vals = np.linspace(t0, t1, window_size)
                    line.set_data(time_vals, emg_smoot[:, 4])
                    ax.set_xlim(t0, t1) 
                    
                else: 
                    # Preprocessing of emg signals      
                    for ch in range(8):
                        emg_filtered = bandpass_filter(emg_ch[:, ch], fs, high_freq=fs/2 -1)
                        emg_rectified = rectification(emg_filtered)
                        emg_rms = RMS_moving(emg_rectified, fs)
                        emg_norm_filtered = MVC_normalization(emg_rms, channel_names[ch], mvc_values)
            
                        # Store filtered and normalized values to local buffer
                        emg_filt[:, ch] = np.roll(emg_filt[:, ch], -n)
                        emg_filt[-n:, ch] = emg_filtered[-n:]
                        emg_norm[:, ch] = np.roll(emg_norm[:, ch], -n)
                        emg_norm[-n:, ch] = emg_norm_filtered[-n:]
                        
                    if first_else == True:
                        print(mvc_values)
                        ax.set_title("Live Normalized EMG Channel 5")
                        ax.set_ylabel("EMG/MVC")
                        ax.set_ylim(0, 1.8)
                        first_else = False 
                        start_transition = time.time()
                        
                        for ch in range(6):
                            scaler_temp = MinMaxScaler(feature_range=(-1, 1))
                            scaler_temp.fit(imu_filt_calib[:, ch].reshape(-1, 1))
                            imu_scalers.append(scaler_temp)
                        
                                                
                    # Update Live Plot   
                    t0 = (sample_counter - window_size) / fs
                    t1 = sample_counter / fs
                    time_vals = np.linspace(t0, t1, window_size)
                    line.set_data(time_vals, emg_norm[:, 4])
                    ax.set_xlim(t0, t1)    
                        
                       
            if 'imu' in data and len(data['imu']) > 0:
                curr_imu = np.array(data['imu']) # Take only imu samples
                curr_imu = curr_imu[:, -6:]
                    
                # Add only new data to the local buffer
                curr_new_imu = curr_imu[-n:, :]
                imu_ch = np.roll(imu_ch, -n, axis=0)
                imu_ch[-n:, :] = curr_new_imu
                        
                if elapsed <= calibration:
                    imu_filt_temp = []
                    for ch in range(6):
                        imu_filtered = imu_lowpass_filt(imu_ch[:, ch], fs=fs_imu).reshape(-1, 1) 
                        # Store filtered values 
                        imu_filt_temp.append(imu_filtered)
                    
                    imu_filt_temp = np.hstack(imu_filt_temp)  # shape: (N, 6)
                    
                    if imu_filt_calib is None:
                        imu_filt_calib = imu_filt_temp
                    else:
                        imu_filt_calib = np.vstack((imu_filt_calib, imu_filt_temp))
                        
                else:  
                    for ch in range(6):
                        imu_filtered = imu_lowpass_filt(imu_ch[:, ch], fs=fs_imu).reshape(-1, 1) 
                        imu_scaled = imu_scalers[ch].transform(imu_filtered).ravel()
                        
                        # Store filtered values to local buffer
                        imu_filt[:, ch] = np.roll(imu_filt[:, ch], -n)
                        imu_filt[-n:, ch] = imu_scaled[-n:]

                    
                    
                    # Extract features and predict weight class
                    if sample_counter > window_feature_size:
                        # Create EMG dictionaries
                        normalized_win = {channel_names[ch]: [emg_norm[-window_feature_size:, ch].reshape(1, -1)] for ch in range(8)}
                        filtered_win = {channel_names[ch]: [emg_filt[-window_feature_size:, ch].reshape(1, -1)] for ch in range(8)}
                        
                        # Extract EMG features
                        time_features_norm = extract_emg_features(normalized_win, features_list_norm)
                        time_features_filt = extract_emg_features(filtered_win, features_list_filt)
                        df_norm = time_features_norm[0]
                        df_filt = time_features_filt[0]
                        X_features = pd.concat([df_norm, df_filt], axis=1)
                        
                        
                        imu_win = {imu_names[ch]: [imu_filt[-window_feature_size:, ch].reshape(1, -1)] for ch in range(6)}
                        features_imu_myo = extract_imu_features_myo(imu_win, features_list_imu)
                        vm_features_myo = compute_vm_features_myo(imu_win)
                        df_imu = features_imu_myo[0]
                        df_vm = vm_features_myo[0]
                        X_features = pd.concat([X_features, df_imu, df_vm], axis=1)
                        
                    
                        X_features = X_features[selected_feature_names] # Ordered features                   
                        X_scaled = scaler_new.transform(X_features) # Scale features
                        X_scaled = pd.DataFrame(X_scaled, columns=selected_feature_names)
                        """
                        X_scaled = pca.transform(X_scaled)
                        """
                                
                        # Model Prediction
                        prediction = model.run(X_scaled)
                        pred_class, pred_proba = prediction
                        pred_class = int(pred_class[0])
                        pred_proba = float(pred_proba[0])
                            
                        all_predictions.append(pred_class)
                        all_probabilities.append(pred_proba)
                        
                        
                        # Print prediction every 50 ms
                        if pred_proba > 0.50:
                            print(f"Prediction: {label_map[pred_class]} ({pred_proba:.2f} probability)")
                        
                        
                        if len(all_predictions) >= 2:    
                            last_class = all_predictions[-2] 
                            if last_class == 0 and pred_class !=0:
                                start_transition = time.time()
                            elif last_class != 0 and pred_class == 0:
                                start_transition = time.time()
                        
                        # This logic implements a stability check on predictions.
                        # It accepts a predicted class only if it has been repeated 3 times 
                        # consecutively with high confidence (>0.8).
                        # The code prints a new prediction only when there is a weight is lifted
                        # or released.
                        
                        if len(all_predictions) >= 3:
                            last_classes = all_predictions[-2:]
                            last_probas = all_probabilities[-2:]
                            if (all(c == last_classes[0] for c in last_classes) and
                                all(p > 0.8 for p in last_probas)):
                                stable_class = last_classes[0]
                                if stable_class == 0 and last_printed_class != 0:
                                    end_transition = time.time()
                                    transition = round(end_transition - start_transition, 3)
                                    print(f"Prediction: {label_map[stable_class]}, prediction time: {transition}")
                                    last_printed_class = 0
                        
                                elif stable_class != 0 and last_printed_class == 0:
                                    end_transition = time.time()
                                    transition = round(end_transition - start_transition, 3)
                                    print(f"Prediction: {label_map[stable_class]}, prediction time: {transition}")
                                    last_printed_class = stable_class
                                                               
            plt.pause(0.001) # Update the plot
                            
            end_time = time.time()
            elapsed_time = end_time - start_time
            elapsed_times.append(elapsed_time)
            if 0.050 - elapsed_time > 0:
                time.sleep(0.050 - elapsed_time) # Maintain 50 ms loop interval

    except KeyboardInterrupt:
        total_end = time.time()  # Global end time
        total_duration = total_end - total_start  # Calculate total elapsed time
        
        average_time = sum(elapsed_times) / len(elapsed_times)  # Calculate average processing time
        print(f"\nExecutions: {len(elapsed_times)}")
        print(f"Average processing time: {average_time:.3f} seconds")
        print(f"Total time: {total_duration:.3f} seconds")
        pass
    
    finally:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()