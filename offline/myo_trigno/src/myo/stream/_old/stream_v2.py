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
from sklearn.preprocessing import MinMaxScaler
from joblib import load
sys.path.append('../../../lib')
from emg_utils import (bandpass_filter,rectification, RMS_moving, MVC_normalization,
                        extract_emg_features)
from imu_utils import imu_lowpass_filt, extract_imu_features_myo, compute_vm_features_myo
from collections import deque, Counter



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
    mvc = '../../../data/mvc_values/global.csv'
    model_path = r"C:\Users\nicol\Thesis\my_codes\data\models\gb_emg.pkl"
    scaler_path = r"C:\Users\nicol\Thesis\my_codes\data\models\scaler_feats_emg.pkl"
    model = load(model_path)
    scaler_new = load(scaler_path)
    imu_scalers = load('../../../data/mvc_values/imu_scalers.pkl')
    
    label_map = {
    0: "no weight",
    1: "light",
    2: "medium",
    3: "heavy"}
    
    # Channels of interest
    channel_names = [
        'channel_1', 'channel_2', 'channel_3', 'channel_4',
        'channel_5', 'channel_6', 'channel_7', 'channel_8']
    imu_names = [
        'ACC_X', 'ACC_Y', 'ACC_Z',
        'GYR_X', 'GYR_Y', 'GYR_Z']
    fs = 200  # Myo nominal frequency in Hz
    fs_imu = 50
    window_size = 600  # Local buffer (3 seconds)
    window_feature_size = 40 # Dimension of window size onto which the feature is computed
    tolerance = 0 # Tolerance for buffer overlap 
    elapsed_times = []
    all_predictions = []
    all_probabilities = []
    total_start = time.time() 
    # Inizializza una coda fissa di massimo 11 elementi
    history_classes = deque(maxlen=6)
    history_probas = deque(maxlen=6)
    last_printed_class = None 

    # Supponiamo che 'no weight' sia la classe 0 (modifica se diverso)
    NO_WEIGHT_CLASS = 0
    THRESHOLD = 0.6
    in_weight_state = False 
    
    # Extracted Features Definition
    features_list_norm = ['MAV', 'WL', 'VAR']  # Features on normalized signal
    features_list_filt = ['ZC', 'SSC']  # Features on filtered signal 
    features_list_imu = ['MAV', 'SKEW', 'VAR']
    # Create list of ordered features (same order as scaler and model)
    selected_feature_names = []
    for ch in sorted(channel_names):  
        feats = sorted(features_list_norm + features_list_filt)
        #feats = sorted(features_list_norm)
        selected_feature_names.extend([f"{feat}_emg_{ch}" for feat in feats])
    #for i,ch in enumerate(sorted(imu_names)):
    #    feats = sorted(features_list_imu)
    #    selected_feature_names.extend([f"{feat}_myo_{ch}" for feat in feats])
    #    if i == 2:  # indice 2 = terzo elemento (contando da 0)
    #        selected_feature_names.append("VM_myo_ACC_all")
        
    #selected_feature_names.append("VM_myo_GYR_all")
        

    # Initialize buffer variables
    emg_ch = np.zeros((window_size, 8))       
    emg_norm = np.zeros((window_size, 8))
    emg_filt = np.zeros((window_size, 8))
    previous_block_emg = np.array([])
    sample_counter_emg = 0

    imu_ch = np.zeros((window_size, 6))       
    imu_filt = np.zeros((window_size, 6))
    previous_block_imu = np.array([])
    sample_counter_imu = 0

    # Live Plot Settings
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=1, label='EMG Channel 5')
    ax.set_title("Live EMG Channel 5 Normalized")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(0, 1.5)
    ax.legend()

    # Connect to Myo Armband
    streamer, shared_memory = myo_streamer(emg=True, imu=True, filtered=False)
    odh = OnlineDataHandler(shared_memory_items=shared_memory)
    odh.reset()
    
    try:
        while True:
            start_time = time.time()
            # Take last 20 unfiltered samples from shared buffer
            data, _ = odh.get_data(N=10, filter=False)
            if 'emg' in data and len(data['emg']) > 0:
                # Take only emg samples
                curr_emg = np.array(data['emg']) 
                # Check overlap with previous iteration
                curr_ch5 = curr_emg[:, 4]
                curr_new_ch5 = remove_overlap(previous_block_emg, curr_ch5, tol=tolerance)
                if curr_new_ch5.size == 0:
                    continue
                previous_block_emg = curr_ch5.copy()
                n = len(curr_new_ch5)
                curr_new = curr_emg[-n:, :]
                sample_counter_emg += n
        
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
                
                # Update Live Plot   
                t0 = (sample_counter_emg - window_size) / fs
                t1 = sample_counter_emg / fs
                time_vals = np.linspace(t0, t1, window_size)
                line.set_data(time_vals, emg_norm[:, 4])
                ax.set_xlim(t0, t1)
                    
            if 'imu' in data and len(data['imu']) > 0:
                curr_imu = np.array(data['imu'])
                curr_imu = curr_imu[:, -6:]
                curr_acc_X = curr_imu[:, 0]
                curr_new_acc_X = remove_overlap(previous_block_imu, curr_acc_X, tol=tolerance)
                if curr_new_acc_X.size == 0:
                    continue
                previous_block_imu = curr_acc_X.copy()
                n = len(curr_new_acc_X)
                curr_new_imu = curr_imu[-n:, :]
                sample_counter_imu += n
                    
                imu_ch = np.roll(imu_ch, -n, axis=0)
                imu_ch[-n:, :] = curr_new_imu
                    
                for ch in range(6):
                    imu_filtered = imu_lowpass_filt(imu_ch[:, ch], fs=fs_imu).reshape(-1, 1)
                    
                    # Usa lo scaler già addestrato
                    imu_scaled = imu_scalers[imu_names[ch]].transform(imu_filtered).ravel()
                    
                    imu_filt[:, ch] = np.roll(imu_filt[:, ch], -n)
                    imu_filt[-n:, ch] = imu_scaled[-n:]
                
            # Extract features and predict weight class
            if sample_counter_imu > window_feature_size and sample_counter_emg > window_feature_size:
                # Create dictionaries
                normalized_win = {channel_names[ch]: [emg_norm[-window_feature_size:, ch].reshape(1, -1)] for ch in range(8)}
                filtered_win = {channel_names[ch]: [emg_filt[-window_feature_size:, ch].reshape(1, -1)] for ch in range(8)}
                imu_win = {imu_names[ch]: [imu_filt[-window_feature_size:, ch].reshape(1, -1)] for ch in range(6)}
                
                # Extract features
                time_features_norm = extract_emg_features(normalized_win, features_list_norm)
                time_features_filt = extract_emg_features(filtered_win, features_list_filt)
                features_imu_myo = extract_imu_features_myo(imu_win, features_list_imu)
                vm_features_myo = compute_vm_features_myo(imu_win)
                
                df_imu = features_imu_myo[0]
                df_norm = time_features_norm[0]
                df_filt = time_features_filt[0]
                df_vm = vm_features_myo[0]
                #X_features = pd.concat([df_norm, df_filt, df_imu, df_vm], axis=1)
                X_features = pd.concat([df_norm, df_filt], axis=1)
                X_features = X_features[selected_feature_names]
                        
                # Scale features
                X_scaled = scaler_new.transform(X_features)
                X_scaled = pd.DataFrame(X_scaled, columns=selected_feature_names)
                        
                # Model Prediction
                prediction = model.run(X_scaled)
                pred_class, pred_proba = prediction
                pred_class = int(pred_class[0])  # da array([1]) a 1
                pred_proba = float(pred_proba[0])  # da array([0.97...]) a float
                    
                all_predictions.append(pred_class)
                all_probabilities.append(pred_proba)
                
                """if pred_proba > 0.70:
                    print(f"Prediction: {label_map[pred_class]} ({pred_proba:.2f} probability)")"""
                    
                # Controlla se ci sono almeno 3 predizioni
                if len(all_predictions) >= 4:
                    last_3_classes = all_predictions[-3:]
                    last_3_probas = all_probabilities[-3:]
                
                    # Se tutte le ultime 3 classi sono uguali e hanno probabilità > 0.6
                    if (
                        all(c == last_3_classes[0] for c in last_3_classes) and
                        all(p > 0.7 for p in last_3_probas)
                    ):
                        stable_class = last_3_classes[0]
                
                        # Se siamo in "no_weight" e prima eravamo in un'altra classe → STAMPA
                        if stable_class == 0 and last_printed_class != 0:
                            print(f"Prediction: {label_map[stable_class]}")
                            last_printed_class = 0
                
                        # Se siamo passati da "no_weight" a una nuova classe → STAMPA
                        elif stable_class != 0 and last_printed_class == 0:
                            print(f"Prediction: {label_map[stable_class]}")
                            last_printed_class = stable_class

                                                
            plt.pause(0.001)            
            end_time = time.time()
            elapsed_time = end_time - start_time
            elapsed_times.append(elapsed_time)
            if 0.050-elapsed_time > 0:
                time.sleep(0.050-elapsed_time) # Wait for new data on the buffer

    except KeyboardInterrupt:
        total_end = time.time()  # Tempo finale globale
        total_duration = total_end - total_start
        
        average_time = sum(elapsed_times) / len(elapsed_times)
        print(f"\nEsecuzioni: {len(elapsed_times)}")
        print(f"Tempo medio trascorso: {average_time:.3f} secondi")
        print(f"Tempo totale trascorso: {total_duration:.3f} secondi")
        pass
    finally:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()