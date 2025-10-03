# Nicolas Scalia
# Payload Estimation Master Thesis
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
import tkinter as tk
from tkinter import messagebox

class EMGApp:
    def __init__(self, master):
        self.master = master
        self.already_printed = set() 
        master.title("EMG Payload Estimation GUI")
        master.geometry("1000x800")

        self.label = tk.Label(master, text="Welcome to the EMG Payload Estimator")
        self.label.pack()

        self.calibrate_button = tk.Button(
            master,
            text="CALIBRATE",
            command=self.on_calibrate,
            width=20,
            height=2,
            bg="orange"
        )
        self.calibrate_button.pack(pady=20)

        self.estimate_button = tk.Button(
            master,
            text="ESTIMATE",
            command=self.on_estimate,
            width=20,
            height=2,
            bg="green"
        )
        self.estimate_button.pack(pady=20)
        
        self.status_label = tk.Label(self.master, text="", font=("Arial", 16))
        self.status_label.pack(pady=10)
        
        self.initialize_variables()
        
    def initialize_variables(self):
        # External Variables    
        self.model = load('../../../data/stream/models/emg_imu/74/GB2/model.pkl')
        self.scaler = load('../../../data/stream/models/emg_imu/74/GB2/scaler.pkl') 
        self.pca = load('../../../data/stream/models/pca/emg_imu/pca.pkl')
        # Channels of interest
        self.channel_names = ['channel_1', 'channel_2', 'channel_3', 'channel_4',
                         'channel_5', 'channel_6', 'channel_7', 'channel_8']
        self.imu_names = ['ACC_X', 'ACC_Y', 'ACC_Z',
                     'GYR_X', 'GYR_Y', 'GYR_Z']
        self.fs = 200  # Myo nominal frequency in Hz
        self.window_size = 600  # Local buffer (3 seconds)
        self.tolerance = 1 # Tolerance for buffer overlap
        self.calibration = 15
        self.label_map = {0: "no weight", 1: "light", 2: "medium", 3: "heavy"} # Maps class IDs to labels
        self.mvc_values = {ch: 0 for ch in self.channel_names}
        self.calib_time = 0
             
        self.imu_scalers = []
        self.fs_imu = 50 
        
    def remove_overlap(self, prev, curr, tol):
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
    
    def MVC_normalization(self, signal, muscle_name, mvc_values):
        """
        Normalizes EMG signal by MVC value from a dictionary.
        """
        if muscle_name not in mvc_values:
            raise ValueError(f"Muscle name '{muscle_name}' not found in MVC dictionary.")   
        mvc = mvc_values[muscle_name]
        return signal / mvc
    
    def check_elapsed(self, elapsed):
        if elapsed > self.calibration:
            self.status_label.config(text="End of Calibration")
            self.master.update() 
            self.already_printed.clear() 
            raise Exception("Calibration finished")
        if elapsed > 13 and "lower_13" not in self.already_printed:
            self.status_label.config(text="Lower")
            self.master.update() 
            self.already_printed.add("lower_13")
        if elapsed > 9 and "lift_9" not in self.already_printed:
            self.status_label.config(text="Lift")
            self.master.update() 
            self.already_printed.add("lift_9")
        if elapsed > 6 and "lower_6" not in self.already_printed:
            self.status_label.config(text="Lower")
            self.master.update() 
            self.already_printed.add("lower_6")
        if elapsed > 2 and "lift_2" not in self.already_printed:
            self.status_label.config(text="Lift")
            self.master.update() 
            self.already_printed.add("lift_2")

    def on_calibrate(self):
        messagebox.showinfo("Calibration", "The calibration will begin in few seconds")
        elapsed_times = [] # Stores processing times
        previous_block_emg = np.array([])
        sample_counter = 0
        emg_ch = np.zeros((self.window_size, 8)) 
        emg_smoot = np.zeros((self.window_size, 8))
        imu_ch = np.zeros((self.window_size, 6)) 
        imu_filt = None
        smooth_threshold = 3
        
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
            self.status_label.config(text="Beginning of Calibration")
            self.master.update()
            while True:
                start_time = time.time()
                data, _ = odh.get_data(N=10, filter=False)  # Take last 10 unfiltered samples from shared buffer
                elapsed = time.time() - total_start
                self.check_elapsed(elapsed)
                
                if 'emg' in data and len(data['emg']) > 0:
                    curr_emg = np.array(data['emg']) # Take only emg samples
                    
                    # Check overlap with previous iteration
                    curr_ch5 = curr_emg[:, 4]
                    curr_new_ch5 = self.remove_overlap(previous_block_emg, curr_ch5, self.tolerance)
                    if curr_new_ch5.size == 0:
                        continue
                    previous_block_emg = curr_ch5.copy()
                    n = len(curr_new_ch5)
                    
                    # Add only new data to the local buffer
                    curr_new = curr_emg[-n:, :]
                    sample_counter += n
                    emg_ch = np.roll(emg_ch, -n, axis=0) 
                    emg_ch[-n:, :] = curr_new
                        
                    # Preprocessing of emg signals      
                    for ch, ch_name in enumerate(self.channel_names):
                        emg_filtered = bandpass_filter(emg_ch[:, ch], self.fs, high_freq=self.fs/2 -1)
                        emg_rectified = rectification(emg_filtered)
                        emg_rms = RMS_moving(emg_rectified, self.fs)
                            
                        window_med = np.median(emg_rms)
                        smoothness = np.mean(np.abs(np.diff(emg_rms)))

                        if window_med > self.mvc_values[ch_name] and smoothness < smooth_threshold:
                            self.mvc_values[ch_name] = window_med
                
                        # Store smoothed values to local buffer
                        emg_smoot[:, ch] = np.roll(emg_smoot[:, ch], -n)
                        emg_smoot[-n:, ch] = emg_rms[-n:]
                        
                    # Update Live Plot   
                    t0 = (sample_counter - self.window_size) / self.fs
                    t1 = sample_counter / self.fs
                    time_vals = np.linspace(t0, t1, self.window_size)
                    line.set_data(time_vals, emg_smoot[:, 4])
                    ax.set_xlim(t0, t1) 
                    plt.pause(0.001) # Update the plot
                
                if 'imu' in data and len(data['imu']) > 0:
                    curr_imu = np.array(data['imu']) # Take only imu samples
                    curr_imu = curr_imu[:, -6:]
                        
                    # Add only new data to the local buffer
                    curr_new_imu = curr_imu[-n:, :]
                    imu_ch = np.roll(imu_ch, -n, axis=0)
                    imu_ch[-n:, :] = curr_new_imu
                            
                    imu_filt_temp = []
                    for ch in range(6):
                        imu_filtered = imu_lowpass_filt(imu_ch[:, ch], fs=self.fs_imu).reshape(-1, 1) 
                        # Store filtered values 
                        imu_filt_temp.append(imu_filtered)
                        
                    imu_filt_temp = np.hstack(imu_filt_temp)  # shape: (N, 6)
                        
                    if imu_filt is None:
                        imu_filt = imu_filt_temp
                    else:
                        imu_filt = np.vstack((imu_filt, imu_filt_temp))
                                      
                end_time = time.time()
                elapsed_time = end_time - start_time
                elapsed_times.append(elapsed_time)
                if 0.050 - elapsed_time > 0:
                    time.sleep(0.050 - elapsed_time) # Maintain 50 ms loop interval
                        
        except Exception as e:
            print(f"Caught exception: {e}")
            
        finally:
            self.calib_time = time.time()
            total_end = time.time()  # Global end time
            total_duration = total_end - total_start  # Calculate total elapsed time
            average_time = sum(elapsed_times) / len(elapsed_times)  # Calculate average processing time
            print(f"\nExecutions: {len(elapsed_times)}")
            print(f"Average processing time: {average_time:.3f} seconds")
            print(f"Total time: {total_duration:.3f} seconds")
            
            for ch in range(6):
                scaler_temp = MinMaxScaler(feature_range=(-1, 1))
                scaler_temp.fit(imu_filt[:, ch].reshape(-1, 1))
                self.imu_scalers.append(scaler_temp)
            
            plt.ioff()
            plt.show()

    def on_estimate(self):
        if any(value == 0 for value in self.mvc_values.values()):
            messagebox.showerror("Calibration required", "You must perform the calibration first.")
            return
        
        messagebox.showinfo("Estimation", "The estimation will begin in few seconds")
        elapsed_times = [] # Stores processing times
        previous_block_emg = np.array([])
        sample_counter = 0
        emg_ch = np.zeros((self.window_size, 8)) 
        emg_norm = np.zeros((self.window_size, 8))
        emg_filt = np.zeros((self.window_size, 8))
        imu_ch = np.zeros((self.window_size, 6)) 
        imu_filt = np.zeros((self.window_size, 6))
        window_feature_size = 40 # Dimension of window size onto which the feature is computed
        
        # Extracted Features Definition
        features_list_norm = ['MAV', 'WL','VAR', 'WAMP']  # Features on normalized signal
        features_list_filt = ['ZC', 'SSC']  # Features on filtered signal
        features_list_imu = ['MAV', 'SKEW', 'VAR','WAMP']
        
        # Create list of ordered features (same order as scaler and model)
        selected_feature_names = []
        for ch in sorted(self.channel_names):  
            feats = sorted(features_list_norm + features_list_filt)
            selected_feature_names.extend([f"{feat}_emg_{ch}" for feat in feats])
        
        for i,ch in enumerate(sorted(self.imu_names)):
            feats = sorted(features_list_imu)
            selected_feature_names.extend([f"{feat}_myo_{ch}" for feat in feats])
            if i == 2:  
                selected_feature_names.append("VM_myo_ACC_all")           
        selected_feature_names.append("VM_myo_GYR_all")
        
        all_predictions = [] # Stores predicted classes
        all_probabilities = [] # Stores class probabilities
        last_printed_class = None # Tracks last printed class
        warning_shown = False
        
        # Live Plot Settings
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot([], [], lw=1, label='EMG Channel 5')
        ax.set_title("Live Normalized EMG Channel 5")
        ax.set_ylabel("EMG/MVC")
        ax.set_ylim(0, 1.6)
        ax.legend()

        # Connect to Myo Armband
        streamer, shared_memory = myo_streamer(emg=True, imu=True, filtered=False)
        odh = OnlineDataHandler(shared_memory_items=shared_memory)
        odh.reset()
        
        try:
            total_start = time.time() 
            start_transition = time.time()
            self.status_label.config(text="Beginning of Estimation")
            self.master.update()
            while True:
                if time.time() - self.calib_time > 1800 and not warning_shown:
                    messagebox.showwarning("Calibration timeout", "It has been over 30 minutes since the last calibration. Please recalibrate for accurate results.")
                    warning_shown = True
                    
                start_time = time.time()
                data, _ = odh.get_data(N=10, filter=False)  # Take last 10 unfiltered samples from shared buffer

                if 'emg' in data and len(data['emg']) > 0:
                    curr_emg = np.array(data['emg']) # Take only emg samples
                    
                    # Check overlap with previous iteration
                    curr_ch5 = curr_emg[:, 4]
                    curr_new_ch5 = self.remove_overlap(previous_block_emg, curr_ch5, self.tolerance)
                    if curr_new_ch5.size == 0:
                        continue
                    previous_block_emg = curr_ch5.copy()
                    n = len(curr_new_ch5)
                    
                    # Add only new data to the local buffer
                    curr_new = curr_emg[-n:, :]
                    sample_counter += n
                    emg_ch = np.roll(emg_ch, -n, axis=0) 
                    emg_ch[-n:, :] = curr_new
                        
                    # Preprocessing of emg signals      
                    for ch in range(8):
                        emg_filtered = bandpass_filter(emg_ch[:, ch], self.fs, high_freq=self.fs/2 -1)
                        emg_rectified = rectification(emg_filtered)
                        emg_rms = RMS_moving(emg_rectified, self.fs)
                        emg_norm_filtered = self.MVC_normalization(emg_rms, self.channel_names[ch], self.mvc_values)
                
                        # Store filtered and normalized values to local buffer
                        emg_filt[:, ch] = np.roll(emg_filt[:, ch], -n)
                        emg_filt[-n:, ch] = emg_filtered[-n:]
                        emg_norm[:, ch] = np.roll(emg_norm[:, ch], -n)
                        emg_norm[-n:, ch] = emg_norm_filtered[-n:]
                            
                    # Update Live Plot   
                    t0 = (sample_counter - self.window_size) / self.fs
                    t1 = sample_counter / self.fs
                    time_vals = np.linspace(t0, t1, self.window_size)
                    line.set_data(time_vals, emg_norm[:, 4])
                    ax.set_xlim(t0, t1)  
                
                if 'imu' in data and len(data['imu']) > 0:
                    curr_imu = np.array(data['imu']) # Take only imu samples
                    curr_imu = curr_imu[:, -6:]
                        
                    # Add only new data to the local buffer
                    curr_new_imu = curr_imu[-n:, :]
                    imu_ch = np.roll(imu_ch, -n, axis=0)
                    imu_ch[-n:, :] = curr_new_imu

                    for ch in range(6):
                        imu_filtered = imu_lowpass_filt(imu_ch[:, ch], fs=self.fs_imu).reshape(-1, 1) 
                        imu_scaled = self.imu_scalers[ch].transform(imu_filtered).ravel()
                            
                        # Store filtered values to local buffer
                        imu_filt[:, ch] = np.roll(imu_filt[:, ch], -n)
                        imu_filt[-n:, ch] = imu_scaled[-n:]
                      
                # Extract features and predict weight class
                if sample_counter > window_feature_size:
                    # Create EMG dictionaries
                    normalized_win = {self.channel_names[ch]: [emg_norm[-window_feature_size:, ch].reshape(1, -1)] for ch in range(8)}
                    filtered_win = {self.channel_names[ch]: [emg_filt[-window_feature_size:, ch].reshape(1, -1)] for ch in range(8)}
                            
                    # Extract EMG features
                    time_features_norm = extract_emg_features(normalized_win, features_list_norm)
                    time_features_filt = extract_emg_features(filtered_win, features_list_filt)
                    df_norm = time_features_norm[0]
                    df_filt = time_features_filt[0]
                    X_features = pd.concat([df_norm, df_filt], axis=1)
                    
                    imu_win = {self.imu_names[ch]: [imu_filt[-window_feature_size:, ch].reshape(1, -1)] for ch in range(6)}
                    features_imu_myo = extract_imu_features_myo(imu_win, features_list_imu)
                    vm_features_myo = compute_vm_features_myo(imu_win)
                    df_imu = features_imu_myo[0]
                    df_vm = vm_features_myo[0]
                    X_features = pd.concat([X_features, df_imu, df_vm], axis=1)
                    
                    
                    X_features = X_features[selected_feature_names] # Ordered features                   
                    X_scaled = self.scaler.transform(X_features) # Scale features
                    X_scaled = pd.DataFrame(X_scaled, columns=selected_feature_names)
                    
                    """
                    X_scaled = self.pca.transform(X_scaled)
                    """
                                    
                    # Model Prediction
                    prediction = self.model.run(X_scaled)
                    pred_class, pred_proba = prediction
                    pred_class = int(pred_class[0])
                    pred_proba = float(pred_proba[0])
                    
                    all_predictions.append(pred_class)
                    all_probabilities.append(pred_proba)
                            
                        
                    # Print prediction every 50 ms
                    if pred_proba > 0.50:
                        print(f"Prediction: {self.label_map[pred_class]} ({pred_proba:.2f} probability)")
                        
                        
                    if len(all_predictions) >= 2:    
                        last_class = all_predictions[-2] 
                        if last_class == 0 and pred_class !=0:
                            start_transition = time.time()
                        elif last_class != 0 and pred_class == 0:
                            start_transition = time.time()
                        
                    # This logic implements a stability check on predictions.
                    # It accepts a predicted class only if it has been repeated 3 times 
                    # consecutively with high confidence (>0.7).
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
                                self.status_label.config(text=f"Prediction: {self.label_map[stable_class]}, prediction time: {transition}")
                                self.master.update()
                                last_printed_class = 0
                    
                            elif stable_class != 0 and last_printed_class == 0:
                                end_transition = time.time()
                                transition = round(end_transition - start_transition, 3)
                                self.status_label.config(text=f"Prediction: {self.label_map[stable_class]}, prediction time: {transition}")
                                self.master.update()
                                last_printed_class = stable_class
                                
                plt.pause(0.001) # Update the plot
                end_time = time.time()
                elapsed_time = end_time - start_time
                elapsed_times.append(elapsed_time)
                if 0.050 - elapsed_time > 0:
                    time.sleep(0.050 - elapsed_time) # Maintain 50 ms loop interval
            
        except Exception as e:
            print(f"Caught exception: {e}")
            
        finally:
            total_end = time.time()  # Global end time
            total_duration = total_end - total_start  # Calculate total elapsed time
            average_time = sum(elapsed_times) / len(elapsed_times)  # Calculate average processing time
            print(f"\nExecutions: {len(elapsed_times)}")
            print(f"Average processing time: {average_time:.3f} seconds")
            print(f"Total time: {total_duration:.3f} seconds")
            
            plt.ioff()
            plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = EMGApp(root)
    root.mainloop()