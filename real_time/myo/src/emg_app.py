# -----------------------------------------------------------------------------
# Copyright (c) 2025 Nicolas Scalia - Politecnico di Milano
# All rights reserved.
#
# This script is part of the research published in:
# [Your Paper Title], [Conference/Journal Name], [Year]
# DOI: [Insert DOI if available]
#
# Author: Nicolas Scalia (nicolas.scalia@mail.polimi.it)
# -----------------------------------------------------------------------------

#%% CODE EXPLAINATION 
# This script defines EMGApp class for real-time EMG/IMU processing, including calibration, feature extraction, prediction, and data logging. 
# Manages GUI elements and user interactions for calibration, estimation, stopping, and exiting. 
# Saves all relevant raw and processed data.

# Necessary libraries
import time
import sys
import os
import collections
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ttkbootstrap as ttk
from libemg.streamers import myo_streamer
from libemg.data_handler import OnlineDataHandler
from joblib import load
from sklearn.preprocessing import MinMaxScaler
from ttkbootstrap import Style
from ttkbootstrap.constants import *
from tkinter import messagebox
from datetime import datetime

# Define directories
base_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(base_dir, os.pardir))
grandparent_dir = os.path.abspath(os.path.join(base_dir, os.pardir, os.pardir))
lib_dir = os.path.join(parent_dir, "lib")
sys.path.append(lib_dir)
from emg_utils import remove_overlap, bandpass_filter, rectification, RMS_moving, normalization, extract_emg_features
from imu_utils import imu_lowpass_filt, extract_imu_features_myo, compute_vm_features_myo
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

#%% Definition of the EMGApp Class
class EMGApp:
    def __init__(self, master):       
        self.master = master
        self.already_printed = set() 
        self.master.title("EMG Payload Estimator GUI")
        self.master.geometry("600x400")

        self.label = ttk.Label(self.master, text="Welcome to the EMG Payload Estimator", font=("Segoe UI", 14, "bold"))
        self.label.pack()

        style = Style()
        style.configure('primary.TButton', font=('Segoe UI', 10, 'bold'))
        style.configure('success.TButton', font=('Segoe UI', 10, 'bold'))
        style.configure('warning.TButton', font=('Segoe UI', 10, 'bold'))
        style.configure('danger.TButton', font=('Segoe UI', 10, 'bold'))

        self.calibrate_button = ttk.Button(
            self.master,
            text="CALIBRATE",
            command=self.on_calibrate,
            bootstyle="primary",  
            padding=10,
        )
        self.calibrate_button.pack(fill='x', expand=True, padx=230, pady=5)

        self.estimate_button = ttk.Button(
            self.master,
            text="ESTIMATE",
            command=self.on_estimate,
            bootstyle="success",  
            padding=10,      
        )
        self.estimate_button.pack(fill='x', expand=True, padx=230, pady=5)

        self.stop_button = ttk.Button(
            self.master,
            text="STOP",
            command=self.stop_process,
            bootstyle="warning",  
            padding=10,    
        )
        self.stop_button.pack(fill='x', expand=True, padx=230, pady=5)

        self.exit_button = ttk.Button(
            self.master,
            text="EXIT",
            command=self.confirm_exit,
            bootstyle="danger",  
            padding=10,           
        )
        self.exit_button.pack(fill='x', expand=True, padx=230, pady=5)

        self.separator = ttk.Separator(self.master, orient='horizontal')
        self.separator.pack(fill='x', pady=10)
        
        self.status_label = ttk.Label(self.master, text="Ready for next action.", font=("Segoe UI", 12))
        self.status_label.pack(pady=(5,20))

        self.initialize_variables()
        
    def initialize_variables(self):
        # External Variables    
        self.data_folder = os.path.join(parent_dir, "data", timestamp)
        self.cali_folder = os.path.join(self.data_folder, "calibrate")
        self.est_folder = os.path.join(self.data_folder, "estimate")
        
        self.models_dir = os.path.join(parent_dir, "models", "emg_imu", "74", "GB2")
        self.output_dir = os.path.join(grandparent_dir, "shared", "temp")
        self.gate_path = os.path.join(self.output_dir, "myo_gate.csv")
        self.pred_path = os.path.join(self.output_dir, "myo_data.csv")
        
        
        self.model = load(os.path.join(self.models_dir, "model.pkl"))
        self.scaler = load(os.path.join(self.models_dir, "scaler.pkl"))
        #self.pca = load('/../data/stream/models/pca/emg_imu/pca.pkl')

        # Channels of interest
        self.channel_names = ['channel_1', 'channel_2', 'channel_3', 'channel_4',
                         'channel_5', 'channel_6', 'channel_7', 'channel_8']
        self.imu_names = ['ACC_X', 'ACC_Y', 'ACC_Z',
                     'GYR_X', 'GYR_Y', 'GYR_Z']
        self.fs = 200  # Myo nominal frequency in Hz
        self.window_size = 600  # Local buffer (3 seconds)
        self.tolerance = 1 # Tolerance for buffer overlap
        self.label_map = {0: "no weight", 1: "light", 2: "medium", 3: "heavy"} # Maps class IDs to labels
        self.norm_values = {ch: 0 for ch in self.channel_names}

        self.imu_scalers = []
        self.fs_imu = 50

        self.estimate_count = 1
        self.calibrate_count = 1
        self.calib_time = 0
        self.calibration = 17
        self.interrupt_flag = False
     
    
    def check_elapsed(self, elapsed):
        if elapsed > self.calibration:
            self.status_label.config(text="End of Calibration", font=("Segoe UI", 12, "bold"))
            self.master.update() 
            self.already_printed.clear() 
            raise Exception("Calibration finished")
        if elapsed > 13 and "lower_2" not in self.already_printed:
            self.status_label.config(text="Lower", font=("Segoe UI", 14, "bold"))
            self.master.update() 
            self.already_printed.add("lower_2")
        if elapsed > 9 and "lift_2" not in self.already_printed:
            self.status_label.config(text="Lift", font=("Segoe UI", 14, "bold"))
            self.master.update() 
            self.already_printed.add("lift_2")
        if elapsed > 6 and "lower_1" not in self.already_printed:
            self.status_label.config(text="Lower", font=("Segoe UI", 14, "bold"))
            self.master.update() 
            self.already_printed.add("lower_1")
        if elapsed > 2 and "lift_1" not in self.already_printed:
            self.status_label.config(text="Lift",font=("Segoe UI", 14, "bold"))
            self.master.update() 
            self.already_printed.add("lift_1")

    # =============================================================================
    # MAIN METHODS CONNECTED TO GUI BUTTONS
    # Each method corresponds to one of the four main buttons:
    # - CALIBRATE → on_calibrate()
    # - ESTIMATE  → on_estimate()
    # - STOP      → stop_process()
    # - EXIT      → confirm_exit()
    # =============================================================================

    def on_calibrate(self):
        # Create data folder (ony the first time) and calibation folder
        if self.calibrate_count == 1: os.makedirs(self.data_folder, exist_ok=True)
        self.cali_folder_i = os.path.join(self.cali_folder, f"{self.calibrate_count}")
        os.makedirs(self.cali_folder_i, exist_ok=True)

        messagebox.showinfo("Calibration", "The calibration will begin in few seconds")

        # Initialize variables
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
        fig, ax = plt.subplots(figsize=(5, 4))
        line, = ax.plot([], [], lw=1, label='EMG Channel 5')
        ax.set_title("Live Smoothed EMG Channel 5")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("bit")
        ax.set_ylim(0, 100)
        ax.legend()

        # Connect to Myo Armband
        _, shared_memory = myo_streamer(emg=True, imu=True, filtered=False)
        odh = OnlineDataHandler(shared_memory_items=shared_memory)
        odh.reset()
        
        try:
            total_start = time.time() 
            self.status_label.config(text="Beginning of Calibration", font=("Segoe UI", 14, "bold"))
            self.master.update()
            while not self.interrupt_flag:
                start_time = time.time()
                data, _ = odh.get_data(N=10, filter=False)  # Take last 10 unfiltered samples from shared buffer
                elapsed = time.time() - total_start
                self.check_elapsed(elapsed)
                
                if 'emg' in data and len(data['emg']) > 0:
                    curr_emg = np.array(data['emg']) # Take only emg samples
                
                    # Check overlap with previous iteration
                    curr_ch5 = curr_emg[:, 4]
                    curr_new_ch5 = remove_overlap(previous_block_emg, curr_ch5, self.tolerance)
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

                        if window_med > self.norm_values[ch_name] and smoothness < smooth_threshold:
                            self.norm_values[ch_name] = window_med
                
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
                    time.sleep(0.050 - elapsed_time) # Maintain at least 50 ms loop interval
                        
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

            plt.ioff()
            plt.close(fig)

            # Store calibration data
            for ch in range(6):
                scaler_temp = MinMaxScaler(feature_range=(-1, 1))
                scaler_temp.fit(imu_filt[:, ch].reshape(-1, 1))
                self.imu_scalers.append(scaler_temp)
            imu_scaler_path = os.path.join(self.cali_folder_i, "imu_scalers.pkl")
            norm_path = os.path.join(self.cali_folder_i, "norm_values.csv")
            norm_df = pd.DataFrame(list(self.norm_values.items()), columns=["Channel", "Norm_Value"])
            norm_df.to_csv(norm_path, index=False)
            with open(imu_scaler_path, 'wb') as f:
                pickle.dump(self.imu_scalers, f)

            self.calibrate_count += 1
            self.interrupt_flag = False

            self.status_label.config(text="Ready for next action.", font=("Segoe UI", 12))
            self.master.update() 


    def on_estimate(self):
        # Require calibation before estimation (only first time)
        if any(value == 0 for value in self.norm_values.values()):
            messagebox.showerror("Calibration required", "You must perform the calibration first.")
            return
        
        # Create estimation folder
        self.est_folder_i = os.path.join(self.est_folder, f"{self.estimate_count}")
        os.makedirs(self.est_folder_i, exist_ok=True)
        metadata = {
            "calibration_index": self.calibrate_count-1,
            "model_used": self.models_dir,
        }  

        messagebox.showinfo("Estimation", "The estimation will begin in few seconds")

        # Initialize variables
        elapsed_times = [] # Stores processing times
        previous_block_emg = np.array([])
        sample_counter = 0
        emg_ch = np.zeros((self.window_size, 8)) 
        emg_norm = np.zeros((self.window_size, 8))
        emg_filt = np.zeros((self.window_size, 8))
        imu_ch = np.zeros((self.window_size, 6)) 
        imu_filt = np.zeros((self.window_size, 6))
        window_feature_size = 40 # Dimension of window size onto which the feature is computed
        self.raw_emg_data = []  
        self.norm_emg_data = []  
        self.raw_imu_data = []
        all_predictions = [] # Stores predicted classes
        all_probabilities = [] # Stores class probabilities
        stable_classes_log = []
        last_printed_class = None # Tracks last printed class
        warning_shown = False
        pd.DataFrame({'boolean':[2]}).to_csv(self.gate_path)
        
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

        # Live Plot Settings
        plt.ion()
        fig, ax = plt.subplots(figsize=(5, 4))
        line, = ax.plot([], [], lw=1, label='EMG Channel 5')
        ax.set_title("Live Normalized EMG Channel 5")
        ax.set_ylabel("Normalized EMG")
        ax.set_ylim(0, 1.6)
        ax.legend()

        # Connect to Myo Armband
        _, shared_memory = myo_streamer(emg=True, imu=True, filtered=False)
        odh = OnlineDataHandler(shared_memory_items=shared_memory)
        odh.reset()
        
        try:
            total_start = time.time() 
            start_transition = time.time()
            self.status_label.config(text="Beginning of Estimation",font=("Segoe UI", 14, "bold"))
            self.master.update()
            while not self.interrupt_flag:
                # Please recalibrate every 30 minutes to ensure accuracy
                if time.time() - self.calib_time > 1800 and not warning_shown:
                    messagebox.showwarning("Calibration timeout", "It has been over 30 minutes since the last calibration. Please recalibrate for accurate results.")
                    warning_shown = True
                    
                start_time = time.time()
                data, _ = odh.get_data(N=10, filter=False)  # Take last 10 unfiltered samples from shared buffer

                if 'emg' in data and len(data['emg']) > 0:
                    curr_emg = np.array(data['emg']) # Take only emg samples
                    
                    # Check overlap with previous iteration
                    curr_ch5 = curr_emg[:, 4]
                    curr_new_ch5 = remove_overlap(previous_block_emg, curr_ch5, self.tolerance)
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
                        emg_norm_filtered = normalization(emg_rms, self.channel_names[ch], self.norm_values)
                
                        # Store filtered and normalized values to local buffer
                        emg_filt[:, ch] = np.roll(emg_filt[:, ch], -n)
                        emg_filt[-n:, ch] = emg_filtered[-n:]
                        emg_norm[:, ch] = np.roll(emg_norm[:, ch], -n)
                        emg_norm[-n:, ch] = emg_norm_filtered[-n:]
                            
                    self.raw_emg_data.append(curr_new)          
                    self.norm_emg_data.append(emg_norm[-n:, :]) 

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
                    
                    self.raw_imu_data.append(imu_filt[-n:, :])    
                      
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
                    #X_scaled = self.pca.transform(X_scaled) # Only for PCA features
                              
                    # Model Prediction
                    prediction = self.model.run(X_scaled)
                    pred_class, pred_proba = prediction
                    pred_class = int(pred_class[0])
                    pred_proba = float(pred_proba[0])
                    
                    all_predictions.append(pred_class)
                    all_probabilities.append(pred_proba)
                            
                    print(f"Prediction: {self.label_map[pred_class]} ({pred_proba:.2f} probability)")
                        
                    # Onset and Offset Detection
                    if len(all_predictions) >= 2:    
                        last_class = all_predictions[-2] 
                        if last_class == 0 and pred_class !=0:
                            start_transition = time.time()
                            pd.DataFrame({'boolean':[1]}).to_csv(self.gate_path)
                        elif last_class != 0 and pred_class == 0:
                            start_transition = time.time()
                            pd.DataFrame({'boolean':[0]}).to_csv(self.gate_path)
                        
                    # This logic implements a stability check on predictions.
                    # The code validates a prediction only if an onset or offset has been detected.
                    # It accepts a predicted class only if it has been repeated 3 times 
                    # consecutively with high confidence (>0.8).

                    if len(all_predictions) >= 3:
                        last_classes = all_predictions[-3:]
                        last_probas = all_probabilities[-3:]
                        stable_class = None

                        if start_transition is not None: 
                            elapsed = time.time() - start_transition
                            
                            if (all(c == last_classes[0] for c in last_classes) and
                                all(p > 0.8 for p in last_probas)):
                                stable_class = last_classes[0]

                            # If no class is validated within 0.5s from the beginning of the transition, 
                            # the most frequent class with prob >0.6 is assigned.
                            elif elapsed > 0.5:
                                recent_classes = all_predictions[-10:]
                                recent_probas = all_probabilities[-10:]

                                filtered_classes = [cls for cls, prob in zip(recent_classes, recent_probas) if prob >0.6]
                                counter = collections.Counter(filtered_classes)
                                max_freq = max(counter.values())
                                candidates = [cls for cls, freq in counter.items() if freq == max_freq]
                                if len(candidates) > 1:
                                    for cls in reversed(filtered_classes):
                                        if cls in candidates:
                                            stable_class = cls
                                            break
                                else:
                                    stable_class = candidates[0]

                        # Print and save the validated payload estimate
                        if stable_class is not None:
                            pd.DataFrame({'boolean':[2]}).to_csv(self.gate_path)

                            if stable_class == 0 and last_printed_class != 0:
                                pd.DataFrame({'boolean':[4]}).to_csv(self.pred_path, index=False)
                                end_transition = time.time()
                                transition = round(end_transition - start_transition, 3)
                                self.status_label.config(text=f"Prediction: {self.label_map[stable_class]}, prediction time: {transition}")
                                self.master.update()
                                last_printed_class = 0
                                    
                                stable_classes_log.append({
                                    'label': stable_class,
                                    'probability': np.mean(last_probas), 
                                    'transition_time': transition
                                })
                                start_transition = None
                        
                            elif stable_class != 0 and last_printed_class == 0:
                                pd.DataFrame({'boolean':[stable_class-1]}).to_csv(self.pred_path, index=False)
                                end_transition = time.time()
                                transition = round(end_transition - start_transition, 3)
                                self.status_label.config(text=f"Prediction: {self.label_map[stable_class]}, prediction time: {transition}")
                                self.master.update()      
                                last_printed_class = stable_class
                               
                                stable_classes_log.append({
                                     'label': stable_class,
                                    'probability': np.mean(last_probas), 
                                    'transition_time': transition
                                    })
                                start_transition = None
                                
                plt.pause(0.001) # Update the plot
                end_time = time.time()
                elapsed_time = end_time - start_time
                elapsed_times.append(elapsed_time)
                if 0.050 - elapsed_time > 0:
                    time.sleep(0.050 - elapsed_time) # Maintain at least 50 ms loop interval
            
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
            plt.close(fig)

            # Store estimation data
            df = pd.DataFrame({
                'class': all_predictions,
                'probability': all_probabilities
            })
            df2 = pd.DataFrame(stable_classes_log)
            pred_file = os.path.join(self.est_folder_i, 'predictions_log.csv')
            class_file = os.path.join(self.est_folder_i, 'classes_log.csv')
            df.to_csv(pred_file, index=False)
            df2.to_csv(class_file, index=False)

            emg_raw_all = np.vstack(self.raw_emg_data)
            imu_raw_all = np.vstack(self.raw_imu_data)
            emg_norm_all = np.vstack(self.norm_emg_data)
            combined_data = np.hstack((emg_raw_all, imu_raw_all))
            columns = self.channel_names + self.imu_names
            df_data = pd.DataFrame(combined_data, columns=columns)
            df_data.to_csv(os.path.join(self.est_folder_i, "raw_emg_imu.csv"), index=False)
            df_emg_norm = pd.DataFrame(emg_norm_all, columns=self.channel_names)
            df_emg_norm.to_csv(os.path.join(self.est_folder_i, "emg_norm.csv"), index=False)        
            self.interrupt_flag = False
            with open(os.path.join(self.est_folder_i, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=4)

            self.estimate_count += 1
            
            self.status_label.config(text="Ready for next action.", font=("Segoe UI", 12))
            self.master.update() 

    def stop_process(self):
        self.interrupt_flag = True
        self.status_label.config(text="Process interrupted by user.", font=("Segoe UI", 14, "bold"))
        self.master.update()
        time.sleep(1)
        self.status_label.config(text="Ready for next action.", font=("Segoe UI", 12))
        self.master.update() 

    def confirm_exit(self):
        if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
            self.master.destroy()