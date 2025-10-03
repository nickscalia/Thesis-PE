EMG: features_list_norm = ['MAV', 'WL', 'VAR'] features_list_filt = ['ZC', 'SSC'] 
IMU: features_list_imu = ['MAV', 'SKEW', 'VAR']
EMG2: features_list_norm = ['MAV', 'WL', 'VAR', 'SAMPEN']  features_list_filt = ['ZC', 'SSC', 'KURT', 'SKEW'], gb_params_2 = {
    'n_estimators': 800,          # più alberi
    'learning_rate': 0.02,        # più lento ma più accurato
    'max_depth': 6,               # alberi più profondi
    'subsample': 0.7,             # un po’ più casualità per migliorare generalizzazione
    'min_samples_split': 3,       # split più aggressivi
    'max_features': 0.8           # più feature per split
}