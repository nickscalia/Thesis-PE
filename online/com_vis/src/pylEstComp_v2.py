# -----------------------------------------------------------------------------
# Copyright (c) 2025 Andrea Dal Prete - Politecnico di Milano
# Modifications (c) 2025 Nicolas Scalia - Politecnico di Milano
# All rights reserved.
#
# This script is based on work by Andrea Dal Prete 
# and has been modified by Nicolas Scalia.
#
# Original research published in:
# [Original Paper Title], [Conference/Journal Name], [Year]
# DOI: [Insert DOI if available]
#
# Author: Andrea Dal Prete (andrea.dalprete@polimi.it)
# Author of modifications: Nicolas Scalia (nicolas.scalia@mail.polimi.it)
# -----------------------------------------------------------------------------

#%%  CODE EXPLAINATION
# This script contains the payload estimation module for boxes weight classification. This script is meant to be run 
# in parallel to a raspberry pi running and returning via cloud the depth estimation map and the RGB video. This scripts uses dinov2 
# (blog: https://ai.meta.com/blog/dino-v2-computer-vision-self-supervised-learning/, paper: https://arxiv.org/abs/2304.07193).
# Given Vision Transformers and especially DINOv2 outstanding generlaization capabilities, we take the dinov2 base model provided 
# by Meta AI, add an additional attention block and multilayer perception, and while freezing the weights of the base model 
# we fine-tune the last layers with a customized dataset on a specific classification class (payload categorization based on input 
# images from the object detection script). 
# In this extended version (PylEstComp), the script also performs a comparison with the payload estimation obtained from EMG and IMU 
# signals of the Myo armband, when available (i.e., during the time window between the lift onset and offset detected by Myo). 
# In case of conflicting predictions, the Myo-based estimation is prioritized and sent to the raspberry pi.

#%% ### Set data path and import necessary libraries 
# data preprocessing section needed libraries 
import cv2
import time

# payload network needed libraries 
import os
#os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # this enables those operations of pytorch 
                                                # not supported yet for computation on GPU
                                                # to fall back to the CPU, and compute the 
                                                # rest on GPU.  (Only mac)

import numpy as np
import cv2
import time
import pandas as pd
from torchvision import transforms
from datetime import datetime
import torch
import warnings
warnings.filterwarnings("ignore") # ignore warnings 

# Libraries for real-time communication with socket
import socket

# import and build up the encoder base 
from NeuralNetwork_architectures_Pytorch import get_customizedDINOv2

# data path
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
base_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(base_dir, os.pardir))
grandparent_dir = os.path.abspath(os.path.join(base_dir, os.pardir, os.pardir))
input_dir = os.path.join(grandparent_dir, "shared", "temp") # these are the information collected and stored in the local folder by the "object detection and selection module".
model_path = os.path.join(parent_dir, "models", "CustomDino_models", "customized_DINOv2_v6grayscale.pth")
log_dir = os.path.join(parent_dir, 'data', timestamp)
log_file = os.path.join(log_dir, 'realtime_log.csv')

#%% ### Load data
# define load data function
def load_data(path_to_data): # this function checks in the output folder if the object selection and detection module detected anything and if there is a candidate loads it. 
                             # "data.csv" contians also the physical information of the box. Even though we load them too, we don't use them with DINOv2 inference. 
    gate_path = os.path.join(path_to_data, "gate.csv")
    gate_myo_path = os.path.join(path_to_data, "myo_gate.csv")
    data_path = os.path.join(path_to_data, "data.csv")
    image_path = os.path.join(path_to_data, "candidate.png")

    # Import and process image
    input = cv2.imread(image_path) # import the candidate image of the most likely obejct to be picked up by the user 
                                     # selected in the object detection and selection script 
    # Import and preprocess data
    try: 
        if os.path.exists(data_path) and os.path.exists(gate_path) and os.path.exists(gate_myo_path): # data.csv stores some physical information on the object, gate.csv and gate_myo contains a boolean variable
                                                                                                         # controlling if the model sends the actual prediction to the exoskeleton (1) or, if the lifting 
                                                                                                         # procedure is likely started already, it keeps on sending the last prediction to avoid instability
                                                                                                         # during lifting.  
            df1 = pd.read_csv(data_path) # import data 
            detected_class = df1.loc[0,'d'] # last column in the data stores the detected object label
            gate = pd.read_csv(gate_path).drop('Unnamed: 0', axis=1).loc[0,'boolean'] # extract boolean value from the gate.csv file 
            print("---------------->> Gate cv : ", gate)
            gate_myo = pd.read_csv(gate_myo_path).drop('Unnamed: 0', axis=1).loc[0,'boolean'] # extract boolean value from the gate_myo.csv file
            return input, gate, gate_myo, detected_class
        else:
            print('\nWarning: either "data.csv" or "gate.csv" or "myo_gate.csv" file not found! Skipping for now...\n')
            time.sleep(0.1)
    except pd.errors.EmptyDataError:
        print('\nWarning: either gate.csv or data.csv or "gate_myo.csv" are empty or corrupt. Skipping for now...\n')
        time.sleep(0.1) 

def display_outputs(object_class, model_output, myo_output): # once the data are received, print what's the outcoming prediction    
    if model_output == 0:
        print('The model output is ---------------->>     ' + object_class + ' < 7.5 kg')
    elif model_output == 1:
        print('The model output is ---------------->>     ' + '7.5 < ' + object_class + ' < 12.5 kg')
    else:
        print('The model output is ---------------->>     ' + object_class + ' > 12.5 kg')

    if myo_output == 0:
        print('The Myo output is ---------------->>     ' + object_class + ' < 7.5 kg')
    elif myo_output == 1:
        print('The Myo output is ---------------->>     ' + '7.5 < ' + object_class + ' < 12.5 kg')
    elif myo_output == 2:
        print('The Myo output is ---------------->>     ' + object_class + ' > 12.5 kg')
    
    if myo_output == model_output:
        print("Model and Myo output are the same! Sending Myo prediction to raspi.")
        #time.sleep(0.2)

#%% SET-UP
# import and construct the payload estimation model
# choose which DINO model you want to use as a base encoder. 
dino_model1 = 'dinov2_vitb14'
dino_model2 = 'dinov2_vits14'

# call the concatenated model constructor. The concatenated model takes the embedded images (features coming from the convolutional base output)
# and feeds them into the fully connected dense network. The output from the fully connected dense network is then concatenated with the object 
# information and fed into another little fully connected network, and consequently mapped into the classification output. 
customizedDino6 = get_customizedDINOv2(dino_model1, 3)
#print('Concatenated model summary:')
#print(concatenated_model)
print("WARNING: Check the 'CustomDino_models' folder, we are not providing the payload estimation model here, see repository details! You'll need a payload estimation model to proceed in running this code! If you already have a model instead you can skip this working.")
time.sleep(1) 
customizedDino6.load_state_dict(torch.load(model_path, map_location='xpu', weights_only=True))

#%% Pre settings (run the model on a GPU if available)
#device = 'mps' if torch.backends.mps.is_available() else 'cpu'
device = 'xpu' if torch.xpu.is_available() else 'cpu'
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('\nThe model will be tested on the following device: ', device)
customizedDino6.eval().to(device) # set the model in evlauation settings 

# Define transformations
pixels = 266
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((pixels,pixels)),  # Resize to model's input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard ImageNet normalization
])

#%% Set-up for real-time communicaiton with socket

# Define the server host and port
HOST = '192.168.0.102'
#HOST = '172.20.10.2'  # To find your mac's IP you can type the following in the terminal: ipconfig getifaddr en0
PORT = 65500           # This is the selected port, it could be any number from 0 to 65535. We choose int he range 50000-65535 to avoid conflicts. 

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))

# initialie the variables 
classification_output6 = torch.tensor(([1,0,0])).to(device)
log_dir_created = False 

# Start the while loop for real-time inference 
while True:
    if not log_dir_created: 
        os.makedirs(log_dir, exist_ok=True)
        log_dir_created = True 
        df_init = pd.DataFrame(columns=['timestamp','cv_inf_ms', 'proc_time_ms', 'gate_cv', 'gate_myo', 'response', 'model_output', 'myo_output'])
        df_init.to_csv(log_file, index=False)

    start = time.time()
    inf_time = 0
    detected_class = 'object'
    # load candidate and 3D shape generation
    try:
        image, gate, gate_myo, detected_class = load_data(input_dir)
        input_tensor = transform(image).unsqueeze(0).to(device)
        if gate_myo == 0:
            print('\nMyo says: detected offset\n')
            #time.sleep(0.1)
        elif gate_myo == 1:
            print('\nMyo says: detected onset\n')
            #time.sleep(0.1)

        myo_data_path = os.path.join(input_dir, 'myo_data.csv')
        myo_output = pd.read_csv(myo_data_path).loc[0, 'boolean']

        start_inf = time.perf_counter()
        if gate == 1:
            with torch.no_grad():
                classification_output6 = customizedDino6(input_tensor)
        model_output = classification_output6[0] 
        model_output = np.argmax(model_output.cpu().numpy())
        inf_time = round(1000*(time.perf_counter()-start_inf),3)
        now = datetime.now()
        time_ms =  now.strftime('%H_%M_%S') + f'_{int(now.microsecond / 1000):03d}'
        print(f'Timestamp: {time_ms}')
        display_outputs(detected_class, model_output, myo_output)

        response = model_output
        if myo_output != 4:
            response = myo_output
            if myo_output != model_output:
                print("Model and Myo output are different! Sending Myo prediction to raspi.")
                response = myo_output
                #time.sleep(0.2)

        # Send a response back to the client
        response = str(response)
        print('response: ' + response)
        client_socket.sendall(response.encode('utf-8'))

        processing_time = round(1000*(time.time()-start),2)
        log_entry = pd.DataFrame([{
            'timestamp': time_ms,
            'cv_inf_ms': inf_time,
            'proc_time_ms': processing_time,
            'gate_cv': gate,
            'gate_myo': gate_myo,
            'response': response,
            'model_output': model_output,
            'myo_output': myo_output,
        }])
        log_entry.to_csv(log_file, mode='a', index=False, header=False)

    except TypeError:
        print('\nWarning: No disposable data for now. Waiting...\n')
        now = datetime.now()
        time_ms =  now.strftime('%H_%M_%S') + f'_{int(now.microsecond / 1000):03d}'

        log_entry = pd.DataFrame([{
            'timestamp': time_ms,
            'cv_inf_ms': np.nan,
            'proc_time_ms': np.nan,
            'gate_cv': np.nan,
            'gate_myo': np.nan,
            'response': np.nan,
            'model_output': np.nan,
            'myo_output': np.nan,
        }])
        log_entry.to_csv(log_file, mode='a', index=False, header=False)

        time.sleep(0.1)
        continue
    
    except AttributeError:
        print('\nWarning: No disposable data for now. Waiting...\n')
        now = datetime.now()
        time_ms =  now.strftime('%H_%M_%S') + f'_{int(now.microsecond / 1000):03d}'

        log_entry = pd.DataFrame([{
            'timestamp': time_ms,
            'cv_inf_ms': np.nan,
            'proc_time_ms': np.nan,
            'gate_cv': np.nan,
            'gate_myo': np.nan,
            'response': np.nan,
            'model_output': np.nan,
            'myo_output': np.nan,
        }])
        log_entry.to_csv(log_file, mode='a', index=False, header=False)

        time.sleep(0.1)
        continue

    except pd.errors.EmptyDataError:
        print('\nWarning: No disposable data for now. Waiting...\n')
        now = datetime.now()
        time_ms =  now.strftime('%H_%M_%S') + f'_{int(now.microsecond / 1000):03d}'

        log_entry = pd.DataFrame([{
            'timestamp': time_ms,
            'cv_inf_ms': np.nan,
            'proc_time_ms': np.nan,
            'gate_cv': np.nan,
            'gate_myo': np.nan,
            'response': np.nan,
            'model_output': np.nan,
            'myo_output': np.nan,
        }])
        log_entry.to_csv(log_file, mode='a', index=False, header=False)

        time.sleep(0.1)
        continue
    
    print('\n\nProcessing time: ', processing_time, 'ms')