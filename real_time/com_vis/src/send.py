# -----------------------------------------------------------------------------
# Copyright (c) 2025 Andrea Dal Prete - Politecnico di Milano
# All rights reserved.
#
# This script is part of the research published in:
# [Your Paper Title], [Conference/Journal Name], [Year]
# DOI: [Insert DOI if available]
#
# Author: Andrea Dal Prete (andrea.dalprete@polimi.it)
# -----------------------------------------------------------------------------

# CODE EXPLAINATION
# this script uses the environment "ObjectDetection_env.yml"
#%% # Introduction

# This script contains the payload estimation module for boxes weight classification. This script is meant to be run 
# in parallel to a raspberry pi running and returning via cloud the depth estimation map. This scripts uses dinov2 
# (blog: https://ai.meta.com/blog/dino-v2-computer-vision-self-supervised-learning/, paper: https://arxiv.org/abs/2304.07193).
# Given Vision Transformers and especially DINOv2 outstanding generlaization capabilities, we take the dinov2 base model provided 
# by Meta AI, add an additional attention block and multilayer perception, and while freezing the weights of the base model 
# we fine-tune the last layers with a customized dataset on a specific classification class (payload categorization based on input 
# images from the object detection script). 

#%% ### Set data path and import necessary libraries 
# data path
image_path = "candidate.png" # these are the information collected and stored in the local folder by the "object detection and selection module".
data_path = 'data.csv' # these are the information collected and stored in the local folder by the "object detection and selection module".
# data preprocessing section needed libraries 
import cv2
import time

# payload network needed libraries 
import os
#os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # this enables those operations of pytorch 
                                                # not supported yet for computation on GPU
                                                # to fall back to the CPU, and compute the 
                                                # rest on GPU. 

import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import pandas as pd
from torchvision import transforms
import torch
import warnings
import random
warnings.filterwarnings("ignore") # ignore warnings 

# Libraries for real-time communication with socket
import socket

# import and build up the encoder base 
from NeuralNetwork_architectures_Pytorch import get_customizedDINOv2

#%% ### Load data

# define load data function
def load_data(path_to_data): # this function checks in the local folder if the object selection and detection module detected anything and if there is a candidate loads it. 
                             # "data.csv" contians also the physical information of the box. Even though we load them too, we don't use them with DINOv2 inference. 
    # Import and process image
    input = cv2.imread(path_to_data) # import the candidate image of the most likely obejct to be picked up by the user 
                                     # selected in the object detection and selection script 
    # Import and preprocess data
    try: 
        if os.path.exists('data.csv') and os.path.exists('gate.csv') and os.path.exists('gate_myo.csv'): # data.csv stores some physical information on the object, gate.csv contains a boolean variable 
                                                                      # controlling if the model sends the actual prediction to the exoskeleton (1) or, if the lifting 
                                                                      # procedure is likely started already, it keeps on sending the last prediction to avoid instability
                                                                      # during lifting.  
            df1 = pd.read_csv('data.csv') # import data 
            detected_class = df1.loc[0,'d'] # last column in the data stores the detected object label
            gate = pd.read_csv('gate.csv').drop('Unnamed: 0', axis=1).loc[0,'boolean'] # extract boolean vlaue from the gate.csv file 
            gate_myo = pd.read_csv('gate_myo.csv').drop('Unnamed: 0', axis=1).loc[0,'boolean']
            return input, gate, gate_myo, detected_class
        else:
            print('\nWarning: either "data.csv" or "gate.csv" or "gate_myo.csv" file not found! Skipping for now...\n')
            time.sleep(1.5)
    except pd.errors.EmptyDataError:
        print('\nWarning: either gate.csv or data.csv are empty or corrupt. Skipping for now...\n')
        time.sleep(1.5) 

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
        print("Model and Myo output are the same!")
        time.sleep(0.3)

#%% SET-UP
# import and construct the payload estimation model
# choose which DINO model you want to use as a base encoder. 
dino_model1 = 'dinov2_vitb14'
dino_model2 = 'dinov2_vits14'

# call the concatenated model constructor. The concatenated model takes the embedded images (features coming from the convolutional base output)
# and feeds them into the fully connected dense network. The output from the fully connected dense network is then concatenated with the object 
# information and fed into another little fully connected network, and consequently mapped into the classification output. 
customizedDino6 = get_customizedDINOv2(dino_model2, 3)
#print('Concatenated model summary:')
#print(concatenated_model)
print("WARNING: Check the 'CustomDino_models' folder, we are not providing the payload estimation model here, see repository details! You'll need a payload estimation model to proceed in running this code! If you already have a model instead you can skip this working.")
time.sleep(1.5) 
customizedDino6.load_state_dict(torch.load('CustomDino_models/customized_DINOv2_v6grayscale_small.pth', map_location='xpu', weights_only=True))

#%% Pre settings (run the model on a GPU if available)
#device = 'mps' if torch.backends.mps.is_available() else 'cpu'
device = 'xpu' if torch.xpu.is_available() else 'cpu'
print(device)
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
HOST = '192.168.52.43'
#HOST = '172.20.10.2'  # To find your mac's IP you can type the following in the terminal: ipconfig getifaddr en0
PORT = 65500           # This is the selected port, it could be any number from 0 to 65535. We choose int he range 50000-65535 to avoid conflicts. 

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))

# initialie the variables 
classification_output6 = torch.tensor(([1,0,0])).to(device)

# Start the while loop for real-time inference 
while True:
    start = time.time()
    detected_class = 'object'
    # load candidate and 3D shape generation
    try:
        image, gate, gate_myo, detected_class = load_data(image_path)
        input_tensor = transform(image).unsqueeze(0).to(device)
        if gate_myo == 0:
            print('\nMyo says: detected offset\n')
            time.sleep(1.5)
        elif gate_myo == 1:
            print('\nMyo says: detected onset\n')
            time.sleep(1)

        myo_output = pd.read_csv('myo_data.csv').loc[0, 'boolean']


        if gate == 1:
            with torch.no_grad():
                classification_output6 = customizedDino6(input_tensor)
        model_output = classification_output6[0] 
        model_output = np.argmax(model_output.cpu().numpy())
        #model_output = random.randint(0, 2) 
        display_outputs(detected_class, model_output, myo_output)

        if myo_output != 4 and myo_output  != model_output:
            print("Model and Myo output are different! Sending Myo prediction to exo.")
            model_output = myo_output
            time.sleep(0.3)

        # Send a response back to the client
        response = str(model_output)
        print('response: ' + response)
        client_socket.sendall(response.encode('utf-8'))
        plt.pause(0.01)
    except TypeError:
        print('\nWarning: No disposable data for now. Waiting...\n')
        time.sleep(1.5)
        continue
    except AttributeError:
        print('\nWarning: No disposable data for now. Waiting...\n')
        time.sleep(1.5)
        continue
    print('\n\nProcessing time: ', round(1000*(time.time()-start),2), 'ms')
    