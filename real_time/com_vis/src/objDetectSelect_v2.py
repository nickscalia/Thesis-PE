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

#%% CODE EXPLAINATION 
# This code's aim is to receive data from the raspberry pi, and peform object detection, isolation, and environment comprehension. 
# This code uses socket to create a bridge between the raspberry pi and the laptop, and receive the images acquired by the raspberry pi 
# to the local pc for processing. 

# N.B.: You will need a camera connected to the macbook to run this code or to the raspberry pi.

# %% START THE SCRIPT
print('----------------------------------------- INITIALIZATION -----------------------------------------')
#%% IMPRORT THE NECESSARY LIBRARIES & DEFINE USEFUL FUNCTIONS

# Video-obects recognition algorithm needed libraries
import os
#os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # this enables those operations of pytorch 
                                                # not supported yet for computation on GPU
                                                # to fall back to the CPU, and compute the 
                                                # rest on GPU.  (Only mac)
                                                
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import time 
import torch
from threading import Thread

# Define output directory
base_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(base_dir, os.pardir))
grandparent_dir = os.path.abspath(os.path.join(base_dir, os.pardir, os.pardir))
output_dir = os.path.join(grandparent_dir, "shared", "temp")
gate_path = os.path.join(output_dir, "gate.csv")
data_path = os.path.join(output_dir, "data.csv")
image_path = os.path.join(output_dir, "candidate.png")

# Define useful functions for stereovision
def distance_decay_factor(d): # exponential decay factors for object selection (see paper for more info)
    lamb = 0.5 # decay factor
    print('Decay distance: ', np.exp(-lamb*d))
    return np.exp(-lamb*d)

def angle_decay_factor(x,width_image): # exponential decay factors for object selection (see paper for more info)
    lamb = 0.1
    theta = 180 * np.arctan((x - width_image/2) / f) / np.pi
    print('Decay theta: ', np.exp(-lamb*abs(theta)))
    return np.exp(-lamb*abs(theta))

def softmax_probability(probabilities): # softmax final probability evlauation for object selection (see paper for more info)
    """Compute the softmax of a list or numpy array."""
    # Convert the input to a numpy array if it's not already
    probabilities = np.array(probabilities)
    # Subtract the max value from the input to prevent overflow (for numerical stability)
    exp_p = np.exp(probabilities - np.max(probabilities))
    # Compute and return softmax values
    return exp_p / exp_p.sum()

# function to process the results obtained from YOLO  
def process_results(results, classNames, frame, disparity_map_plasma, distance_map, f, conf_thresholds, width_image):
    # initialize varibales 
    final_probabilities = [] # this list will store the final softmax probabilities of each object/box in the image  
    PositionsAndDimensions = [] # this list will store the coordinate (related to where the boxes are) of where to put the text in the image 
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract bounding boxes and confidence in one step
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert coordinates to integers
            confidence = round(float(box.conf[0]), 2) # Rounding the confidence
            cls = int(box.cls[0])
            class_name = classNames[cls]
            # Process only those results with a higher confidence than the threhsold
            if (class_name == 'box' and confidence > conf_thresholds['box']):
                # Crop the distance map (in the are where an object was detected) once and reuse it
                object_area = distance_map[y1:y2, x1:x2]
                # Calculate the dimensions once
                d = round(np.median(object_area),3) # calculate the median distance (within the distance area) of the object
                x_dim, y_dim = d * (x2-x1) / f, d * (y2-y1) / f # calculate the x and y dimensions of the frontal view of the object 
                # Compute overall pick-up probability and add it to a list to be used later 
                final_probabilities.append(10*angle_decay_factor((x1+x2)/2,width_image)*distance_decay_factor(d)) # collect the probability for a specific object 
                                                                                                                  # to be picked up based on where it is located 
                                                                                                                  # in the environment. 
                PositionsAndDimensions.append([round(x1,2),round(y1,2),round(x2,2),round(y2,2),round(x_dim,2),round(y_dim,2),round(d,2),class_name])
                # Draw rectangles and put text on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.rectangle(disparity_map_plasma, (x1, y1), (x2, y2), color, 3)
                # Prepare text for the frame
                class_text = f'Cl: {class_name}, Conf: {confidence}'
                distance_text = 'd: ' + str(d) + ' m' #, x: {round(x_dim, 2)} m, y: {round(y_dim, 2)} m' (add this if you want to display the object estimated dimensions)
                # Apply text to the frame in one operation
                cv2.putText(frame, class_text, (x1, y1-70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(frame, distance_text, (x1, y1-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return final_probabilities, PositionsAndDimensions

# this function contains the whole object detection and selection code. This function now takes as input the frame and returns the object detection 
# and candidate selection, saving the results in the local folder. 
def ObjectDetectionAndIsolation(frame, width):
    width_image = int(frame.shape[1]/2) 
    distance_map = frame[:,width_image:-1].astype(np.float32) / (255 / 4) # go back from image to distance data up to 4 meters
    frame = frame[:,0:width_image] # separate the image from the depth map
    # Initialise the variable that will be returned
    final_class = 'nothing'
    # START PROCESSING
    Start = time.time() # keep track of inference time 
    frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB) # go back to rgb images from grayscale 
    frame2 = frame.copy() # make a copy of the environment image for later use
    # Normalize and modify distance map
    distance_map_normalized = cv2.normalize(distance_map, None, 0, 255, cv2.NORM_MINMAX)
    distance_map_8bit = distance_map_normalized.astype(np.uint8) # make sure it is np.uint8 type 
    disparity_map_plasma = cv2.applyColorMap(distance_map_8bit, cv2.COLORMAP_PLASMA) # this is for later displaying 
    # YOLO ALGORITHM - Boxes detection in the environment 
    # Detect objects
    results1 = model(frame, stream=True, device=device, imgsz=width)
    # Define confidence thresholds for specific classes
    conf_thresholds_model = {'box': 0.75} # can select the minimum confidence to consider the yolo detection
    # Process results sequentially (as required)
    final_probabilities, PositionsAndDimensions = process_results(results1, classNames_model, frame, disparity_map_plasma, distance_map, f, conf_thresholds_model, width_image)
    # Print the final softmax probabilities on the image 
    if np.array(final_probabilities).any(): # if at least one object was detected in the environment, do the rest
        final_probabilities = 100*softmax_probability(final_probabilities) # preform softmax normalization of the 'likelyhood parameters' 
        for i in range(0,len(final_probabilities)): # for cycle to display on the frame the likelyhood of each detected box to be picked up by the user
            probability_text = f'p: {round(final_probabilities[i], 2)} %'
            cv2.putText(frame, probability_text, (PositionsAndDimensions[i][0], PositionsAndDimensions[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        # Select the object with the highest likelyhood to be picked up, and extract its properties 
        which = np.argmax(final_probabilities)
        x1, y1, x2, y2 = PositionsAndDimensions[which][0], PositionsAndDimensions[which][1], PositionsAndDimensions[which][2], PositionsAndDimensions[which][3]
        shift = max(y2 - y1, x2 - x1)
        candidate_image = frame2[y1:y1 + shift, x1:x1 + shift] # crop the image of the object you selected as the most likely to be picked up by the user 
        # save object properties 
        pd.DataFrame({'a':[PositionsAndDimensions[which][4]], 'b':[PositionsAndDimensions[which][5]], 'c':[PositionsAndDimensions[which][6]], 'd':[PositionsAndDimensions[which][7]]}).to_csv(data_path)
        # save object image (for later payload estimation)
        cv2.imwrite(image_path, candidate_image)
        # deifne the gate: if the selected object is within the Xm range, close the gate. That is, keep sending the previous payload estimation to avoid instability while lifting
        if PositionsAndDimensions[which][6] > 0.1:
            pd.DataFrame({'boolean':[1]}).to_csv(gate_path)
        else:
            pd.DataFrame({'boolean':[0]}).to_csv(gate_path)
        final_class = PositionsAndDimensions[which][7] # which in this script will always be 'box'
    print('------------------------------------------------------------> Processing time: ' + str(round(time.time()-Start,2)) + '[s]')
    # plot the results
    final_plot = cv2.vconcat((frame[:,0:-1,:], disparity_map_plasma))
    orig_height, orig_width = final_plot.shape[:2]
    # reduce plot dimensions
    new_width = orig_width // 2
    new_height = orig_height // 2
    small_plot = cv2.resize(final_plot, (new_width, new_height), interpolation=cv2.INTER_AREA)
    cv2.imshow('Recognition with depth', small_plot)
    #cv2.imshow('Recognition with depth', final_plot)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        exit(1)
    
    return final_class

#%% GENERAL SET-UP 
# SET-UP STEREO CAMERA PARAMETERS 

# The Focal Length is around: 528 pixels. And the Baseline is around: 
f = 528 # [pxl] focal length 

# SET-UP THE YOU ONLY LOOK ONCE(YOLO) ALGOTIRHM (Version-8)
print('----> YOLO-v11 model general set-up')
# YOLO model 8 set-up weights 
color = (0, 65, 255)
#device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(base_dir, "..", "models", "yolo-Weights", "yolov11_new.pt")
model = YOLO(weights_path)

print('---- CLASSES NAMES ----')
print('The following classes are the ones contained in the YOLO-v11 fine-tuned model:\n', model.names)

# object classes
classNames_model = ["box"]

video_source = "rasp"  # "rasp" or "phone"
print('----> Video source:', video_source)
print('----> Device:', device)
print('----> Initialization done.\n\n')

# SET-UP OBJECTS PROBABILITIES ESTIMATION 

# %% START THE REAL-TIME CODE 

import logging
from numpysocket import NumpySocket

# Define the server host and port
HOST = '192.168.52.175'
#HOST = '172.20.10.13'  # To find your macbook IP you can type the following in the terminal: ipconfig getifaddr en0
PORT = 65400            # This is the selected port, it could be any number from 0 to 65535. We choose in the range 50000-65535 to avoid conflicts. 

logger = logging.getLogger("OpenCV server")
logger.setLevel(logging.INFO)

final_class = 'nothing'

# Open a cloud bridge to connect this cloud pc with the rapsberry pi and exchange data over the cloud 
with NumpySocket() as s:
    # Open the connection 
    s.bind((HOST, PORT))
    print('Ready for connection. Waiting...')

    if video_source == "rasp":
        url = "http://192.168.52.43:8000/"  # IP del Pi
        cap = cv2.VideoCapture(url)
        last_frame = None

        # Continuously grabs frames from the video stream and save only the latest frame
        def grab_thread():
            global last_frame
            while True:
                ret, frame = cap.read()
                if ret:
                    last_frame = frame
        Thread(target=grab_thread, daemon=True).start()

    elif video_source == "phone":     
        cap = cv2.VideoCapture(0) # start video capture 
            
    if not cap.isOpened():
        print("Error opening the stream")
        exit()

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Current resolution: {int(width)} x {int(height)}") 
    
    while True:
        try:
            s.listen() # listen for someone who's willing to connect until you make it 
            conn, addr = s.accept() # accept the connection 
            logger.info(f"connected: {addr}") 
            while conn: # if and until the connection is true (which means you are connected over the cloud) do
                if video_source == "rasp":
                    if last_frame is None:
                        continue
                    else:
                        video_frame = last_frame.copy()
                elif video_source == "phone":
                    ret, video_frame = cap.read() # take an image form the phone camera (it couold be any general remote camera)
                    if not ret: # if you didn't manage to collect the image break  
                        print('Error: issues encountered while trying to collect an image. Check embedded camera connection...')
                        break
                    
                video_frame = cv2.cvtColor(video_frame,cv2.COLOR_RGB2GRAY) # ocnvert to gray for more standard and general processing 
                #video_frame = cv2.resize(video_frame, (int(video_frame.shape[1]/2),int(video_frame.shape[0]/2))) # reduce image size if high-quality camera 
                frame = conn.recv(86400) # the data you receive here are sent from raspberry pi in np.uint8. If you transfer data in np.uint8 it's hundreds of times faster
                print("Packet received: ", frame.nbytes)
                if len(frame) == 0: # if you didn't manage to collect any data, break 
                    print('Error: issues encountered while trying to collect sensor depth map. Check cloud connection/depth sensor data integrity...')
                    break
                # Here you receiving the depth map ('frame') in [0-255] scale, which should be mapped back to [0-4m]. Since from 0 to 255 there are 256 numbers (which 
                # you can completely describe with np.uint8 array type), to each one you can relate a number in the range [0-4]. Hence, you have a resolution of 4/256 approx 
                # 0.016m. This is close to the sensor accuracy and resolution, therefore, we are completely exploiting the sensor capabilities without discarding any information.    
                frame = np.array(frame, dtype=np.uint8) # make sure what you receive is an array of np.uint8 (this is the format in which you are receiving the frame)
                width_image = int(frame.shape[1]/2) # indeed what you are receiving is a "double frame", where the left side is a depth map and the right side is a confidence buffer (this is the output of the sensor)
                confid_buf = frame[:,0:width_image] # divide confidence buffer from the depth map
                # since the emebdded camera and the depth sensor have different "equivalent lenses" and field of viewes, you'll need to crop a bit the depth map to fit it 
                # into the camera field of view.
                sensor_depth = frame[:,width_image:-1][:, :-20] # crop a bit the sensor depth to make sure it aligns wiht the image taken from the embedded camera
                frame = np.concatenate((video_frame, cv2.resize(sensor_depth, (video_frame.shape[1],video_frame.shape[0]))), axis=1) # concatenate image and received depth map
                # Here the frame is passed to a series fo functions which process the data to: reocgnize boxes in the environment, identify the most likely one to be picked 
                # up, and save the data relative to that object in a local folder where another script will have access to to read the data and run payload estimation inference. 
                final_class = ObjectDetectionAndIsolation(frame, width) # process the data
                # Press Q on keyboard to exit
                if cv2.waitKey(1) & 0xFF == ord("q"): # press "Q" key on the keyboar to exit the process at anytime
                    exit(1)
            logger.info(f"disconnected: {addr}")
        except ConnectionResetError:
            pass
