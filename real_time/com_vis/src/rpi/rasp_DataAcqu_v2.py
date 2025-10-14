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
# This code's aim is to send data from the raspberry pi to the server computer, which will process them for improved performances.
# This code uses socket to create a bridge between the raspberry pi and the cloud, and send the images acquired by the raspberry pi
# to the local pc (mac) for processing.

# Define the server host and port
HOST = '192.168.0.100'  # when using the wifi router
#HOST = '172.20.10.13'  # To find your mac's IP you can type the following in the terminal: ipconfig getifaddr en0
PORT = 65400            # This is the selected port, it could be any number from 0 to 65535. We choose it in the range
                        # 50000-65535 to avoid conflicts.

# import libraries 
from numpysocket import NumpySocket
import cv2
import time
import numpy as np
from preview_depth import run_depth_sensor

frame_duration = 0.1  # minimum time interval between sending consecutive frames (controls the sending rate)
last_send_time = 0    # timestamp of the last frame sent

with NumpySocket() as s:
    s.connect((HOST, PORT))
    #while cap.isOpened():
    for depth_map,frame_resize in run_depth_sensor():
                start_time = time.time()       
                # skip this iteration if the time since the last sent frame is less than frame_duration
                # ensures that frames are not sent too frequently, controlling the sending rate
                if start_time - last_send_time < frame_duration: 
                        continue

                # Here the data acquisition from the depth sensor is happening (it'll give values between 0 and 4 [m]).
                # The data from the depth sensor will be converted from the [0-4]m float scale to the [0-255] integer scale
                # because the array will be consequently transformed into a uint8 numpy array. Each number in a uint8 numpy
                # array can only assume an integer value between 0 and 255. We are doing this because sending via cloud a
                # np.uint8 array is much less computationally expensive than other formats, and it enables us to increase
                # the image quality (allowing us to send more pixels). In the end, since we are mapping vlaues contained in
                # the [0,4]m range into the range [0,255], there are 256 numbers we can describe in the range, hence, our
                # conversion resolution will be 4m/256 approx 0.016m, which is approx 1.6cm. Since the sensor measurement
                # accuracy is 2cm, we're not throwing away any information, we're instead sending the smallest information
                # variation the sensor is able to capture.
                depth_map = 255 * depth_map / 4 # convert in range [0,255]
                depth_map[depth_map > 255] = 255 # all the vlaues greather than 255 (greather than 4m) are set to max measurable distance (4m)

                depth_map = cv2.resize(depth_map, (frame_resize.shape[1], frame_resize.shape[0])) # resize distance map
                data = np.concatenate((frame_resize,depth_map), axis=1) # concatenate distance map and confidence buffer
                data = np.array(data, dtype=np.uint8) # convert both the image and the depth map into a np.uint8 array
                print('Final data resolution: ', data.shape) # display the final shape fo the data which will be sent
                                
                try:
                    s.sendall(data) # send the images via socket cloud
                    last_send_time = time.time()
                    print('Processing time --------> ', round(time.time()-start_time,2)) # display processing time
                except Exception:
                    print('Unexpected error while trying to send the images over the cloud. Aborting...')
