#%% CODE EXPLAINATION
# This code's aim is to receive data from the cloud pc, collect the final resutls, and send to the back exo microcontroller the support
# level modulation. This code uses socket to create a bridge between the raspberry pi and the mac, and receive the results computed by the cloud pc.


# This script uses the environment "ObjectDetection_env.yml"

# %% START THE SCRIPT
print('----------------------------------------- INITIALIZATION -----------------------------------------')
# %% START THE REAL-TIME CODE 

# Needed libraries for socket processing
import socket
import sys
import cv2
import pickle
import numpy as np
import struct ## new
import zlib
from datetime import datetime
import serial
import time
import pandas as pd

# Define useful functions

# This function displays the prediction of the pipeline dased on the results received from the cloud
def display_outputs(input):
    if input == 0:
        print('--------------------------->> model output: x < 7.5 kg')
    elif input == 1:
        print('--------------------------->> model output: 7.5 < x < 12.5 kg')
    else:
        print('--------------------------->> model output: x > 12.5 kg')

# This function is to send data through the serial port
def send_to_serial(ser,input_):
    message = 's'
    if input_ == 1:
        message = 'm' # 'm' stands for medium assistance
    if input_ == 2:
        message = 'l' # 'l' stands for large/strong assistance
    else:
        message = 's' # 's' stands for small/light assistance
    try:
        ser.write(message.encode('utf-8')) # sends the message throught the serial port
        print(f"Sent to the serial port: {message}") # prints what was sent through theserial port
        return True # this is what will be sotred inside the 'condition' variable
    except Exception as e:
        print('Warning: Unexpected error while sending the data to the serial port. Data were not sent...')
        print(f"Encountered this error while trying to send the data:\n\n {e}\n")

# Define the server host and port
#HOST = '172.20.10.2'  # To find your mac's IP you can type the following in the terminal: ipconfig getifaddr en0
HOST = '192.168.52.43'
PORT = 65500          # This is the selected port, it could be any number from 0 to 65535. We choose int he range 50000-65535 to avoid conflicts. 

# Configuration for the cloud bridge
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('Socket created')
s.bind((HOST,PORT))
print('Socket bind complete')
s.listen(3)
print('Socket now listening...')

conn,addr=s.accept()

data = b""
payload_size = struct.calcsize(">L")
print("payload_size: {}".format(payload_size))

# define necessary variables for data collection
time_stamps  = []
dino_outputs = []

try:
    print('\n\n-------->> STARTING TO RECEIVE AND SEND THE DATA:')
    start_time = time.time()
    while True:
        # receive and process data from the cloud
        response = conn.recv(1024)  # Assuming the response is small (e.g., string)
        display_outputs(int(response.decode('utf-8')[-1]))
        time_stamps.append(round(time.time() - start_time,3))
        # print some stuff
        print('--------------------------->> time stamp:  ', time_stamps[-1], ' s')
        time.sleep(0.1)
except serial.SerialException as e:
    print(f"Error establishing serial port communications...: {e}")




