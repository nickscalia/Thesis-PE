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
# This code's aim is to receive data from the cloud pc, collect the final results, and print the results in the terminal
# This code uses socket to create a bridge between the raspberry pi and the mac, and receive the results computed by the cloud pc.

# %% START THE SCRIPT
print('----------------------------------------- INITIALIZATION -----------------------------------------')
# %% START THE REAL-TIME CODE 

# Needed libraries for socket processing
import socket
import struct 
import serial
import time

# Define useful functions

# This function displays the prediction of the pipeline based on the results received from the cloud
def display_outputs(input):
    if input == 0:
        print('--------------------------->> model output: x < 7.5 kg')
    elif input == 1:
        print('--------------------------->> model output: 7.5 < x < 12.5 kg')
    else:
        print('--------------------------->> model output: x > 12.5 kg')

# Define the server host and port
HOST = '192.168.52.43'
#HOST = '172.20.10.2'  # To find your mac's IP you can type the following in the terminal: ipconfig getifaddr en0
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
    print('\n\n-------->> STARTING TO RECEIVE THE DATA:')
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