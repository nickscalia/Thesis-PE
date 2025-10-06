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
# This script captures video from a webcam, converts it to grayscale, and streams it via MJPEG over HTTP.
# Uses OpenCV for frame capture and processing, and http.server to serve frames to clients.
# Controls the frame rate with frame_duration and sends each frame as a multipart JPEG image.

# import libraries 
import cv2
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

# configuration
HOST = '192.168.52.43' 
PORT = 8000
DEVICE = 1
frame_duration = 0.1

# initialize webcam
cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FPS, 10)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))

# define HTTP request handler
class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != '/':
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, jpeg = cv2.imencode('.jpg', gray)
            if not ret:
                continue
            self.wfile.write(b"--frame\r\n")
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(jpeg.tobytes())))
            self.end_headers()
            self.wfile.write(jpeg.tobytes())
            self.wfile.write(b'\r\n')
            
            elapsed = time.time() - start_time
            if elapsed < frame_duration:
                time.sleep(frame_duration - elapsed)

# start HTTP server
server = HTTPServer(('', PORT), MJPEGHandler)
print(f"MJPEG server listening on http://{HOST}:{PORT}")
server.serve_forever()