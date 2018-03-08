# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 21:53:18 2018

@author: Bijta
"""

import cv2
import numpy as np

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
video = cv2.VideoCapture() 

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")