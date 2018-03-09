# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 21:53:18 2018

@author: Bijta
"""

import cv2
import numpy as np

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
video = cv2.VideoCapture(0) 

# Check if camera opened successfully
if (video.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
while(video.isOpened()):
  # Capture frame-by-frame
  ret, frame = video.read()
  if ret == True:
 
    # Display the resulting frame
    cv2.imshow('Frame',frame)
 
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
video.release()
 
# C loses all the frames
cv2.destroyAllWindows()