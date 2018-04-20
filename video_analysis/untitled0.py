# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 12:39:55 2018

@author: Bijta
"""

import cv2
import datetime
import numpy as np

firstFrame = None
time = []
cap = cv2.VideoCapture(0) # open webcam
cap.set(cv2.cv2.CAP_PROP_FPS, 10)
cap.set(15, 0.1)
if cap.isOpened() == False:
    print("Failed to open webcam")
frame_num = 0 # start counting the frames
while cap.isOpened(): 
    ret, frame = cap.read() # read in the frame, status
    if ret == True: # if status is true
        frame_num += 1 # count frame
        try:
            if firstFrame is None: 
                firstFrame = frame
                start = datetime.datetime.now() # start time
                time.append(0)
                cv2.imshow('frame',firstFrame)
                cv2.waitKey(1)
            else:
                current = datetime.datetime.now() - start
                cv2.imshow("frame",frame)
                current = current.total_seconds()
                time.append(current)
                cv2.waitKey(1)
        except KeyboardInterrupt:
            cap.release()
            
            