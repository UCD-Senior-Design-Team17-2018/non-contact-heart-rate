# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 18:51:04 2018

@author: Bijta
"""

import numpy as np
import cv2
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#connecting with the captured video file taken from mobile
cap = cv2.VideoCapture(0)

#getting the number of frames 
no_of_frames = 300

#assigning an initial zero red value for every frame
red_plane = np.zeros(no_of_frames)

#time_list is used for storing occurence time of each frame in the video  
time_list=[]
t=0

#camera frame per second is 30 and so each frame acccurs after 1/30th second
difference = 1/30
for i in range(no_of_frames):

    #reading the frame
    ret,frame = cap.read()
    length,width,channels = frame.shape

    #calculating average red value in the frame
    red_plane[i] = np.sum(frame[:,:,2])/(length*width)
    time_list.append(t)
    t = t+ difference
    cv2.imshow('frame',frame)
    cv2.waitKey(1)

cap.release()
b, a = signal.butter(4, [0.5/15, 1.6/15], btype='band')
red_plane = StandardScaler().fit_transform(np.array(red_plane.reshape(-1, 1)))
red_plane = red_plane.reshape(1, -1)[0]
red_plane = signal.lfilter(b, a, red_plane)
f, Px_ssd = signal.welch(red_plane,30)
plt.plot(f,Px_ssd)