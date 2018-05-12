# -*- coding: utf-8 -*-
"""
Created on Sat May 12 01:09:13 2018

@author: Bijta
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 13:55:28 2018

@author: Bijta
"""

import cv2
import datetime
import numpy as np
from scipy import signal
from scipy.fftpack import fft, fftfreq, fftshift
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# Change these variables based on the location of your cloned, local repositories on your computer
PATH_TO_HAAR_CASCADES = "C:/Users/Bijta/Documents/GitHub/non-contact-heart-rate/video_analysis/" 
face_cascade = cv2.CascadeClassifier(PATH_TO_HAAR_CASCADES+'haarcascade_frontalface_default.xml') # Full pathway must be used
firstFrame = None
G = []
# open video file
cap = cv2.VideoCapture("C:\\Users\\Bijta\\Documents\\GitHub\\non-contact-heart-rate\\video_analysis\\test\\DSC_0009.mov")
if cap.isOpened() == False:
    print("Failed to open file")
frame_num = 0 # start counting the frames
#plt.ion() # interactive plotting
b = signal.firwin(100, [0.7,3], window = 'hamming', pass_zero=False,fs=29.97) #fs needs to be changed, doing in dark environment so...

while cap.isOpened(): 
    ret, frame = cap.read() # read in the frame, status
    if ret == True: # if status is true
        frame_num += 1 # count frame
        if firstFrame is None: 
            # Take first frame and find face in it
            firstFrame = frame
            #cv2.imshow("frame",firstFrame)
            old_gray = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(old_gray, 1.3, 5) # Use Viola-Jones classifier to detect face
            if faces == ():
                firstFrame = None # set firstFrame to None to try again
            else:
                for (x,y,w,h) in faces: # VJ outputs x,y, width and height
                    x2 = x+w # other side of rectangle, x
                    y2 = y+h # other side of rectangle, y
                    cv2.rectangle(firstFrame,(x,y),(x+w,y+h),(255,0,0),2) #draw rect.
                    #cv2.imshow("frame",firstFrame)
                    # Make a mask
                    VJ_mask = np.zeros_like(firstFrame) 
                    VJ_mask = cv2.rectangle(VJ_mask,(x,y),(x+w,y+h),(255,0,0),-1)
                    VJ_mask = cv2.cvtColor(VJ_mask, cv2.COLOR_BGR2GRAY)
                    break
                ROI = VJ_mask
                ROI_color = cv2.bitwise_and(ROI,ROI,mask=VJ_mask) 
                #cv2.imshow('ROI',ROI_color)
                
                #take average signal in the region of interest (mask)
                _,G_new,_,_ = cv2.mean(ROI_color,mask=ROI) 
                G.append(G_new)
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        else:
            #cv2.imshow('frame',frame)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ROI_color = cv2.bitwise_and(frame,frame,mask=ROI)
            #cv2.imshow('ROI',ROI_color)
            #take average signal in the region of interest (mask)
            _,G_new,_,_ = cv2.mean(ROI_color, mask=ROI)
            #R_bg, G_bg, B_bg,_ = cv2.mean(frame,mask=np.logical_not(ROI).astype(np.uint8))
           # R_new = R_new - R_bg
           # G_new = G_new - G_bg
           # B_new = B_new - B_bg
            G.append(G_new)
            if frame_num >= 900: # when 900 frames collected, start calculating heart rate (sliding window)
                if (frame_num-900) % 1 == 0: # after every 1 frame, calculate heart rate using the data in the sliding window
                    N = 800
                    G_std = signal.detrend(G[-(N-1):])
                    T = 1/29.97
                    X_f = signal.lfilter(b, 1, G_std) 
                    #N = len(np.pad(X_f[1],(0,1024),'constant'))
                    #yf = fft(np.pad(X_f[1],(0,1024),'constant'))
                    N = len(X_f)
                    yf = fft(X_f)
                    yf = yf/np.sqrt(N) #Normalize FFT
                    xf = fftfreq(N, T) # FFT frequencies 
                    xf = fftshift(xf) #FFT shift
                    yplot = fftshift(abs(yf))
                    plt.figure(1)
                    plt.gcf().clear()
                    fft_plot = yplot
                    # Find highest peak between 0.75 and 4 Hz 
                 #   fft_plot[xf<=0.75] = 0 
                    if frame_num == 900:
                        bpm = xf[(xf>=0.75) & (xf<=4)][fft_plot[(xf>=0.75) & (xf<=4)].argmax()]*60
                    else:
                        bpm = 0.9*bpm + 0.1*(xf[(xf>=0.75) & (xf<=4)][fft_plot[(xf>=0.75) & (xf<=4)].argmax()]*60)
                    print(str(bpm)+' bpm') # Print heart rate
                    plt.plot(xf[(xf>=0.75) & (xf<=4)], fft_plot[(xf>=0.75) & (xf<=4)]) # Plot FFT
                    plt.pause(0.001)
            if frame_num % 10 == 0:
                print(frame_num)
                
            if frame_num == 1000:
                cap.release()   