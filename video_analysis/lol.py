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
import os

'''
IMPORTANT: All changes made not by team has "NEW" comment.
'''

# NEW gets the current directory this file is in
# The filename
HAARCLASSIFIER = 'haarcascade_frontalface_default.xml'

# cap is used to check if there is webcam
# and run program while video is on

# frame_num is only used per iteration of while loop
# and in second if statement
'''
while cap.isOpened(): 
    ret, frame = cap.read() # read in the frame, status
    if ret == True: # if status is true
        frame_num += 1 # count frame
        
        # RGB used only - must be at least here (both if statements)
        # firstFrame used only for first if statements
        # pca is only used in second if-else statement 
        
        if firstFrame is None: 
            start = datetime.datetime.now() # start time
            time.append(0)
            # Take first frame and find face in it
            firstFrame = frame
            cv2.imshow("frame",firstFrame)
            old_gray = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(old_gray, 1.3, 5) # Use Viola-Jones classifier to detect face
            if faces == ():
                firstFrame = None
            else:
                for (x,y,w,h) in faces: # VJ outputs x,y, width and height
                    x2 = x+w # other side of rectangle, x
                    y2 = y+h # other side of rectangle, y
                    cv2.rectangle(firstFrame,(x,y),(x+w,y+h),(255,0,0),2) #draw rect.
                    cv2.imshow("frame",firstFrame)
                    # Make a mask
                    VJ_mask = np.zeros_like(firstFrame) 
                    VJ_mask = cv2.rectangle(VJ_mask,(x,y),(x+w,y+h),(255,0,0),-1)
                    VJ_mask = cv2.cvtColor(VJ_mask, cv2.COLOR_BGR2GRAY)
                ROI = VJ_mask
                ROI_color = cv2.bitwise_and(ROI,ROI,mask=VJ_mask) 
                cv2.imshow('ROI',ROI_color)

                #take average signal in the region of interest (mask)
                R_new,G_new,B_new,_ = cv2.mean(ROI_color,mask=ROI) 
                R.append(R_new)
                G.append(G_new)
                B.append(B_new)
        
        # NEW (just a comment) checkpoint here
        # Whats under, will run after top no matter what
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        else:
            # time for the current frame
            current = datetime.datetime.now()-start # NEW (Comment only) this start?
            current = current.total_seconds()
            time.append(current)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ROI_color = cv2.bitwise_and(frame,frame,mask=ROI)
            cv2.imshow('ROI',ROI_color)
            #take average signal in the region of interest (mask)
            R_new,G_new,B_new,_ = cv2.mean(ROI_color, mask=ROI)
            R.append(R_new)
            G.append(G_new)
            B.append(B_new)
            if frame_num >= 900: # when 900 frames collected, start calculating heart rate (sliding window)
                N = 900
                #normalize RGB signals
                G_std = StandardScaler().fit_transform(np.array(G[-(N-1):]).reshape(-1, 1)) 
                G_std = G_std.reshape(1, -1)[0]
                R_std = StandardScaler().fit_transform(np.array(R[-(N-1):]).reshape(-1, 1)) 
                R_std = R_std.reshape(1, -1)[0]
                B_std = StandardScaler().fit_transform(np.array(B[-(N-1):]).reshape(-1, 1))
                B_std = B_std.reshape(1, -1)[0]
                T = 1/(len(time[-(N-1):])/(time[-1]-time[-(N-1)])) #calculate time between first and last frame (period)
                # do ICA (called PCA because originally tried PCA)
                X_f=pca.fit_transform(np.array([R_std,G_std,B_std]).transpose()).transpose() 
                b, a = signal.butter(4, [0.5/15, 1.6/15], btype='band') #Butterworth filter
                X_f = signal.lfilter(b, a, X_f) 
                N = len(X_f[0])
                yf = fft(X_f[1]) # FFT
                yf = yf/np.sqrt(N) #Normalize FFT
                xf = fftfreq(N, T) # FFT frequencies 
                xf = fftshift(xf) #FFT shift
                yplot = fftshift(abs(yf))
                plt.figure(1)
                plt.gcf().clear()
                fft_plot = yplot
                # Find highest peak between 0.75 and 4 Hz 
                fft_plot[xf<=0.75] = 0 
                print(str(xf[fft_plot[xf<=4].argmax()]*60)+' bpm') # Print heart rate
                plt.plot(xf[(xf>=0) & (xf<=4)], fft_plot[(xf>=0) & (xf<=4)]) # Plot FFT
                plt.pause(0.001)'''

# First if-statement
def first_if_statement(firstFrame, R, G, B, frame, time, start, ROI):
    start = datetime.datetime.now() # start time
    time.append(0)
    # Take first frame and find face in it
    firstFrame = frame
    cv2.imshow("frame",firstFrame)
    old_gray = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(old_gray, 1.3, 5) # Use Viola-Jones classifier to detect face
    if faces == ():
        firstFrame = None
    else:
        for (x,y,w,h) in faces: # VJ outputs x,y, width and height
            x2 = x+w # other side of rectangle, x
            y2 = y+h # other side of rectangle, y
            cv2.rectangle(firstFrame,(x,y),(x+w,y+h),(255,0,0),2) #draw rect.
            cv2.imshow("frame",firstFrame)
            # Make a mask
            VJ_mask = np.zeros_like(firstFrame) 
            VJ_mask = cv2.rectangle(VJ_mask,(x,y),(x+w,y+h),(255,0,0),-1)
            VJ_mask = cv2.cvtColor(VJ_mask, cv2.COLOR_BGR2GRAY)
        ROI = VJ_mask
        ROI_color = cv2.bitwise_and(ROI,ROI,mask=VJ_mask) 
        cv2.imshow('ROI',ROI_color)

        #take average signal in the region of interest (mask)
        R_new,G_new,B_new,_ = cv2.mean(ROI_color,mask=ROI) 
        R.append(R_new)
        G.append(G_new)
        B.append(B_new)
    return firstFrame, R, G, B, frame, time, start, ROI


# That else-statement
def that_else_statement(R, G, B, frame, time, frame_num, start, pca, ROI):
    # time for the current frame
    current = datetime.datetime.now()-start # NEW (Comment only) this start?
    current = current.total_seconds()
    time.append(current)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ROI_color = cv2.bitwise_and(frame,frame,mask=ROI)
    cv2.imshow('ROI',ROI_color)
    #take average signal in the region of interest (mask)
    R_new,G_new,B_new,_ = cv2.mean(ROI_color, mask=ROI)
    R.append(R_new)
    G.append(G_new)
    B.append(B_new)
    if frame_num >= 900: # when 900 frames collected, start calculating heart rate (sliding window)
        N = 900
        #normalize RGB signals
        G_std = StandardScaler().fit_transform(np.array(G[-(N-1):]).reshape(-1, 1)) 
        G_std = G_std.reshape(1, -1)[0]
        R_std = StandardScaler().fit_transform(np.array(R[-(N-1):]).reshape(-1, 1)) 
        R_std = R_std.reshape(1, -1)[0]
        B_std = StandardScaler().fit_transform(np.array(B[-(N-1):]).reshape(-1, 1))
        B_std = B_std.reshape(1, -1)[0]
        T = 1/(len(time[-(N-1):])/(time[-1]-time[-(N-1)])) #calculate time between first and last frame (period)
        # do ICA (called PCA because originally tried PCA)
        X_f=pca.fit_transform(np.array([R_std,G_std,B_std]).transpose()).transpose() 
        b, a = signal.butter(4, [0.5/15, 1.6/15], btype='band') #Butterworth filter
        X_f = signal.lfilter(b, a, X_f) 
        N = len(X_f[0])
        yf = fft(X_f[1]) # FFT
        yf = yf/np.sqrt(N) #Normalize FFT
        xf = fftfreq(N, T) # FFT frequencies 
        xf = fftshift(xf) #FFT shift
        yplot = fftshift(abs(yf))
        plt.figure(1)
        plt.gcf().clear()
        fft_plot = yplot
        # Find highest peak between 0.75 and 4 Hz 
        fft_plot[xf<=0.75] = 0 
        print(str(xf[fft_plot[xf<=4].argmax()]*60)+' bpm') # Print heart rate
        plt.plot(xf[(xf>=0) & (xf<=4)], fft_plot[(xf>=0) & (xf<=4)]) # Plot FFT
        plt.pause(0.001)
    return R, G, B, frame, time, frame_num, start, pca, ROI
# NEW Main Func
if __name__ == '__main__':
    
    # NEW (PATH_TO_HAAR_CASCADES -> WORKINGDIR)
    # assumes this file is in the video folder
    haarpath = os.getcwd() + '/' + HAARCLASSIFIER # get path of harr xml file
    face_cascade = cv2.CascadeClassifier(haarpath) # Full path must be used
    
    # open webcam
    cap = cv2.VideoCapture(0)
    
    # check if webcam is open
    if cap.isOpened() == False:
        # since webcam doesnt work, exit on standard error
        sys.exit("Failed to open webcam")
    
    # create plot display object
    plt.ion() # interactive
    
    # start main functionality
    
    # declare variables
    firstFrame = None #Michelle asks what is the point of variables firstFrame?
    time = []
    R = []
    G = []
    B = []
    pca = FastICA(n_components=3) #the ICA class
    frame_num = 0 # start counting the frames
    start = datetime.datetime.now()
    ROI = None
    # capture loop
    while cap.isOpened(): 
        ret, frame = cap.read() # read in the frame, status
        if ret == True: # if status is true
            frame_num += 1 # count frame
            
            # First if statement, do the first if stuff
            if firstFrame is None:
                firstFrame, R, G, B, frame, time, start, ROI = first_if_statement(firstFrame, R, G, B, frame, time, start, ROI)
            
            # check something (had no comment)
            # else, run big chunk
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            else:
                R, G, B, frame, time, frame_num, start, pca, ROI = that_else_statement(R, G, B, frame, time, frame_num, start, pca, ROI)
    
