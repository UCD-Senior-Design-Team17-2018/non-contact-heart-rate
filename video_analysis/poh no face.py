# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:13:20 2018

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
time = []
R = []
G = []
B = []
pca = FastICA(n_components=3) #the ICA class
cap = cv2.VideoCapture(0) # open webcam
if cap.isOpened() == False:
    print("Failed to open webcam")
frame_num = 0 # start counting the frames
plt.ion() # interactive
while cap.isOpened():
    ret, frame = cap.read() # read in the frame, status
    if ret == True: # if status is true
        frame_num += 1 # count frame
        if firstFrame is None:
            start = datetime.datetime.now() # start time
            time.append(0)
            # Take first frame and find face in it
            firstFrame = frame 
            cv2.imshow("frame",firstFrame)
            old_gray = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
            R_new,G_new,B_new,_ = cv2.mean(firstFrame) # take mean
            R.append(R_new)
            G.append(G_new)
            B.append(B_new)
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        else:
            current = datetime.datetime.now()-start
            current = current.total_seconds()
            time.append(current)
            cv2.imshow('frame',frame)
            R_new,G_new,B_new,_ = cv2.mean(frame)
            R.append(R_new)
            G.append(G_new)
            B.append(B_new)
            if frame_num >= 900:
                N = 900
                G_std = StandardScaler().fit_transform(np.array(G[-(N-1):]).reshape(-1, 1))
                G_std = G_std.reshape(1, -1)[0]
                R_std = StandardScaler().fit_transform(np.array(R[-(N-1):]).reshape(-1, 1))
                R_std = R_std.reshape(1, -1)[0]
                B_std = StandardScaler().fit_transform(np.array(B[-(N-1):]).reshape(-1, 1))
                B_std = B_std.reshape(1, -1)[0]
                T = 1/(len(time[-(N-1):])/(time[-1]-time[-(N-1)]))
                X_f=pca.fit_transform(np.array([R_std,G_std,B_std]).transpose()).transpose()
              #  b, a = signal.butter(4, [0.5/15, 1.6/15], btype='band')
               # X_f = signal.lfilter(b, a, X_f)
                N = len(np.pad(X_f[1],(0,0),'constant'))
                yf = fft(np.pad(X_f[1],(0,0),'constant'))
                yf = yf/np.sqrt(N)
                xf = fftfreq(N, T)
                xf = fftshift(xf)
                yplot = fftshift(abs(yf))
                plt.figure(1)
                plt.gcf().clear()
                fft_plot = yplot
                fft_plot[xf<=0.75] = 0
                print(str(xf[fft_plot[xf<=4].argmax()]*60)+' bpm')
                plt.plot(xf[(xf>=0) & (xf<=4)], fft_plot[(xf>=0) & (xf<=4)])
                plt.pause(0.001)
                
            if frame_num == 2000:
                cap.release()
                