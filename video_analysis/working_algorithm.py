# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 13:55:28 2018

@author: Tanishq Abraham

Version: 1.0
(for Design Review 3)
"""

import cv2
import datetime
import numpy as np
from scipy import signal
from scipy.fftpack import fftfreq, fftshift
import pyfftw
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# Change these variables based on the location of your cloned, local repositories on your computer
PATH_TO_HAAR_CASCADES = "C:/Users/Bijta/Documents/GitHub/non-contact-heart-rate/video_analysis/" 
face_cascade = cv2.CascadeClassifier(PATH_TO_HAAR_CASCADES+'haarcascade_frontalface_default.xml') # Full pathway must be used
firstFrame = None
time = []
G = []
pca = FastICA(n_components=3) #the ICA class
# Constants for finding range of skin color in YCrCb
min_YCrCb = np.array([80,133,77],np.uint8)
max_YCrCb = np.array([255,173,127],np.uint8)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.03,
                       minDistance = 10,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))
def checkedTrace(img0, img1, p0, back_threshold = 1.0):
    p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
    p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
    d = abs(p0-p0r).reshape(-1, 2).max(-1)
    status = d < back_threshold
    return p1, status


hamming = signal.firwin(100, [0.7,3], window = 'hamming', pass_zero=False,fs=29.97) #fs needs to be changed, doing in dark environment so...
def skin_detection(frame):
    # Convert image to YCrCb
    imageYCrCb = cv2.cvtColor(frame,cv2.COLOR_BGR2YCR_CB)
    # Find region with skin tone in YCrCb image
    skinRegion = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
    # Do contour detection on skin region
    im, contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return im

pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(1.0)
# open video file
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("C:\\Users\\Bijta\\Documents\\GitHub\\non-contact-heart-rate\\video_analysis\\test\\HR2.mov")
if cap.isOpened() == False:
    print("Failed to open file")
frame_num = 0 # start counting the frames
plt.ion() # interactive plotting
while cap.isOpened(): 
    ret, frame = cap.read() # read in the frame, status
    if ret == True: # if status is true
        frame_num += 1 # count frame
        im = skin_detection(frame)
        if firstFrame is None: 
            start = datetime.datetime.now() # start time
            time.append(0)
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
                    VJ_mask = np.zeros_like(firstFrame)
                    VJ_mask = cv2.rectangle(VJ_mask,(x,y),(x+w,y+h),(255,0,0),-1)
                    VJ_mask = cv2.cvtColor(VJ_mask, cv2.COLOR_BGR2GRAY)
                ROI = cv2.bitwise_and(VJ_mask,im)
                ROI_color = cv2.bitwise_and(ROI,ROI,mask=VJ_mask)
                #cv2.imshow('ROI',ROI_color)
                
                #take average signal in the region of interest (mask)
                _,G_new,_,_ = cv2.mean(ROI_color,mask=ROI) 
                G.append(G_new)
                
                p0 = cv2.goodFeaturesToTrack(old_gray, mask = VJ_mask, **feature_params)
                # Create a mask image for drawing purposes
                mask = np.zeros_like(firstFrame)

                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        else:
            current = datetime.datetime.now()-start
            # time for the current frame
            current = current.total_seconds()
            time.append(current)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # calculate optical flow
            p1, st = checkedTrace(old_gray,frame_gray,p0)
            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]
            if good_new.shape[0] < 4:
                print('No more good features')
                break
            else:
                transformed = np.zeros_like(np.array([[[x,y]],[[x2,y]],[[x2,y2]],[[x,y2]]]))
                tmatrix = cv2.estimateRigidTransform(good_old,good_new,fullAffine=True)
                transformed = cv2.transform(np.array([[[x,y]],[[x2,y]],[[x2,y2]],[[x,y2]]]),tmatrix)
                #cv2.imshow('frame',frame)
                x = transformed[0,0,0]
                y = transformed[0,0,1]
                x2 = transformed[2,0,0]
                y2 = transformed[2,0,1]
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.rectangle(frame,(x,y),(x2,y2),(0,255,0),2)
                VJ_mask = np.zeros_like(frame_gray)
                VJ_mask = cv2.rectangle(VJ_mask,(x,y),(x2,y2),(255,255,255),-1)
                ROI = cv2.bitwise_and(VJ_mask,im)
                ROI_color = cv2.bitwise_and(frame,frame,mask=ROI)
                #cv2.imshow('ROI',ROI_color)
                #take average signal in the region of interest (mask)
                _,G_new,_,_ = cv2.mean(ROI_color, mask=ROI)
                G.append(G_new)
            if frame_num >= 700: # when 900 frames collected, start calculating heart rate (sliding window)
                if (frame_num-600) % 1 == 0: # after every 1 frame, calculate heart rate using the data in the sliding window
                    N = 600 # about 25 seconds of data
                        #normalize RGB signals
                    G_std = signal.detrend(G[-N:-1])
                    
                    #T = 1/(len(time[-(N-1):])/(time[-1]-time[-(N-1)])) #calculate time between first and last frame (period)
                    T = 1/29.97 #period
                   # do ICA (called PCA because originally tried PCA)
                    #X_f=pca.fit_transform(np.array([R_std,G_std,B_std]).transpose()).transpose() 
                    
                    #filtering
                   # b, a = signal.butter(4, [0.75/15, 1.6/15], btype='band') #Butterworth filter
                    X_f = signal.lfilter(hamming, 1, G_std) 
                    #N = len(np.pad(X_f,(0,1024),'constant'))
                   # yf = fft(np.pad(X_f,(0,1024),'constant'))
                    
                    #FFT
                    N = len(X_f)
                    yf = pyfftw.interfaces.scipy_fftpack.fft(X_f)
                    yf = yf/np.sqrt(N) #Normalize FFT
                    xf = fftfreq(N, T) # FFT frequencies 
                    xf = fftshift(xf) #FFT shift
                    yplot = fftshift(abs(yf))
                    #plt.figure(1)
                    #plt.gcf().clear()
                    fft_plot = yplot
                    # Find highest peak between 0.75 and 4 Hz 
                 #   fft_plot[xf<=0.75] = 0 
                    if frame_num == 700:
                        bpm = xf[(xf>=0.75) & (xf<=4)][fft_plot[(xf>=0.75) & (xf<=4)].argmax()]*60
                    else:
                        bpm = 0.9*bpm + 0.1*(xf[(xf>=0.75) & (xf<=4)][fft_plot[(xf>=0.75) & (xf<=4)].argmax()]*60)
                    print(str(bpm)+' bpm') # Print heart rate
                    #plt.plot(xf[(xf>=0)], fft_plot[(xf>=0)]) # Plot FFT
                    plt.pause(0.001)
            if frame_num % 10 == 0:
                print(frame_num)
                
            if frame_num == 2000:
                cap.release()
            
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)