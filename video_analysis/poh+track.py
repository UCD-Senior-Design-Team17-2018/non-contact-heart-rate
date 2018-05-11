# -*- coding: utf-8 -*-
"""
Created on Sat May  5 20:37:46 2018

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
firstFrame = None
time = []
R = []
G = []
B = []
pca = FastICA(n_components=3) #the ICA class


def checkedTrace(img0, img1, p0, back_threshold = 1.0):
    p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
    p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
    d = abs(p0-p0r).reshape(-1, 2).max(-1)
    status = d < back_threshold
    return p1, status


#cap = cv2.VideoCapture(0) # open webcam
cap = cv2.VideoCapture("C:\\Users\\Bijta\\Documents\\GitHub\\non-contact-heart-rate\\video_analysis\\test\\DSC_0008.mov")
if cap.isOpened() == False:
    print("Failed to open webcam")
frame_num = 0 # start counting the frames
plt.ion() # interactive
while cap.isOpened(): 
    ret, frame = cap.read() # read in the frame, status
    if ret == True: # if status is true
        frame_num += 1 # count frame
        # Convert image to YCrCb
        imageYCrCb = cv2.cvtColor(frame.copy(),cv2.COLOR_BGR2YCR_CB)
        # Find region with skin tone in YCrCb image
        skinRegion = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
        # Do contour detection on skin region
        im, contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
                    break
                ROI = cv2.bitwise_and(VJ_mask,im)
                ROI_color = cv2.bitwise_and(ROI,ROI,mask=VJ_mask)
                cv2.imshow('ROI',ROI_color)                
                #take average signal in the region of interest (mask)
                R_new,G_new,B_new,_ = cv2.mean(ROI_color,mask=ROI) 
                R.append(R_new)
                G.append(G_new)
                B.append(B_new)
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
            cv2.imshow('frame',frame)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # calculate optical flow
            p1, st = checkedTrace(old_gray,frame_gray,p0)
            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]
            if good_new.shape[0] < 4:
                print('No more good features')
                break
            if ret:
                transformed = np.zeros_like(np.array([[[x,y]],[[x2,y]],[[x2,y2]],[[x,y2]]]))
                #tmatrix = cv2.getPerspectiveTransform(good_old[np.argsort(good_err,axis=0)[:4]],good_new[np.argsort(good_err,axis=0)[:4]])
                tmatrix = cv2.estimateRigidTransform(good_old,good_new,fullAffine=True)
                #tmatrix = cv2.estimateRigidTransform(good_old[np.argsort(good_err,axis=0)[:4]],good_new[np.argsort(good_err,axis=0)[:4]],fullAffine=True)
                transformed = cv2.transform(np.array([[[x,y]],[[x2,y]],[[x2,y2]],[[x,y2]]]),tmatrix)
                #frame = cv2.warpAffine(frame,tmatrix,dsize=frame.shape[1::-1])
                cv2.imshow("frame",frame)
                x = transformed[0,0,0]
                y = transformed[0,0,1]
                x2 = transformed[2,0,0]
                y2 = transformed[2,0,1]
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.rectangle(frame,(x,y),(x2,y2),(0,255,0),2)
                cv2.imshow('frame',frame)
                VJ_mask = np.zeros_like(frame_gray)
                VJ_mask = cv2.rectangle(VJ_mask,(x,y),(x2,y2),(255,255,255),-1)
                ROI = cv2.bitwise_and(VJ_mask,im)
                ROI_color = cv2.bitwise_and(frame,frame,mask=ROI)
                cv2.imshow('ROI',ROI_color)
                R_new,G_new,B_new,_ = cv2.mean(ROI_color, mask=ROI)
                R.append(R_new)
                G.append(G_new)
                B.append(B_new)
                for i,(new,old) in enumerate(zip(good_new,good_old)):
                    a,b = new.ravel()
                    c,d = old.ravel()
                    mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                    frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1) #
                    
            if frame_num >= 900: # when 900 frames collected, start calculating heart rate (sliding window)
                N = 800
                    #normalize RGB signals
                G_std = StandardScaler().fit_transform(np.array(G[-(N-1):]).reshape(-1, 1)) 
                G_std = G_std.reshape(1, -1)[0]
                R_std = StandardScaler().fit_transform(np.array(R[-(N-1):]).reshape(-1, 1)) 
                R_std = R_std.reshape(1, -1)[0]
                B_std = StandardScaler().fit_transform(np.array(B[-(N-1):]).reshape(-1, 1))
                B_std = B_std.reshape(1, -1)[0]

                #G_std = np.array(G[-(N-1):])
                #R_std = np.array(R[-(N-1):])
                #B_std = np.array(B[-(N-1):])
                
#                G_std = signal.detrend(G_std)
#                R_std = signal.detrend(R_std)
#                B_std = signal.detrend(B_std)
                
               # T = 1/(len(time[-(N-1):])/(time[-1]-time[-(N-1)])) #calculate time between first and last frame (period)
                T = 1/29.97
               # do ICA (called PCA because originally tried PCA)
                X_f=pca.fit_transform(np.array([R_std,G_std,B_std]).transpose()).transpose() 
                #X_f = (R_std+G_std+B_std)/3
               # b, a = signal.butter(4, [0.75/15, 1.6/15], btype='band') #Butterworth filter
               # X_f = signal.lfilter(b, a, X_f) 
                #N = len(np.pad(X_f[1],(0,1024),'constant'))
                #yf = fft(np.pad(X_f[1],(0,1024),'constant'))
                N = len(X_f[1])
                yf = fft(X_f[1])
                yf = yf/np.sqrt(N) #Normalize FFT
                xf = fftfreq(N, T) # FFT frequencies 
                xf = fftshift(xf) #FFT shift
                yplot = fftshift(abs(yf))
                plt.figure(1)
                plt.gcf().clear()
                fft_plot = yplot
                # Find highest peak between 0.75 and 4 Hz 
                fft_plot[xf<=0.75] = 0 
                if frame_num == 900:
                    bpm = xf[(xf>=0) & (xf<=4)][fft_plot[(xf>=0) & (xf<=4)].argmax()]*60
                else:
                    bpm = 0*bpm + 1*(xf[(xf>=0) & (xf<=4)][fft_plot[(xf>=0) & (xf<=4)].argmax()]*60)
                print(str(bpm)+' bpm') # Print heart rate
                plt.plot(xf[(xf>=0) & (xf<=4)], fft_plot[(xf>=0) & (xf<=4)]) # Plot FFT
                plt.pause(0.001)
                old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)
            if frame_num % 10 == 0:
                print(frame_num)
                
            if frame_num == 2000:
                cap.release()