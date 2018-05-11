# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:26:39 2018

@author: Bijta

    NOT DONE
"""

import cv2
import datetime
import numpy as np
from scipy import signal
from scipy.fftpack import fft, fftfreq, fftshift
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
color = np.random.randint(0,255,(100,3))
firstFrame = None

def checkedTrace(img0, img1, p0, back_threshold = 1.0):
    p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
    p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
    d = abs(p0-p0r).reshape(-1, 2).max(-1)
    status = d < back_threshold
    return p1, status

cap = cv2.VideoCapture("C:\\Users\\Bijta\\Documents\\GitHub\\non-contact-heart-rate\\video_analysis\\test\\DSC_0009.mov")
time = []
R = []
G = []
B = []
R_q = []
clf = LinearDiscriminantAnalysis(n_components=1)
pca = FastICA(n_components=3)
hamming = signal.firwin(21, [0.7,4], window = 'hamming', pass_zero=False,fs=30) #fs needs to be changed, doing in dark environment so...
def R_quantize(R,num):
    n = np.floor((R/(256/num))+0.5)
    return n

#cap = cv2.VideoCapture(0)
if cap.isOpened() == False:
    print("Failed to open webcam")
frame_num = 0
plt.ion()
while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        frame_num += 1
        # Convert image to YCrCb
        imageYCrCb = cv2.cvtColor(frame.copy(),cv2.COLOR_BGR2YCR_CB)
        # Find region with skin tone in YCrCb image
        skinRegion = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
        # Do contour detection on skin region
        im, contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if firstFrame is None:
            start = datetime.datetime.now()
            time.append(0)
            # Take first frame and find face in it
            firstFrame = frame
            cv2.imshow("frame",firstFrame)
            old_gray = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(old_gray, 1.3, 5) 
            if faces == ():
                firstFrame = None
            else:
                for (x,y,w,h) in faces: 
                    x2 = x+w
                    y2 = y+h
                    cv2.rectangle(firstFrame,(x,y),(x+w,y+h),(255,0,0),2)
                    cv2.imshow("frame",firstFrame)
                    VJ_mask = np.zeros_like(firstFrame)
                    VJ_mask = cv2.rectangle(VJ_mask,(x,y),(x+w,y+h),(255,0,0),-1)
                    VJ_mask = cv2.cvtColor(VJ_mask, cv2.COLOR_BGR2GRAY)
                ROI = cv2.bitwise_and(VJ_mask,im)
                ROI_color = cv2.bitwise_and(ROI,ROI,mask=VJ_mask)
                cv2.imshow('ROI',ROI_color)
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
                VJ_mask = np.zeros_like(frame_gray)
                VJ_mask = cv2.rectangle(VJ_mask,(x,y),(x2,y2),(255,255,255),-1)
                ROI = cv2.bitwise_and(VJ_mask,im)
                ROI_color = cv2.bitwise_and(frame,frame,mask=ROI)
                cv2.imshow('ROI',ROI_color)
                R_new,G_new,B_new,_ = cv2.mean(ROI_color, mask=ROI)
                R.append(R_new)
                G.append(G_new)
                B.append(B_new)
                R_q.append(R_quantize(R_new,250))

                # cv2.polylines(frame,[transformed],True,(255,0,0))
                # draw the tracks
                for i,(new,old) in enumerate(zip(good_new,good_old)):
                    a,b = new.ravel()
                    c,d = old.ravel()
                    mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                    frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1) #
                img = cv2.add(frame,mask)
                cv2.imshow('frame',img) #img
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break

                if frame_num >= 600:
                    #X = np.array([G[-199:],B[-199:]]).transpose()
                    #Y=np.array(R_q[-199:]).transpose()
                   # X_f = clf.fit(X,Y).transform(X) #LDA
                #    X = np.array([R[-100:],G[-100:],B[-100:]]).transpose()
                    #X_std = signal.detrend(X_std, type='linear')
                #    b, a = signal.butter(5, [0.5/15, 4/15], btype='band')
                #    X = signal.lfilter(b, a, X)
                #    X_std = X                    
                #    X_std = StandardScaler().fit_transform(X_std)
                #    X_f=pca.fit(X_std).transform(X_std)
                #    X_f_filt = X_f[:,2]
                    #X_f_filt = np.convolve(X_f[:,2],hamming,mode='same') #apply filter
                    #X_f_filt = np.insert(X_f_filt,0,np.zeros(5))
                    #X_f_filt = X_f
                   # X_f_filt = np.convolve(G[-99:],hamming,mode='same') #apply filter (green)
                #    T = 1/30
                #    N = 406
                #    yf = fft(X_f_filt)
                #   xf = fftfreq(N, T)
                #    xf = fftshift(xf)
                #    yplot = fftshift(abs(yf))
                #    plt.figure(1)
                #    plt.gcf().clear()
                #    fft_plot = yplot
                #    fft_plot[xf<=0.8] = 0
                #    print(str(xf[fft_plot[xf<=2].argmax()]*60)+' bpm')
                #    plt.plot(xf[(xf>=0) & (xf<=4)], fft_plot[(xf>=0) & (xf<=4)])
                    N = 600
                    G_std = StandardScaler().fit_transform(np.array(G[-(N-1):]).reshape(-1, 1))
                    G_std = G_std.reshape(1, -1)[0]
                    R_std = StandardScaler().fit_transform(np.array(R[-(N-1):]).reshape(-1, 1))
                    R_std = R_std.reshape(1, -1)[0]
                    B_std = StandardScaler().fit_transform(np.array(B[-(N-1):]).reshape(-1, 1))
                    B_std = B_std.reshape(1, -1)[0]
                    #T = 1/(len(time[-(N-1):])/(time[-1]-time[-(N-1)]))
                    T = 1/30
                   # b, a = signal.butter(4, [0.5/(1/(2*T)), 1.6/(1/(2*T))], btype='band')
                    G_std = signal.lfilter(hamming, 1, G_std)
                   # R_std = signal.lfilter(b, a, R_std)
                   # B_std = signal.lfilter(b, a, B_std)
                   # X_f=pca.fit_transform(np.array([R_std,G_std,B_std]).transpose()).transpose()
                    
                    N = len(G_std)
                    yf = fft(G_std)
                   # yf = fft(R_std)
                    xf = fftfreq(N, T)
                    xf = fftshift(xf)
                    yplot = fftshift(abs(yf))
                    plt.figure(1)
                    plt.gcf().clear()
                    fft_plot = yplot
                    fft_plot[xf<=0.8] = 0
                    print(str(xf[fft_plot[xf<=2].argmax()]*60)+' bpm')
                    plt.plot(xf[(xf>=0) & (xf<=4)], fft_plot[(xf>=0) & (xf<=4)])
                    
                    
                    #f,Px_ssd = signal.welch(X_f[1],30)
                   # print(f[Px_ssd.argmax()]*60)
                   # plt.plot(f,Px_ssd)
                    plt.pause(0.001)

                    
                
#            plt.scatter(current,R_new)
#            plt.scatter(current,G_new)
#            plt.scatter(current,B_new)
#            plt.pause(0.001)
            R_q.append(R_quantize(R_new,150))
            
            #plt.scatter(frame_num,G_new)
            #plt.scatter(frame_num,B_new)             
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)


cap.release()
cv2.destroyAllWindows()