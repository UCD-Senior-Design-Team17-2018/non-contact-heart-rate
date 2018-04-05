# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 15:43:54 2018

@author: Bijta
"""
import os
import cv2
import datetime
import numpy as np
from scipy import signal
from scipy.fftpack import fft, fftfreq, fftshift
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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

class HeartRateMeasure():
    def __init__(self,video_src=0, window_size=100, fps=30):
        self.fps = fps
        self.cap = cv2.VideoCapture(video_src)
        self.frame = []
        self.window_size = window_size
        self.start = 0
        self.time = []
        self.R = []
        self.G = []
        self.B =[]
        self.frame_num = 0
        self.firstFrame = []
        self.faces = ()
        # Change this variable based on the location of your cloned, local repositories on your computer
        PATH_TO_HAAR_CASCADES = "C:/Users/Bijta/Documents/GitHub/non-contact-heart-rate/video_analysis/haarcascade_frontalface_default.xml" 
        if not os.path.exists(PATH_TO_HAAR_CASCADES):
            print("Cascade file not present!")
        self.face_cascade = cv2.CascadeClassifier(PATH_TO_HAAR_CASCADES)
        # Constants for finding range of skin color in YCrCb
        self.min_YCrCb = np.array([80,133,77],np.uint8)
        self.max_YCrCb = np.array([255,173,127],np.uint8)
        
    def skin_detection(self,frame):
        # Convert image to YCrCb
        imageYCrCb = cv2.cvtColor(frame,cv2.COLOR_BGR2YCR_CB)
        # Find region with skin tone in YCrCb image
        skinRegion = cv2.inRange(imageYCrCb,self.min_YCrCb,self.max_YCrCb)
        # Do contour detection on skin region
        im, contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return im
    def time_now(self,t0):
        return (datetime.datetime.now()-t0).total_seconds()
    
    def run(self):
        if self.cap.isOpened() == False:
            print("Failed to open webcam")
        while self.cap.isOpened():
            ret, self.frame = self.cap.read()
            if ret:
                self.frame_num += 1
                self.im = self.skin_detection(self.frame)
                if len(self.firstFrame) == 0:
                    self.frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                    self.first_frame()
                    print('hello')
                else:
                    lk_stat, self.frame = self.LK_track(self.old_gray,self.frame,self.p0)
                    if lk_stat:
                        cv2.imshow('frame',self.frame)
                        cv2.waitKey(10)
                        print('lk')
                        self.faces = np.array([[self.x,self.y,self.x2,self.y2]])
                        self.ROI_color, self.R_new, self.G_new, self.B_new, self.VJ_mask = self.findROIavgRGB(self.faces,self.frame,self.im)
                        self.R.append(R_new)
                        self.G.append(G_new)
                        self.B.append(B_new)
                        self.old_gray = self.frame_gray.copy()
                        self.p0 = self.good_new.reshape(-1,1,2)
                        if self.frame_num >= self.window_size:
                            self.X = np.array([self.R[(-self.window_size+1):],self.G[(-self.window_size+1):],self.B[(-self.window_size+1):]]).transpose()
                            

                    else:
                        break
                    
            
            
    def first_frame(self):
        self.start = datetime.datetime.now() #inital time for plotting, fps calculation
        self.time.append(0) # times list
        print('face')
        cv2.imshow("frame",self.frame) 
        self.faces = self.faceDetect(self.frame)
        if not len(self.faces): #if no faces, set self.firstFrame to empty array so a new frame can be searched for a face
            self.firstFrame = []
        elif len(self.faces) > 1: # If too many faces, notify user
            print("Extra face being detected! Make sure only one face is present in the view")
            self.firstFrame = []
        else:
            self.firstFrame = self.frame
            cv2.imshow("frame",self.firstFrame)
            ROI_color, R_new,G_new,B_new,VJ_mask = self.findROIavgRGB(self.faces,self.firstFrame,self.im,True)
            self.R.append(R_new)
            self.G.append(G_new)
            self.B.append(B_new)
            self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask = VJ_mask, **feature_params)
            self.x,self.y,self.w,self.h = self.faces.tolist()[0]
            self.x2 = self.x+self.w
            self.y2 = self.y+self.h
    
    def faceDetect(self,frame):
        self.old_gray = self.frame_gray  #convert to gray for cascade classifier
        return self.face_cascade.detectMultiScale(self.old_gray, 1.3, 5) # detect faces with cascade classifier
    
    def findROIavgRGB(self,faces,frame,im,wh = False):
        if wh:
            (x,y,w,h) = faces.tolist()[0] # positions of ROI bounding box for face
            x2 = x+w
            y2 = y+h
        else:
            (x,y,x2,y2) = faces.tolist()[0] # positions of ROI bounding box for face
        cv2.rectangle(frame,(x,y),(x2,y2),(255,0,0),2)
        VJ_mask = np.zeros_like(frame) #make mask
        VJ_mask = cv2.rectangle(VJ_mask,(x,y),(x2,y2),(255,0,0),-1)
        VJ_mask = cv2.cvtColor(VJ_mask, cv2.COLOR_BGR2GRAY)
        ROI = cv2.bitwise_and(VJ_mask,im)
        ROI_color = cv2.bitwise_and(frame,frame,mask=ROI)
        R_new,G_new,B_new,_ = cv2.mean(ROI_color,mask=ROI)
        cv2.imshow('ROI',ROI_color)
        cv2.waitKey(10)
        return ROI_color, R_new, G_new, B_new, VJ_mask
    
    def LK_track(self,old_gray,frame,p0):
        current = self.time_now(self.start)
        self.time.append(current)
        print('in lk')
        self.frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        self.p1, st = checkedTrace(old_gray,self.frame_gray,p0)
        # Select good points
        self.good_new = self.p1[st==1]
        self.good_old = self.p0[st==1]
        if self.good_new.shape[0] < 4:
            print('No more good features')
            return None, frame
        else:
            # array with current bounding box points
            points = np.array([[[self.x,self.y]],[[self.x2,self.y]],[[self.x2,self.y2]],[[self.x,self.y2]]]) 
            transformed = np.zeros_like(points)
            # Detertmine the affine matrix for the transformation between current and last frame
            tmatrix = cv2.estimateRigidTransform(self.good_old,self.good_new,fullAffine=True) 
            cv2.imshow("frame",frame)
            transformed = cv2.transform(np.array([[[self.x,self.y]],[[self.x2,self.y]],[[self.x2,self.y2]],[[self.x,self.y2]]]),tmatrix)
            self.x = transformed[0,0,0]
            self.y = transformed[0,0,1]
            self.x2 = transformed[2,0,0]
            self.y2 = transformed[2,0,1]
            cv2.rectangle(frame,(self.x,self.y),(self.x+self.w,self.y+self.h),(255,0,0),2)
            cv2.rectangle(frame,(self.x,self.y),(self.x2,self.y2),(0,255,0),2) 
          #  cv2.imshow('frame',frame)
            return True, frame
        
if __name__ == "__main__":
    try:
        main = HeartRateMeasure()
        main.run()
    except KeyboardInterrupt:
        main.cap.release()
        print('done')
    finally:
        main.cap.release()