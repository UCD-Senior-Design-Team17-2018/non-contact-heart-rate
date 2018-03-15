# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 11:54:18 2018

@author: Bijta
"""

# -*- coding: utf-8 -*-

import cv2
import numpy as np
# Change these variables based on the location of your cloned, local repositories on your computer
PATH_TO_HAAR_CASCADES = "C:/Users/Bijta/Documents/GitHub/non-contact-heart-rate/video_analysis/" 
face_cascade = cv2.CascadeClassifier(PATH_TO_HAAR_CASCADES+'haarcascade_frontalface_default.xml') # Full pathway must be used

# Constants for finding range of skin color in YCrCb
min_YCrCb = numpy.array([80,133,77],numpy.uint8)
max_YCrCb = numpy.array([255,173,127],numpy.uint8)


# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 10,
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
#cap = cv2.VideoCapture("C:/Users/Bijta/Documents/GitHub/non-contact-heart-rate/video_analysis/test/VJ+KLT_test.mp4")
cap = cv2.VideoCapture(0)
if cap.isOpened() == False:
    print("Failed to open webcam")
frame_num = 0;
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
            # Take first frame and find face in it
            firstFrame = frame
            cv2.imshow("frame",firstFrame)
            old_gray = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(old_gray, 1.3, 5) 
            if faces == ():
                firstFrame = None
            else:
                for (x,y,w,h) in faces:
                    cv2.rectangle(firstFrame,(x,y),(x+w,y+h),(255,0,0),2)
                    cv2.imshow("frame",firstFrame)
                    VJ_mask = np.zeros_like(firstFrame)
                    VJ_mask = cv2.rectangle(VJ_mask,(x,y),(x+w,y+h),(255,0,0),-1)
                    VJ_mask = cv2.cvtColor(VJ_mask, cv2.COLOR_BGR2GRAY)
                ROI = cv2.bitwise_and(VJ_mask,im)
                p0 = cv2.goodFeaturesToTrack(old_gray, mask = VJ_mask, **feature_params)
                # Create a mask image for drawing purposes
                mask = np.zeros_like(firstFrame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
        else:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]
            good_err = err[st==1]
            if good_new.shape[1] < 4 & bool(a.size)==False:
                print('No more good features')
                break
            else:
                transformed = np.zeros_like(np.array([[[x,y]],[[x+w,y]],[[x+w,y+h]],[[x,y+h]]]))
                #tmatrix = cv2.getPerspectiveTransform(good_old[np.argsort(good_err,axis=0)[:4]],good_new[np.argsort(good_err,axis=0)[:4]])
                tmatrix = cv2.estimateRigidTransform(good_old,good_new,fullAffine=False)
                #tmatrix = cv2.estimateRigidTransform(good_old[np.argsort(good_err,axis=0)[:4]],good_new[np.argsort(good_err,axis=0)[:4]],fullAffine=True)
                transformed = cv2.transform(np.array([[[x,y]],[[x+w,y]],[[x+w,y+h]],[[x,y+h]]]),tmatrix)
                #frame = cv2.warpAffine(frame,tmatrix,dsize=frame.shape[1::-1])
                cv2.imshow("frame",frame)
                x1 = transformed[0,0,0]
                y1 = transformed[0,0,1]
                x2 = transformed[1,0,0]
                y2 = transformed[1,0,1]
                cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),(255,0,0),2)
                VJ_mask = cv2.rectangle(VJ_mask,(x,y),(x+w,y+h),(255,255,255),-1)
                ROI = cv2.bitwise_and(VJ_mask,im)
                cv2.imshow('ROI',ROI)
                #cv2.polylines(frame,[transformed],True,(255,0,0))
                # draw the tracks
                for i,(new,old) in enumerate(zip(good_new,good_old)):
                    a,b = new.ravel()
                    c,d = old.ravel()
                    mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                    frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1) #
                img = cv2.add(frame,mask)
                cv2.imshow('frame',img) #imgq
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)


cap.release()
cv2.destroyAllWindows()

    
