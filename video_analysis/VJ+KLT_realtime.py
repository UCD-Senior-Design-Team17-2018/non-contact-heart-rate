# -*- coding: utf-8 -*-
"""q
Created on Thu Mar  8 16:43:09 2018

@author: Bijta


Kind of functional... Does not work when face is not detected ex: in low-light conditions
"""
import cv2
import numpy as np
# Change these variables based on the location of your cloned, local repositories on your computer
PATH_TO_HAAR_CASCADES = "C:/Users/Bijta/Documents/GitHub/non-contact-heart-rate/video_analysis/" 
face_cascade = cv2.CascadeClassifier(PATH_TO_HAAR_CASCADES+'haarcascade_frontalface_default.xml') # Full pathway must be used

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
min_corners = 5
firstFrame = None

def checkedTrace(img0, img1, p0, back_threshold = 1.0):
    p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
    p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
    d = abs(p0-p0r).reshape(-1, 2).max(-1)
    status = d < back_threshold
    return p1, status


#cap = cv2.VideoCapture("C:/Users/Bijta/Documents/GitHub/non-contact-heart-rate/video_analysis/test/VJ+KLT_test.mp4")
cap = cv2.VideoCapture(0)
if cap.isOpened() == False:
    print("Failed to open webcam")
frame_num = 0;
while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        frame_num += 1
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
                    x2 = x+w
                    y2 = y+h
                    VJ_mask = cv2.rectangle(VJ_mask,(x,y),(x2,y2),(255,0,0),-1)
                    VJ_mask = cv2.cvtColor(VJ_mask, cv2.COLOR_BGR2GRAY)
                p0 = cv2.goodFeaturesToTrack(old_gray, mask = VJ_mask, **feature_params)
                # Create a mask image for drawing purposes
                mask = np.zeros_like(firstFrame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
        else:
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
                #cv2.polylines(frame,[transformed],True,(0,0,255))
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

    
