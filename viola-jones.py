# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 16:35:37 2018

@author: Tanishq Abraham

Based on code from OpenCV tutorial: https://docs.opencv.org/3.3.0/d7/d8b/tutorial_py_face_detection.html
"""

import cv2
import numpy as np

img = cv2.imread("C:/Users/Bijta/Documents/me.jpg") # Full pathway must be used 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier('C:/Users/Bijta/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml') # Full pathway must be used
eye_cascade = cv2.CascadeClassifier('C:/Users/Bijta/Anaconda3/Lib/site-packages/cv2/data/haarcascade_eye.xml') # Full pathway must be used
faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
cv2.imshow('img',img)
cv2.waitKey(10000) # Image open for 10 seconds
cv2.destroyAllWindows() # Close image






