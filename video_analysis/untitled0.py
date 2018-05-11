# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 18:29:22 2018

@author: Bijta
"""

import numpy as np
import cv2
import datetime
import time
time_video = []
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('ghausi lab background 5/2.avi',fourcc, 30.0, (640,480))
i = 0 
time.sleep(2)
while(cap.isOpened()):
    i += 1
    print(i)
    ret, frame = cap.read()
    time = datetime.datetime.now() #  time
    if ret==True:
        if i == 1:
            start = time
            time_video.append(0)
        else:
            time_video.append((time-start).total_seconds())
        # write the frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

#f = open("ghausi_lab_background_5-2.txt", "w+")
#f.write(str(time))

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()