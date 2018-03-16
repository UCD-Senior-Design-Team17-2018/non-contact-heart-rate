# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 01:14:18 2018

@author: Bijta
"""

import cv2

cap = cv2.VideoCapture("C:/Users/Bijta/Documents/GitHub/non-contact-heart-rate/KLT_test.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
calc_timestamps = [0.0]

while(cap.isOpened()):
    frame_exists, curr_frame = cap.read()
    if frame_exists:
        timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        calc_timestamps.append(calc_timestamps[-1] + 1000/fps)
    else:
        break

cap.release()

for i, (ts, cts) in enumerate(zip(timestamps, calc_timestamps)):
    print('Frame %d difference:'%i, abs(ts - cts))