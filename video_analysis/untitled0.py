# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 13:33:02 2018

@author: Bijta
"""

import cv2
import numpy as np

a = np.array([[1, 2], [4, 5], [7, 8]], dtype='float32')
h = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='float32')
a = np.array([a])

pointsOut = cv2.perspectiveTransform(a, h)