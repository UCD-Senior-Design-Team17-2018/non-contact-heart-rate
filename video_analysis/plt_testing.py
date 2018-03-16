# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 18:40:50 2018

@author: Bijta
"""

import numpy as np
import matplotlib.pyplot as plt

plt.ion()

for i in range(1000):
    y = np.random.random()
    plt.scatter(i, y)