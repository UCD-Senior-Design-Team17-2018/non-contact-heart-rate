# -*- coding: utf-8 -*-
"""
Created on Fri May 11 14:25:45 2018

@author: Bijta
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

sign = 0
filt_signal = 0

freqs = [0.05, 0.1, 0.25, 0.5, 1, 2, 3, 4, 5, 6]
filt_freqs = [1,2,3]
t = np.linspace(0,100,3000)
sign = np.empty(t.shape)
filt_signal = np.empty(t.shape)
for i in freqs:
    sign =sign + np.sin(i*t)
filt_freqs = np.sin(t)+np.sin(2*t)+np.sin(3*t)
hamming = signal.firwin(100, [0.7,3], window = 'hamming', pass_zero=False,fs=30)
X_f = signal.lfilter(hamming, 1, sign)
plt.plot(t,X_f)
plt.plot(t,filt_signal)

    
    