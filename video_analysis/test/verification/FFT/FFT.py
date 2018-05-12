# -*- coding: utf-8 -*-
"""
Created on Fri May 11 16:02:48 2018

@author: Bijta
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
#from scipy.fftpack import fft, fftshift, fftfreq

#sign = 0
#filt_signal = 0
#
t_end = 100
T = 1/800
N = int(t_end/T)    
freqs = [0.05, 0.1, 0.25, 0.5, 1, 2, 3, 4, 5, 6]
t = np.linspace(0,N*T,N)
sign = np.empty(t.shape)
for i in freqs:
    sign =sign + np.sin(2*i*np.pi*t)
#plt.plot(fftshift(fftfreq(len(sign),1/30)),fftshift(abs(fft(sign))))
sign = signal.detrend(sign)
from scipy.fftpack import fft, fftfreq, fftshift
yf = fft(sign)
yf = yf/np.sqrt(N) #Normalize FFT
xf = fftfreq(N, T) # FFT frequencies 
xf = fftshift(xf) #FFT shift
yplot = fftshift(abs(yf))
plt.plot(xf,yplot)