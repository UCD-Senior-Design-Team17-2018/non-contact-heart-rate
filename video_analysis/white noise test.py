# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 15:17:21 2018

@author: Bijta
"""

import numpy
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq, fftshift

mean = 0
std = 1 
num_samples = 899
samples = numpy.random.normal(mean, std, size=num_samples)
plt.plot(samples)

yf = fft(samples) # FFT
yf = yf/np.sqrt(N) #Normalize FFT
xf = fftfreq(N, T) # FFT frequencies 
xf = fftshift(xf) #FFT shift
yplot = fftshift(abs(yf))
plt.figure()
plt.gcf().clear()
fft_plot = yplot
# Find highest peak between 0.75 and 4 Hz 
fft_plot[xf<=0.75] = 0 
print(str(xf[fft_plot[xf<=4].argmax()]*60)+' bpm') # Print heart rate
plt.plot(xf[(xf>=0) & (xf<=4)], fft_plot[(xf>=0) & (xf<=4)]) # Plot FFT


plt.show()