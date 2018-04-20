# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 15:22:56 2018

@author: Bijta
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft, fftfreq, fftshift
from sklearn.decomposition import FastICA, PCA

# #############################################################################
# Generate sample data

T = 0.00890869
#T = 1/30
N = 899
np.random.seed(int(np.random.normal(100,10)))

n_samples = 899
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

s3 = np.sin(4 * np.pi * time)  # Signal 1 : sinusoidal signal


S = np.c_[s1, s2, s3]
S += 0.2 * np.random.normal(size=S.shape)  # Add noise

S /= S.std(axis=0)  # Standardize data
# Mix data
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations

# Compute ICA
ica = FastICA(n_components=3)
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix

# We can `prove` that the ICA model applies by reverting the unmixing.
assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)


plt.figure(1)
plt.gcf().clear()


models = [X, S, S_]
names = ['Observations (mixed signal)',
         'True Sources',
         'ICA recovered signals']
colors = ['red', 'steelblue', 'orange']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(4, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)

b, a = signal.butter(4, [0.5/15, 1.6/15], btype='band') #Butterworth filter
S_filter = signal.lfilter(b, a, S) 


yf = fft(S_filter[:,0])
#yf = fft(s1)
yf = yf/np.sqrt(N) #Normalize FFT
xf = fftfreq(N, T) # FFT frequencies 
xf = fftshift(xf) #FFT shift
yplot = fftshift(abs(yf))
plt.figure(2)
#plt.gcf().clear()
fft_plot = yplot
# Find highest peak between 0.75 and 4 Hz 
#fft_plot[xf<=0.75] = 0 
print(str(xf[(xf>=0) & (xf<=4)][fft_plot[(xf>=0) & (xf<=4)].argmax()]*60)+' bpm') # Print heart rate
plt.plot(xf[(xf>=0) & (xf<=4)], fft_plot[(xf>=0) & (xf<=4)]) # Plot FFT


plt.show()


