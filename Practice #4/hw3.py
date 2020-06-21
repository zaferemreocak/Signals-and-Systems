# -*- coding: utf-8 -*-
"""
Created on Tue May 28 20:54:36 2019

@author: Emre
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft
from scipy.io import wavfile
from scipy import signal

sampFreq, data = wavfile.read('allthree.wav')
f_data = fft.fft(data)

#   plot raw allthree.wav file
mft=f_data.copy()
df=sampFreq/len(f_data)
f_plot=np.arange(-sampFreq/2+df,(sampFreq+1)/2,df)
p1=np.fft.fftshift(mft)
p1=abs(p1)
plt.plot(f_plot, p1)
plt.title('allthree.wav before filter is applied')
plt.xlabel('Frequency(Hz)')
plt.show()





#   GET FIRST SOUND
mft=f_data.copy()
mft[20000:len(mft)-20000]=0
#   plot first signal with filter is applied on 2nd & 3rd signals
df=sampFreq/len(f_data)
f_plot=np.arange(-sampFreq/2+df,(sampFreq+1)/2,df)
p1=np.fft.fftshift(mft)
p1=abs(p1)
plt.plot(f_plot, p1)
plt.title('First signal')
plt.xlabel('Frequency(Hz)')
plt.show()
#   resample the signal and record it to the first.wav file
mft=fft.ifft(mft)
secs=len(mft)/sampFreq
sf=int(secs*44100)
mft=abs(mft)
mft=signal.resample(mft, sf).astype(np.int16)
wavfile.write('first.wav', int(sf/2), mft)





#   GET THE SECOND SOUND
mft=f_data.copy()
mft=np.fft.fftshift(mft)
mft[:80000]=0
mft[-80000:]=0
mft[140000:len(mft)-140000]=0
#   plot second signal with filter is applied on 1st & 3rd signals
df = sampFreq/len(mft)
f_plot=np.arange(-sampFreq/2+df,(sampFreq+1)/2,df)
p2=abs(mft)
plt.plot(f_plot, p2)
plt.title('Second signal')
plt.xlabel('Frequency(Hz)')
plt.show()
#   (demodulate & resample) the signal and record it to the second.wav file
mft=fft.ifft(mft)
t=np.arange(0, 2, 1/sampFreq)
mft=np.sin(2*(np.pi)*32000*t)*mft

#   filter out the side lobes
mft=fft.fft(mft)
mft[:100000]=0
mft[-100000:]=0
#   plot second signal after demodulation
df = sampFreq/len(mft)
f_plot=np.arange(-sampFreq/2+df,(sampFreq+1)/2,df)
p2=abs(mft)
plt.plot(f_plot, p2)
plt.title('Second signal after demodulation')
plt.xlabel('Frequency(Hz)')
plt.show()


mft=fft.ifft(mft)
secs=len(mft)/sampFreq
sf=int(secs*44100)
mft=abs(mft)
mft=signal.resample(mft, sf).astype(np.int16)
wavfile.write('second.wav', int(sf/2), mft)





#   GET THE THIRD SOUND
mft=f_data.copy()
mft=np.fft.fftshift(mft)
mft[80000:len(mft)-80000]=0
#   plot third signal with filter is applied on 1st & 2nd signals
df = sampFreq/len(mft)
f_plot=np.arange(-sampFreq/2+df,(sampFreq+1)/2,df)
p3=abs(mft)
plt.plot(f_plot, p3)
plt.title('Third signal')
plt.xlabel('Frequency(Hz)')
plt.show()
#   (demodulate & resample) the signal and record it to the second.wav file
mft=fft.ifft(mft)
t=np.arange(0, 2, 1/sampFreq)
mft=np.sin(2*(np.pi)*64000*t)*mft

#   filter out the side lobes
mft=fft.fft(mft)
mft[:100000]=0
mft[-100000:]=0
#   plot third signal after demodulation
df = sampFreq/len(mft)
f_plot=np.arange(-sampFreq/2+df,(sampFreq+1)/2,df)
p3=abs(mft)
plt.plot(f_plot, p3)
plt.title('Third signal after demodulation')
plt.xlabel('Frequency(Hz)')
plt.show()


mft=fft.ifft(mft)
secs=len(mft)/sampFreq
sf=int(secs*48000)
mft=abs(mft)
mft=signal.resample(mft, sf).astype(np.int16)
wavfile.write('third.wav', int(sf/2), mft)