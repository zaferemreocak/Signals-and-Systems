import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.fftpack as fft


#   QUESTION #3 TRIM UPPER & LOWER FREQUENCIES
sampFreq, data = wavfile.read('ozean.wav')

f_data = fft.fft(data)
f_data[int(f_data.shape[0]/2)-415000 : int(f_data.shape[0]/2)] = 0
f_data[int(f_data.shape[0]/2)        : int((f_data.shape[0]/2) + 415000)] = 0
f_data = fft.ifft(f_data)
f_data = f_data.real.astype(np.int16)

wavfile.write('ozean_trimmed.wav', sampFreq, f_data)



#   TO PLOT .WAV FILE IN THE TIME DOMAIN
timeArray = np.arange(0, data.shape[0], 1)
timeArray = timeArray / sampFreq
timeArray = timeArray * 1000  #scale to milliseconds

plt.plot(timeArray, data, color='k')
plt.ylabel('Amplitude')
plt.xlabel('Time (ms)')
plt.title('Time Domain')
plt.show()

#   PLOT .WAV FILE IN THE FREQUENCY DOMAIN
nUniquePts = int(math.ceil((f_data.shape[0]+1)/2.0))
f_data = f_data[0:nUniquePts]
f_data = abs(f_data)
f_data = f_data / float(f_data.shape[0])
f_data = f_data ** 2

if f_data.shape[0] % 2 > 0:
    f_data[1:len(f_data)] = f_data[1:len(f_data)] * 2
else:
    f_data[1:len(f_data) -1] = f_data[1:len(f_data) - 1] * 2
    

freqArray = np.arange(0, nUniquePts, 1.0) * (sampFreq / f_data.shape[0]);
plt.plot(freqArray/1000, 10*(np.log10(f_data)) , color='k')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Power (dB)')
plt.title('Frequency Domain')
plt.show()

#from scipy.io.wavfile import read, write
#from scipy.signal.filter_design import butter, buttord
#from scipy.signal import lfilter
#import numpy as np

##Read .wav file
#rate, data = read('ozean.wav')
##We need to convert int16 data to float64 because buttord function needs values between 0 and 1
#data = np.float64(data / 32768.0) #   32768 -> Max Value of Signed Integer
#
#pass_gain = 0.05 # permissible loss (ripple) in passband (dB)
#stop_gain = 1.0 # attenuation required in stopband (dB)
#
##   APPLY LOW-PASS FILTER
#
##Butterworth filter order selection
##Return the order of the lowest order digital or analog Butterworth filter that
##loses no more than gpass dB in the passband 
##and has at least gstop dB attenuation in the stopband.
#order, Wn = buttord(0.2, 0.3, pass_gain, stop_gain)
#b, a = butter(order, Wn, btype = 'lowpass')
#filtered = lfilter(b, a, data)
##filtered = np.int16(filtered * 32768 * 10)
##write('monty-filtered.wav', rate, filtered)
#
##   APPLY HIGH-PASS FILTER
#
##Butterworth filter order selection
##Return the order of the lowest order digital or analog Butterworth filter that
##loses no more than gpass dB in the passband 
##and has at least gstop dB attenuation in the stopband.
#order, Wn = buttord(0.3, 0.2, pass_gain, stop_gain)
#b, a = butter(order, Wn, btype = 'highpass')
#filtered = lfilter(b, a, filtered)
#
##Get the trimmed sound file back
#filtered = np.int16(filtered * 32768 * 10)
#write('ozean_trimmed.wav', rate, filtered)