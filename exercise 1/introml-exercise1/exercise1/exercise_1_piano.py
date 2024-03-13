from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import os
#from tqdm import tqdm

def load_sample(filename, duration=4*44100, offset=44100//10):
    sound_data = np.load(filename)# loading the data
    sound_Newdata = np.empty(duration) # creating empty array
    start = np.argmax(sound_data)+offset # searching highest peak and adding offset
    for i in range(duration):
        sound_Newdata = sound_data[start+i]
    return sound_Newdata

def compute_frequency(signal, min_freq=20):
    # Complete this function
    fft_spec = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(signal.size, 1./44100)
    fft_spec_abs = np.abs(fft_spec)
    peak = 0.0
    peak_freq = 0.0
    for i in range(fft_spec_abs.size - min_freq):
        if fft_spec_abs[i+min_freq] > peak:
            peak = fft_spec_abs[i+min_freq]
            peak_freq = freq[i+min_freq]

    return peak_freq

if __name__ == '__main__':
    for frequency in ('Piano.ff.A2.npy', 'Piano.ff.A4.npy', 'Piano.ff.A3.npy', 'Piano.ff.A5.npy', 'Piano.ff.A6.npy', 'Piano.ff.A7.npy', 'Piano.ff.XX.npy'):
        filepath = os.path.join('sounds', frequency)
        signal = load_sample(filepath)
        freq = compute_frequency(signal)
        print(frequency, " frequency: ", freq)
 
 #fft.rfft(a, n=None, axis=- 1, norm=None)[source]
#Compute the one-dimensional discrete Fourier Transform for real input.

#This function computes the one-dimensional n-point discrete Fourier Transform (DFT) of a real-valued array by means of an efficient 
# algorithm called the Fast Fourier Transform (FFT).
#numpy.fft.rfftfreq
#fft.rfftfreq(n, d=1.0)[source]
#Return the Discrete Fourier Transform sample frequencies (for usage with rfft, irfft).

#The returned float array f contains the frequency bin centers in cycles per unit of the sample spacing (with zero at the start). 
# For instance, if the sample spacing is in seconds, then the frequency unit is cycles/second.  

