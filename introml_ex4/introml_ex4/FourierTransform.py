'''
Created on 04.10.2016

@author: Daniel Stromer
@modified by Charly, Christian, Max (23.12.2020)
'''

import numpy as np
import matplotlib.pyplot as plt
# do not import more modules!

def polarToKart(shape, r, theta):
    '''
    convert polar coordinates with origin in image center to kartesian
    :param shape: shape of the image
    :param r: radius from image center
    :param theta: angle
    :return: y, x
    '''
    x = np.floor(r * np.cos(theta))
    y = np.floor(r * np.sin(theta))
    yn = int(y + shape[0] // 2)
    xn = int(x + shape[1] // 2)
    coordinates = (yn, xn)

    return coordinates

def calculateMagnitudeSpectrum(img) -> np.ndarray:
    '''
    use the fft to generate a magnitude spectrum and shift it to the image center.
    Hint: This can be done with numpy :)
    :param img:
    :return:
    '''
    magnitude = np.abs(np.fft.fft2(img))
    magnitude_spectrum = 20 * np.log10(magnitude + 1)
    shift = np.fft.fftshift(magnitude_spectrum)
    return shift

def extractRingFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    '''
    Follow the approach to extract ring features
    :param magnitude_spectrum:
    :param k: number of rings to extract = #features
    :param sampling_steps: times to sample one ring
    :return: feature vector of k features
    '''
    angles = np.linspace(0, np.pi, sampling_steps)
    r = np.zeros(k)
    for i in range(k):
        for angle in angles:
            for j in range(i * k, (i + 1) * k + 1):
                kart = polarToKart(magnitude_spectrum.shape, j, angle)
                r[i] += magnitude_spectrum[kart]
    return r
def extractFanFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    """
    Follow the approach to extract Fan features
    Assume all rays have same length regardless of angle.
    Their length should be set by the smallest feasible ray.
    :param magnitude_spectrum:
    :param k: number of fans-like features to extract
    :param sampling_steps: number of rays to sample from in one fan-like area
    :return: feature vector of length k
    """
    e = np.zeros(k)
    len = k
    for i in range(len):
        sum_e = 0
        for a in np.linspace(i, i + 1, sampling_steps - 1):
            for r in range(min(magnitude_spectrum.shape) // 2):
                y, x = polarToKart(magnitude_spectrum.shape, r, (a * np.pi / k))
                x = int(x)
                y = int(y)
            sum_e += magnitude_spectrum[y, x]
        e[i] = sum_e
    return e

def calcuateFourierParameters(img, k, sampling_steps) -> (np.ndarray, np.ndarray):
    '''
    Extract Features in Fourier space following the paper.
    :param img: input image
    :param k: number of features to extract from each method
    :param sampling_steps: number of samples to accumulate for each feature
    :return: R, T feature vectors of length k
    '''
    m = calculateMagnitudeSpectrum(img)
    R = extractRingFeatures(m, k, sampling_steps)
    T = extractFanFeatures(m, k, sampling_steps)

    return T, R