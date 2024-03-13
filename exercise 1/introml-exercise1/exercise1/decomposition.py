import numpy as np
import matplotlib.pyplot as plt

def createTriangleSignal(samples: int, frequency: int, k_max: int):
    # returns the signal as 1D-array (np.ndarray)
    t = np.linspace(0, 1, samples)
    signal = 0
    for i in range(k_max): #Integer k used as an index, is also the number of cycles of the k-th harmonic in interval P.
        signal = ((-1)**i)*(np.sin(2*np.pi*((2*i)+1)*frequency*t))/(((2*i)+1)**2)+signal
    triangle_signal = (signal*8)/((np.pi)**2)
    plt.title('Triangle Signal')
    plt.plot(triangle_signal)
    plt.show()
    return triangle_signal


def createSquareSignal(samples: int, frequency: int, k_max: int):
    # returns the signal as 1D-array (np.ndarray)
    t = np.linspace(0, 1, samples)
    signal = 0
    for i in range(1,k_max+1): #+1 because it is not conitunous function like triangle function
        signal = (np.sin(2*np.pi*((2*i)-1)*frequency*t))/((2*i)-1)+signal
    square_signal = (signal*8)/((np.pi)**2)
    plt.title('Square Signal')
    plt.plot(square_signal)
    plt.show()
    return square_signal
    


def createSawtoothSignal(samples: int, frequency: int, k_max: int, amplitude: int):
    # returns the signal as 1D-array (np.ndarray)
    t = np.linspace(0, 1, samples)
    signal = 0
    for i in range(1,k_max+1): #+1 because it is not conitunous function like triangle function
        signal = (np.sin(2*np.pi*i*frequency*t))/i+signal
    sawTooth_signal = (amplitude/2)-(amplitude/np.pi)*signal
    plt.title('Saw Tooth Signal')
    plt.plot(sawTooth_signal)
    plt.show()
    return sawTooth_signal
