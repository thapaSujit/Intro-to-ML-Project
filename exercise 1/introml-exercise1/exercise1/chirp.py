import numpy as np
import matplotlib.pyplot as plt

def createChirpSignal(samplingrate: int, duration: int, freqfrom: int, freqto: int, linear: bool):
    # returns the chirp signal as list or 1D-array
    # A chirp is a signal in which the frequency increases (up-chirp) or decreases (down-chirp) with time.
    chirp_sample_time = np.linspace(0, duration, samplingrate) #T is the time it takes to sweep from f_{0} to f_{1}.
    #The NumPy linspace function creates sequences of evenly spaced values within a defined interval.
    if linear:#linear chirp signal
        chirp_rate_linear = (freqto-freqfrom)/duration # chirp rate per second
        chirp = np.sin(2*np.pi*(((chirp_rate_linear/2)*(chirp_sample_time**2))+freqfrom*chirp_sample_time))
        plt.plot(chirp)
        plt.show()
        plt.title('Linear Chirp Signal')
        return chirp

    #In a geometric chirp, also called an exponential chirp, the frequency of the signal varies with a geometric relationship over time. 
    else:#exponential chirp signal
        chirp_rate_expo = (freqto/freqfrom)**(1/duration) #Unlike the linear chirp, which has a constant chirpyness, an exponential chirp has an exponentially increasing frequency rate.
        chirp = np.sin(2*np.pi*freqfrom*(((chirp_rate_expo**chirp_sample_time)-1)/(np.log(chirp_rate_expo))))
        plt.plot(chirp)
        plt.show()
        plt.title('Exponential Chirp Signal')
        return chirp
 