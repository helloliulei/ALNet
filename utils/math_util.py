"""
@author Liu Lei
"""
import numpy as np

def gcd(x,y):
    '''greatest common divisor(gcd) '''
    if x > y:
        smaller = y
    else:
        smaller = x
    for i in range(1,smaller + 1):
        if ((x % i == 0) and (y % i == 0)):
            hcf = i
    return hcf
def Envelope(time_data):
    from scipy.signal import hilbert
    from scipy.fftpack import fft
    len_data = len(time_data)
    h = abs(hilbert(time_data))
    h_m = h - np.mean(h)
    h_fft = abs(fft(h_m)) / len_data * 2
    h_fft_half = h_fft[:int(len_data / 2)]
    return h_fft_half
