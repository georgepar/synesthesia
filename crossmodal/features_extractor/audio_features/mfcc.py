import numpy as np 
import scipy.io.wavfile as waver
from librosa.feature import mfcc

def mfccs(segment, 
         sampling_rate,
         n = 13):
    
    float_segment = segment.astype(float) # make float values for mfcc function
    mfccs = mfcc(y = float_segment,
                 sr = sampling_rate,
                 n_mfcc = n)

    return np.mean(mfccs,axis = 1) # mean of rows if more than 1 segments are made 


if __name__ == '__main__':

    path = '/Users/alexkafiris/Desktop/tets/Ses01F_impro01_F004.wav'
    sr, signal = waver.read(path)
    mfccs = mfccs(signal,sr)
    print(mfccs.shape)
    