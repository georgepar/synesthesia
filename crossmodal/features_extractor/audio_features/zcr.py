import numpy as np

import scipy.io.wavfile import read

from librosa.feature import zero_crossing_rate



"""
This function returns the zcr of a wav file.
The function takes as input the normalized wav file(in numpy array 
form)  the relative freq, the desirable window_size(in samples) and the  
desirable overlap for frame segmentation in samples too.
The window_size is by default 2048 samples.
The hop (used for overlap) is by default 512 samples.
The returned value is a Zcr numpy array (each element is the zcr for 
each  frame)
"""


def zcr(wav, window_size=2048, hop=512):

    # frame's length in samples
    frame_length =window_size
    # hop for overlap in samples
    hop_length = hop

    zcr_array = zero_crossing_rate(wav, frame_length=frame_length,
                                   hop_length=hop_length)

    return zcr_array


if __name__== 'main':