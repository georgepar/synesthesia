from librosa.feature import zero_crossing_rate as
import numpy as np


"""
This function returns the zcr of a wav file.
The function takes as input the normalized wav file(in array form) the relative freq, the desirable window_size(in ms) and the desirable overlap for frame segmentation.
The window_size is by default 20ms.
The overlap is by default 0.5 (50ms)
The returned value is a Zcr numpy array (each element is the zcr for each frame)
"""

def zcr(wav,freq,window_size=20,overlap=0.5):

    #calculate frame's length in samples
    frame_length = freq * window_size*(10**-3)
    #calculate hop for overlap in samples
    hop_length = overlap*frame_length

    zcr_array = zero_crossing_rate(wav, frame_length=frame_length, hop_length=hop_length)

    return zcr_array
