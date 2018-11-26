import numpy as np
from librosa.feature import zero_crossing_rate as zcr_lib

"""
This function returns the zcr of a given
segment. 
The given segment must be in numpy array form (float).
The window length is given in ms.
The return value is a float numpy array. 
"""


def zcr(segment,window_length=2048, overlap=0, freq=16000):

    frame_size = window_length

    samples = freq * overlap * 10e-3
    print(samples)
    assert  samples < frame_size,\
        "The given overlap extends in zcr frame's size."

    hoplength = int(frame_size - samples)

    zcr_arr = zcr_lib(segment, frame_length=frame_size,
                      hop_length=hoplength, center=True)

    return zcr_arr.squeeze()


if __name__ == '__main__':
    from scipy.io.wavfile import read
    freq, wav = read('/home/manzar/Desktop/IEMOCAP/Session1/sen'
                     'tences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav')

    mywav_float= wav.astype(np.float)
    print(mywav_float.shape)
    b = zcr(mywav_float,window_length=2048, overlap=0)
    print(b.shape)


