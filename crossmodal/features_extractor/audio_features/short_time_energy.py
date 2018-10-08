from librosa.feature import rmse
import numpy as np

"""
This function returns the short_time_energy of a wav file.
The function takes as input the normalized wav file(in numpy array
form)  the relative freq, the desirable window_size(frame size) and the
desirable overlap for frame segmentation (in samples).
The window_size is by default 2048 samples.
The hop (used for overlapping) is by default 512 samples. 
The returned value is a short time energy numpy array (each element 
is  the short time energy for each  frame)
"""


def short_time_energy(wav, window_size=2048, hop=512):

    # frame's length in samples
    frame_length = window_size

    hop_length = hop

    rmse_vector = rmse(y=wav, frame_length=frame_length,
                       hop_length=hop_length)

    #calculate short time energy from rmse
    short_time_energy_vector = np.power(rmse_vector, 2) * frame_length

    return short_time_energy_vector

