from librosa.feature import rmse
import numpy as np
"""
This function returns the short_time_energy of a wav file.
The function takes as input the normalized wav file(in numpy array
form)  the relative freq, the desirable window_size(in ms) and the
desirable overlap for frame segmentation.
The window_size is by default 20ms.
The overlap is by default 0.5 (50ms)
The returned value is a short time energy numpy array (each element 
is  the short time energy for each  frame)
"""


def short_time_energy(wav, freq, window_size=20, overlap=0.5):

    # calculate frame's length in samples
    frame_length = round(freq * window_size*(10**-3))
    # calculate hop for overlap in samples
    hop_length = round((1-overlap)*frame_length)

    rmse_vector = rmse(y=wav, frame_length=frame_length,
                       hop_length=hop_length)

    short_time_energy_vector = np.power(rmse_vector, 2) * frame_length

    return short_time_energy_vector

