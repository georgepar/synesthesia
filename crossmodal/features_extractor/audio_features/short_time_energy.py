from librosa.feature import rmse
import numpy as np

"""
This function returns the short time energy of a given segment. The sigment
must be in numpy array form.
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

