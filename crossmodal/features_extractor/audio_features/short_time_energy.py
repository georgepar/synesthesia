from librosa.feature import rmse
import numpy as np

"""
This function returns the short time energy of a given segment. The segment
must be in numpy array form.

This function takes as inputs:

segment: a numpy array (float) representing the current segment from which we
will receive the short time energy.

window_size: librosa.rmse cuts into frames the given segment, so the 
window_size represents the length of each frame.

overlap: the overlap that the frames will with each other. It is given in msec.

Return value: the mean of the short time energies (extracted from each frame).
"""


def short_time_energy(segment, window_size=2048, overlap=20, freq=16000):

    # frame's length in samples
    frame_length = window_size

    #calculate hop length
    samples = freq * overlap *1e-3

    assert samples < window_size,"The given overlap is more than frame's size"


    hop_length = frame_length - samples

    #calculat rmse
    rmse_vector = rmse(y=segment, frame_length=frame_length,
                       hop_length=hop_length)

    #calculate short time energy from rmse
    short_time_energy_vector = np.power(rmse_vector, 2) * frame_length

    return short_time_energy_vector


if __name__ == "__main__":


    path = '/home/manzar/Desktop/IEMOCAP/Session1/sentences/wav/Ses01M_impro01/Ses01M_impro01_M000.wav'
    
    b = short_time_energy(a,2,overlap=0)
    print(b)