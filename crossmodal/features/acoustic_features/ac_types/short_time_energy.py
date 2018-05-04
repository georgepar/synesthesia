from librosa.feature import rmse
from librosa.core import load
import numpy as np
import pandas as pd
import scipy

'''
Takes as input an audio (normalized) (wav) file  and returns the short time energy of the wav in a numpy array.

The numpy array's length is equal to the number of frames.

Input arguments:
wav file
sampling frequency
time_frame (given in ms)
overlap (e.g 0.5 for 50% overlap)

'''

def short_time_energy(wav, freq ,  time_frame=20 ,overlap=0.5):

    samples_frame = round(time_frame / 10 **(3) * freq)
    samples_hop = round(overlap * samples_frame)

    #calculate rmse for each frame with rmse function
    #ispwnw sto tetragwno
    #kai pollaplasiazw me arithmo samples sto frame (gia na min ipologisw mean alla athroisma tetragwnwn)
    rmse_vector = rmse( wav , frame_length = samples_frame , hop_length = samples_hop )

    short_time_energy_vector = np.power(rmse_vector,2) * samples_frame

    return short_time_energy_vector

"""
def dummy():
    audio_df = pd.read_pickle('../../../Data/wav.pkl')
    wavpath =  audio_df["WAV Path"][1]
    #freq,wav = scipy.io.wavfile.read(wavpath)
    #print(freq , wav)

    wav,freq = load(wavpath)
    short_time_energy(wav,freq,20,0.5)


dummy()
"""
