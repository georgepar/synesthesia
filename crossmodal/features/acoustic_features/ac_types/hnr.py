from Signal_Analysis.features import signal
from librosa.core import load
import pandas as pd
import numpy as np

"""
This function takes an numpy array representing the normalized audio file and splits it to frames. The HNR is calculated for each frame and then the mean is calculated and returned.

Input values:
1.numpy array (wav file normalized)
2.frequency

Returned Value:
Mean HNR (HNR of the signal)
"""

def hnr(wav,freq):
    HNR = signal.get_HNR(wav,freq)
    return HNR

"""
def dummy():
    audio_df = pd.read_pickle('../../../Data/wav.pkl')
    wavpath =  audio_df["WAV Path"][3]

    wav,freq = load(wavpath)
    hnr(wav,freq)


dummy()
"""
