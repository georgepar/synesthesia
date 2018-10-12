import numpy as np

#from scipy.io.wavfile import read


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


def short_time_average_zcr(segment, window_size):

    diff = np.diff(np.sign(segment))
    a_term = np.abs(np.concatenate((np.array(np.sign(segment[0])), diff),
                                   axis=None))

    zcr = np.zeros(window_size-1)
    for n in range(0, window_size-1):
        summary = 0
        for m in range(0, len(segment)-1):
            if n-m >= 0 and n <= m+window_size-1:
                summary = summary + a_term[m] * 1/(2*window_size)
        zcr[n] = summary

    return zcr


"""
if __name__ == '__main__':
    freq,mywav = read('/home/manzar/Desktop/IEMOCAP/Session1/sentences/wav/'
                      'Ses01F_impro01/Ses01F_impro01_F000.wav')
    mywav_float = mywav.astype(np.float)
    zcr_arr = short_time_average_zcr(mywav_float, 100)
    print(zcr_arr)
"""

