import numpy as np

#from scipy.io.wavfile import read


"""
This function returns the short time average zero crossing rate of a given
segment. The given segment must be in numpy array form.

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

