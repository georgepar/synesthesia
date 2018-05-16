from librosa.feature import mfcc
import numpy as np 


def mfccs(sound = None,
          sr = None,
          coeff_no = 10):

    '''
        Parameters:
            1) sound: A np.ndarray from the sound file.
            2) sr: The sampling rate of the audio file.
            3) coeff_no: The desired number of coefficients.

        Return:
            mfccs: The number of coefficients desired.
    '''
    
    # Check if sound and sampling rate are given.
    assert sound is not None, 'No sound np.ndarray was given.'
    assert sr is not None, 'No sampling rate was given.'

    # Calculate the mfccs using librosa's module.
    mfccs = mfcc(y = sound,
                 sr = sr,
                 S = None,
                 n_mfcc = coeff_no)

    # Return the result.
    return mfccs
