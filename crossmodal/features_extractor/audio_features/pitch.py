import numpy as np 
from librosa.core import piptrack

def pitch(sound = None,
          fmin = 80,
          fmax = 8000,
          sr = None,
          window_length = 20,
          overlap = 5,
          threshold = 0.1):
    
    '''
        In this function we use the librosa.core.piptrack function in order to locate the pitches of the give utterance. We know that piptrack uses the QIFFT and so we give the appropriate parameters in order to calculate the spectrogram and the pitch array.
        I will not write the parameters since it is quite obvious what this method takes as input.
    '''

    # Check if the sound and the sampling rate were given.
    assert sound is not None, 'Error, no sound np.ndarray was given.'
    assert sr is not None, 'Errorm, no sampling rate was given.'

    # Calculate the sample length the overlaps that
    # will be used for the calculations of the spectrogram.
    overlap = int(overlap * 10 ** -3 * sr)

    # Return the pitch array.
    return piptrack(y = sound, sr = sr, hop_length = overlap, fmin = 1.0 * fmin, fmax = 1.0 * fmax, threshold = threshold)

if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.abspath('../../'))
    #import parsers.audio_parser.AudioParser as AParser 
    import parsers.audio_parser.AudioParser as Aparser
    k = Aparser.AudioParser('/Users/alexkafiris/Documents/IEMOCAP/Session1/sentences/wav/Ses01M_impro01/Ses01M_impro01_M000.wav')

    print(pitch(k.get_audio(),sr = k.freq))