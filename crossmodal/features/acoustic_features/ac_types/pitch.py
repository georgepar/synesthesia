import numpy as np
from librosa.core import piptrack
from librosa.core import load

'''
In this function we use the librosa.core.piptrack function in order to locate the pitches of the give utterance. We know that piptrack uses the QIFFT and so we give the appropriate parameters in order to calculate the spectrogram and the pitch array.
I will not write the parameters since it is quite obvious what this method takes as input.

** I am not sure if that works **
'''
def pitch(sound = None,
          fmin = 80,
          fmax = 8000,
          sr = 22050,
          window_length = 20,
          overlap = 5,
          threshold = 0.1,
          normalize = False
          ):

    # If no sound is given, return None.
    if sound is None:
        return None

    # If not an np.array is given, but list or tuple are given then
    # convert them to an np.array.
    if type(sound) is tuple or \
    type(sound) is list:
        sound = np.array(sound)

    # Normalize the array if desired.
    if normalize:
      abs_array = np.absolute(sound)
      sound_max = np.amax(abs_array)
      sound = sound / sound_max

    # Calculate the sample length the overlaps that
    # will be used for the calculations of the spectrogram.
    overlap = int(overlap * 10 ** -3 * sr)

    # Return the pitch array.
    return piptrack(y = sound, sr = sr, hop_length = overlap, fmin = 1.0 * fmin, fmax = 1.0 * fmax, threshold = threshold)



'''
x, sr = load('/Users/alexkafiris/Documents/IEMOCAP/Session1/sentences/wav/Ses01F_script02_2/Ses01F_script02_2_F025.wav')

result, mags = pitch(x, normalize = True)
'''
