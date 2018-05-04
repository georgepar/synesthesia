from librosa.feature import mfcc
import numpy as np

'''

  Function mfccs
  Parameters:
       (sound = None): The sound can be in a list or a np.array
                       type in order to calcualte the MFCCs.
       (coeff_no = 10): The number of the MFCCs coeffcients
                        calculated.
       (sr = 2205): The sampling rate of the sound.
       (normalize = True): Normalize the sound part before
                           extracting the MFCCs.
  Returns:
       None, if variable sound is None.
       Otherwise, np.ndarray of MFCCs.

'''

def mfccs(sound = None,sr=22050,coeff_no = 10,normalize = True):

    # Check if sound is of the desired type.
    if sound is not None and \
    type(sound) is not tuple and \
    type(sound) is not np.ndarray and \
    type(sound) is not list:
        assert False, 'Variable sound of type %s\nShould be of type None, list, tuple or np.ndarray instead.'  % type(sound)

    # If sound is a list convert it to a np.ndarray first.
    if type(sound) is list or \
    type(sound) is tuple:
      sound = np.asarray(sound)

    # Normalize audio by dividing each element of the array
    # by the absolute biggest element of the array.
    if normalize:
      abs_array = np.absolute(sound)
      sound_max = np.amax(abs_array)
      norm_wav = sound / sound_max


    # Calculate the MFCCs using librosa.feature.mfcc.
    if sound is  None:
      return None
    else:
      mfccs_calc = mfcc( y = sound,
                         sr = sr,
                         S = None,
                         n_mfcc = coeff_no)
      return mfccs_calc

"""
def dummy(x):
    print(mfccs(sound = x))

dummy(np.array([1.0,2.0,3.0]))
"""
