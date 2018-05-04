from librosa.feature import zero_crossing_rate
import numpy as np
'''
Function zcr to calculate the zero crossing rate of the given sound  part.
    Parameters:
        (sound = None): The sound part. It can be None, a list, a tuple or an np.ndarray.
        (windows_size = 20): The time duration of the frame used in the zero crossings algorithm. (in ms)
        (overlab = 5): The overlap of the windows in the zero crossings.
        (sampling_frew = 22050): The frequency our data was sampled. (in Hz)
        (normalize = False): Normalize the audio file if wanted.
    Returns: An np.ndarray of the zero crossings rate and its shape.
             If sound = None, the None is returned.
 '''

def zcr(sound = None,
        window_size = 20,
        overlap = 5,
        sampling_freq = 22050,
        normalize = False):

    # Check if sound variable is of accepted type.
    if sound is not None and \
        type(sound) is not tuple and \
        type(sound) is not list and\
        type(sound) is not np.ndarray:
            assert False, 'Variable sound of type %s\nShould be of type None, list, tuple or np.ndarray.' % type(sound)

    # If sound is of type list or tuple convert it to np.ndarray
    if type(sound) is list or \
        type(sound) is tuple:
        sound = np.array(sound)

    # If yes, then normalize the sound.
    if normalize:
        abs_array = np.absolute(sound)
        sound_max = np.amax(abs_array)
        norm_wav = sound / sound_max

     # Calculate the window and overlap size in a sample unit.
    window = int(sampling_freq * (window_size * 10 ** -3))
    jump = int(sampling_freq * ((window - overlap) * 10 ** -3))

    # Calculate the ZCR.
    if sound is None:
        return None
    else:
        zcr_calc = zero_crossing_rate(y = sound,                          frame_length = window,                          hop_length = jump)
        return zcr_calc
