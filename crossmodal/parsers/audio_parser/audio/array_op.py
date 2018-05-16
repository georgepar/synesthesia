import numpy as np 

def normalize(audio,
              bits_per_sample = None,
              unsigned = False):
    
    '''
    Inputs:
        1) audio: The audio np.ndarray
        2) bits_per_sample: The number of bits per sample
                in order to find the denominator. Is 
                used if on_max = False.
        3) unsigned: If on_max = False, then check if 
                the values can become negative.
    Returns:
        Returns the normalized signal.
    '''

    # Check if bits_per_sample were given.
    assert bits_per_sample is not None, "No bits_per_sample were given."

    # Check if samples are signed or unsigned.
    if not unsigned:
        audio_max = 2 ** (bits_per_sample - 1) - 1
    else:
        audio_max = 2 ** bits_per_sample - 1

    # Normalize the audio array.
    norm_audio = np.true_divide(audio,(audio_max * 1.0))

    # Return.
    return norm_audio