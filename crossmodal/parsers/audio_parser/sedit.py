import numpy as np 

def normalize(audio):

    '''
    Input:
        audio: Pydub AudioSegment object

    Returns:
        normalized np.array().
    '''


    # Get np.ndarray from audio.
    array = audio.get_array_of_samples()

    # Normalize.
    array_abs_max = np.amax(np.absolute(array))
    norm_sound = array / array_abs_max

    # Return.
    return norm_sound