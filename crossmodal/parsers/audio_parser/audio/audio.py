from pydub import AudioSegment
import numpy as np 
import os

def load(path,
         sampling_width = True,
         encoding = 'wav'):
    '''
    Inputs:
        1) path: The path to the file.
        2) sampling_width: Flag to check if to return number
                of bits per sample or not.
        3) encoding: Tells the type of the audio encoding in 
                order to use the correct method of 
                AudioSegment.

    Returns:
        1) audio_array: np.ndarray of file loaded.
        2) sampling_width: (If asked) Returns the number of 
                bits per sample of the encoding.
    '''

    # Check if path exists.
    assert os.path.exists(path),"Error: Not right path given."

    # Check audio type and load appropriate audio file.
    if encoding == 'wav':
        audio_file = AudioSegment.from_wav(path)
    else:
        audio_file = AudioSegment.from_file(path)

    # Make array of audio object.
    audio_array = audio_file.get_array_of_samples()

    # Check what to return.
    if sampling_width:

        # Read bytes per sample, make it to bits and return.
        sampling_width = audio_file.sample_width * 8
        return (audio_array, sampling_width)
    else:
        return audio_array


if __name__ == '__main__':
    (load('/Users/alexkafiris/Documents/IEMOCAP/Session1/ sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav'))