from pydub import AudioSegment
import numpy as np 
import os

def load(path,
         freq = True,
         sampling_width = False,
         encoding = 'wav'):
    '''
    Inputs:
        1) path: The path to the file.
        2) freq: Flag to check if to return the frequency of 
                this audio file.
        3) sampling_width: Flag to check if bits per sample
                should be returned.
        4) encoding: Tells the type of the audio encoding in 
                order to use the correct method of 
                AudioSegment.

    Returns:
        1) audio_array: np.ndarray of file loaded.
        2) audio_freq: The sampling rate of the .wav file.
        3) sampling_width: Returns the number of 
                bits per sample of the encoding if asked.
    '''

    # Check if path exists.
    assert os.path.exists(path),"Error: Not right path given: \n %s" % path

    # Check audio type and load appropriate audio file.
    if encoding == 'wav':
        audio_file = AudioSegment.from_wav(path)
    else:
        audio_file = AudioSegment.from_file(path)

    # Make array of audio object.
    audio_array = audio_file.get_array_of_samples()

    # Get audio file's number of bits per sample.
    bits_per_sample = audio_file.sample_width * 8

    # Find frequency of audio.
    if freq:
        audio_freq = audio_file.frame_rate
    else:
        audio_freq = None

    if sampling_width:
        return audio_array, audio_freq, bits_per_sample
    else:
        return audio_array, audio_freq
