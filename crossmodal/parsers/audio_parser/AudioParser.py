from audio.audio import load as load
from audio.array_op import normalize as norm
import numpy as np 

class AudioParser:

    '''
    __init__(): It either reads an audio file and normalizes it if
        asked to do so.
    '''
    def __init__(self,
                 audio_path,
                 normalize = True,
                 default = True):

        # Check if default behaviour is asked.
        if default:

            # Check if normalization is asked.
            if normalize:
                self.sound,bits_per_sample, \
                self.freq = self.read(audio_path)
                self.sound = norm(self.sound,
                                  bits_per_sample)
            else:
                self.sound = self.read(audio_path)


    '''
    read(): It loads the audio file using the audio.load method.
    '''
    def read(self,
             path,
             sampling_width = True):

        # Read from path.

        sound, sampling_width,freq = load(path,sampling_width)
        return sound, freq

    '''
    get_audio(): This function returns the np.ndarray created.
    '''
    def get_audio(self):
        return self.sound
