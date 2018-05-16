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
                self.sound, self.freq, \
                bits_per_sample= self.read(audio_path)
                self.sound = norm(self.sound,
                                  bits_per_sample)
            else:
                self.sound = self.read(audio_path)


    '''
    read(): It loads the audio file using the audio.load method.
    '''
    def read(self,
             path):

        # Read from path and keep the f
        sound,freq,sample_width = load(path)
        return sound, freq, sample_width

    '''
    get_audio(): This function returns the np.ndarray created.
    '''
    def get_audio(self):
        return self.sound


if __name__ == '__main__':
    path = '/Users/alexkafiris/Documents/IEMOCAP/Session1/sentences/wav/Ses01M_impro01/Ses01M_impro01_M000.wav'
    k = AudioParser(path)