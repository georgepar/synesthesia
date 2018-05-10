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
                 norm_way = 'on_bits',
                 default = True):

        # Check if default behaviour is asked.
        if default:

            # Check if normalization is asked.
            if normalize:
                self.sound,bits_per_sample = self.read(audio_path)

                # Check the way of normalization.
                if norm_way == 'on_bits':
                    self.sound = norm(self.sound,
                                      bits_per_sample)
                else:
                    self.sound = norm(self.sound,
                                      on_max = True)
            else:
                self.sound = self.read(audio_path)


    '''
    read(): It loads the audio file using the audio.load method.
    '''
    def read(self,
             path,
             sampling_width = True):

        # Read from path.
        sound, sampling_width = load(path,sampling_width)
        return sound, sampling_width

    '''
    get_audio(): This function returns the np.ndarray created.
    '''
    def get_audio(self):
        return self.sound



if __name__ == '__main__':
    k = AudioParser(audio_path = '/Users/alexkafiris/Documents/IEMOCAP/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav')
    print(k.get_audio())