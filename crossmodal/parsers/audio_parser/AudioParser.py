import glob
import parser

class parse:

    def __init__(self,
                 paths):

        # Find all the paths and store them.
        self.__all_paths = glob.glob(paths)

        # Flag to check if loaded data.
        self.__loaded = False

    def load(self, give_back = False):

        # Make the dictionary.
        self.audio_data = {'Utterance-Id': [],
                           'Path': [],
                           'WAV': [],
                           'Frequency': []}

        # Iterate through the paths and create a dictionary of them.
        for path in self.__all_paths:

            # Load audio and other info and store them.
            utt, path, audio, fq = parser.load(path)
            self.audio_data['Utterance-Id'].append(utt)
            self.audio_data['Path'].append(path)
            self.audio_data['WAV'].append(audio)
            self.audio_data['Frequency'].append(fq)
        else:

            # Mark paths as loaded.
            self.__loaded = True

            # Variable of number of wavs.
            self.size = len(self.__all_paths)

        # If desired, return the dictionary created.
        if give_back:
            return self.audio_data

    def to_dataframe(self):
        import pandas as pd

        # Check if data is loaded already.
        if not self.__loaded:
            self.load()

        return pd.DataFrame.from_dict(self.audio_data)

            

if __name__ == '__main__':
    k = parse('/Users/alexkafiris/Documents/IEMOCAP/Session*/sentences/wav/S*/*.wav')
    k.load()