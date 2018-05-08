import glob
import parser

class parse:

    def __init__(self,
                 paths):

        # Find all the affection data.
        self.__all_paths = glob.glob(paths)

        # Flag to check if loaded data.
        self.__loaded = False

    def load(self, give_back = False):

        # Create the dictionary we will use.
        self.aff_data = {'Utterance-Id': [],
                         'Emotion': [],
                         'Valence': [],
                         'Action': [],
                         'Dominance': []}

        # Iterate through the paths and create the dictionary.
        for path in self.__all_paths:

            # Load the dictionary of the path session.
            current_session = parser.load(path)

            # Merge with object dictionary.
            self.aff_data['Utterance-Id'] += current_session['Utterance-Id']
            self.aff_data['Emotion'] += current_session['Emotion']
            self.aff_data['Valence'] += current_session['Valence']
            self.aff_data['Action'] += current_session['Action']
            self.aff_data['Dominance'] += current_session['Dominance']
        else:
            
            # Mark paths as loaded.
            self.__loaded = True

            # Number of affections.
            self.size = len(self.aff_data['Utterance-Id'])

        # If desired, return the created dictionary.
        if give_back:
            return self.aff_data 
    
    def to_dataframe(self):
        import pandas as pd

        # Check if data is loaded already.
        if not self.__loaded:
            self.load()

        return pd.DataFrame.from_dict(self.aff_data)

if __name__ == '__main__':
    k = parse('/Users/alexkafiris/Documents/IEMOCAP/Session*/dialog/EmoEva*/Ses*.txt')
    k.load()
    print(k.size)