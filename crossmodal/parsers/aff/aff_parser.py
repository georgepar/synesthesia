import os
import re
import pickle
import pandas as pd

'''
This function takes a file as input and extracts the affections table. It returns a pandas DataFrame and saves the DataFrame according to the value of the variable save.
'''

def aff_parser(file_name = None,
               save_path = None,
               save = False):

    # Check if path is given.
    if file_name is None:
        return None

    # Check if given path exist:
    assert os.path.exists(file_name), 'Paths file does not exists.'

    # If generated file already exists, just return it as pandas DataFrame
    if os.path.exists(save_path) and save_path is not None:
        return pd.read_pickle(save_path)

    # Create the dictionary we will use.
    aff_info = {'Utterance-Id': [], 'Gender': [], 'Emotion': [], 'Valence': [], 'Action': [], 'Dominance': []}

    # Read the given file to extract the data.
    with open(file_name,'r') as aff_lines:
        for observation in aff_lines.readlines():
            data = observation.split()
            utt_id = data[3]
            emo = data[4]
            gender = re.search('(M|F)\d\d\d',utt_id)[0][0]

            valence = float(re.search('(\d)+\.(\d)+',data[5])[0])
            action = float(re.search('(\d)+\.(\d)+',data[6])[0])
            dominance = float(re.search('(\d)+\.(\d)+',data[7])[0])

            aff_info['Utterance-Id'].append(utt_id)
            aff_info['Gender'].append(gender)
            aff_info['Emotion'].append(emo)
            aff_info['Valence'].append(valence)
            aff_info['Action'].append(action)
            aff_info['Dominance'].append(dominance)
        else:
            aff_table = pd.DataFrame(aff_info)

            # Check if we have to save the dataframe.
            if save:
                pickle.dump(aff_table,open(save_path,'wb'))

            return aff_table
