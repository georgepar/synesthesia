from pydub import AudioSegment
import os
import re
import numpy as np
import pandas as pd
import pickle

'''
This function loads all the audio files in one dataframe.
Parameters:
    1) The file with the paths.
    2) The keys to be put on the dataframe.
    3) The regular expressions to extract the key values.
Returns:
    A pandas DataFrame
'''

def audio_parser(paths_file = None,
                 save_path = None):

	# Check if path is given.
    if paths_file is None:
        return None

    # Check if paths file exists.
    if not os.path.exists(paths_file):
        assert False, 'Paths file does not exist.'

    # Check if file already exists.
    if os.path.exists(save_path) and save_path is not None:
        return pd.read_pickle(save_path)

    # Create the desired dictionary.
    wav_info = {'Utterance-Id': [],'Gender':[], 'WAV Path': [], 'WAV': []}

	# Itterate through the paths file in order and read each wav.
    with open(paths_file,'r') as audio_paths:
        for path in audio_paths.readlines():

            path = path.replace('\n','')
            wav = AudioSegment.from_wav(path)
            wav_info['Utterance-Id'].append(re.search('Ses\d\d(M|F)_(script|impro)(\d\d_\d|\d\d)(a|b)?_(M|F)\d\d\d',path)[0])
            temp = re.search('(M|F)\d\d',path)[0]
            wav_info['Gender'].append(temp[0])
            wav_info['WAV Path'].append(path)
            wav_info['WAV'].append(wav.get_array_of_samples())
        else:
            wav_table = pd.DataFrame(wav_info)

            # Create the DataFrame and save it if desired.
            if save_path is not None:
                pickle.dump(wav_table,open(save_path,'wb'))
            return wav_table

def dummy():
    audio_parser('audio_data','audio_data.pkl')

dummy()
