import pandas as pd
import numpy as np
import re
import pickle
import os

"""
This method reads the file which contains all utterances and returns a dataframe containing utterance keys and the relevant transcription for each utterance.

Input arguments:
1. File containing utterances and the relevant transcriptions
2. Boolean var "save": if true the DataFrame is saved in savepath
3.savepath

Return value:
DataFrame with Utterance_id , Gender ,  Transcription

"""
def utterances_parser(utt_file,save,savepath):
    #if data already exists just read DataFrame
    if os.path.exists(savepath):
        text_attr_df = pd.read_pickle(savepath)
        return text_attr_df
    else:
        #we create the dictionary
        text_attr_dict = {'Utterance-Id':[], 'Gender':[], 'Transcription':[]}

        with open(utt_file) as infile:
            lines = infile.readlines()


        #now we fill in the dictionary with values searching in the file with utterances

        for line in lines:
            (utterance,transcription) = line.split(":")
            utterance = line.split(" ")[0]
            #remove 1st space in transcription
            transcription = transcription[1:len(transcription)-1]
            utt_g = re.search('(M|F)\d\d\d',utterance)[0]
            gender = utt_g[0]


            text_attr_dict['Utterance-Id'].append(utterance)
            text_attr_dict['Gender'].append(gender)
            text_attr_dict['Transcription'].append(transcription)

        #convert dictionary to dataframe
        text_attr_df = pd.DataFrame(text_attr_dict)
        print(text_attr_df.shape)

        if(save==True):

            pickle.dump(text_attr_df,open(savepath,'wb'))

        return text_attr_df
