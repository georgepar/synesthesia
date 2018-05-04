import spacy
import numpy as np
import pandas as pd
import pickle
class Text_feature_extractor:

    def __init__(self):
        self.nlp = spacy.load('en_core_web_md')

    def feature_extract_from_transcription(self,transcription):
        sentence = self.nlp(transcription)
        return sentence.vector

    def feature_extract_from_dataframe(self,dataframe,save=False, savepath='./text_feature.pkl'):
        vector_list=[]
        (rows,columns) = dataframe.shape
        #print (rows)
        for row in range(0,rows):
            #print(row)
            sentence = dataframe["Transcription"][row]
            vector = self.feature_extract_from_transcription(sentence)
            #print(vector)
            vector_list.append(vector)
            #dataframe['Text_Feature'][row] = vector
        idx = 0
        dataframe.insert(loc=idx, column='Text_Feature', value=vector_list)
        print(dataframe.head(5))
        if(save==True):
            dataframe.to_pickle(savepath)
        return dataframe

"""

def dummy():
    ex = Text_feature_extractor()

    df = pd.read_pickle('../../parsers/text_data.pkl')
    ex.feature_extract_from_dataframe(df,save=True)

dummy()
