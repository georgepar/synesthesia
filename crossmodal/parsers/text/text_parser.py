import glob
import os

"This is a class for parsing the text data of Iemocap dataset"

class Text_Parser:
    datasetPath = "NULL"

    def __init__(self,Dataset_path):
        if os.path.exists(Dataset_path):
            self.datasetPath = Dataset_path
        else:
            assert os.path.lexists(Dataset_path), 'Error:  dataset ' \
                                                 'path does not exists!'

    """
    This method returns a dictionary containing the text_data of the
    dataset
    Dictionary keys: Utterance-id,Gender,Transcription
    """
    def utterance_parser(self):

        text_attr_dict = {"Utterance-Id":[],"Gender":[],
                          "Transcription":[] }
        Utterance_id_list = []
        Gender_list = []
        Transcription_list= []
        listofSessions = ["Session1", "Session2", "Session3",
                          "Session4", "Session5"]

        for Session in listofSessions:
            path = os.path.join(self.datasetPath,Session,"dialog","transcriptions/")
            contained_files = glob.glob(path+"*.txt")

            for utt_file in contained_files:
                with open(utt_file,"r") as inputfile:
                    for line in inputfile.readlines():
                        (utterance_key,transcription) = line.split(":")
                        utterance_key = line.split(" ")[0]
                        #utterance_key must start with Ses and  not
                        # contains MX.. or FX..
                        if(utterance_key.startswith("Ses")
                                and ("MX" not in utterance_key)
                                and ("FX" not in utterance_key)):
                            #delete newline on transcription end  and
                            #also delete space at transcription's
                            #start
                            transcription =  transcription[1:len(
                                transcription)-1]
                            gender =  utterance_key[len(
                                utterance_key)-4:len(utterance_key)-3]
                            #add all to dict
                            text_attr_dict['Utterance-Id'].append(
                                utterance_key)
                            text_attr_dict['Gender'].append(gender)
                            text_attr_dict['Transcription'].append(
                                transcription)

        return text_attr_dict
