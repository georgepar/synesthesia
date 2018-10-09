import os
import glob
from audio_parser.AudioParser import AudioParser
"""
This class parses the IEMOCAP DATASET and receives text,audio,  
and evaluation data.
"""
class IEMOCAP_Dataset_parser:

    datasetPath = "NULL"

    def __init__(self,Dataset_path):
        if os.path.exists(Dataset_path):
            self.datasetPath = Dataset_path
        else:
            assert os.path.lexists(Dataset_path), 'Error:  dataset ' \
                                                 'path does not exists!'


    """
    This method reads the text,audio and evaluation data of  IEMOCAP 
    dataset and creates a dictionary.
    Returns a dictionary with keys:
    main_key:  Utterance-Id
    partial_keys: Speaker-Id, Gender, Transcription, Wav, Freq,  
    Emotion, Valence, Action, Dominance
    """
    def read_dataset(self):

        text_paths,wav_paths,eval_paths = self.gather_paths()

        text_dict = self.read_text_data(text_paths)

        eval_dict = self.read_eval_data(eval_paths)
        audio_dict = self.read_audio_data(wav_paths)

        final_dict = {}
        for utt_id in eval_dict:
            value_dict = {"Speaker-Id": text_dict[utt_id]["Speaker-Id"],
                        "Gender": text_dict[utt_id]["Gender"],
                        "Transcription": text_dict[utt_id][
                            "Transcription"],
                        "Wav": audio_dict[
                        utt_id]["Wav"],
                        "Freq": audio_dict[utt_id]["Freq"],
                        "Emotion": eval_dict[utt_id]["Emotion"],
                        "Valence": eval_dict[utt_id]["Valence"],
                        "Action": eval_dict[utt_id]["Action"],
                        "Dominance": eval_dict[utt_id]["Dominance"]}
            final_dict[utt_id] = value_dict

        return final_dict

    """
    This method collects all files paths containing transcriptions , 
    wavs  and evaluations.
    Returns a 3 lists containing those paths.
    1st list transcription paths.
    2nd list wav paths.
    3rd list eval paths.
    """
    def gather_paths(self):
        transcription_paths = []
        wav_paths = []
        eval_paths =[]

        listofSessions = ["Session1", "Session2", "Session3",
                          "Session4", "Session5"]

        for Session in listofSessions:

            path = os.path.join(self.datasetPath,Session,"dialog",
                                "transcriptions/")
            transcription_paths = transcription_paths + glob.glob(
                path+"*.txt")

            path = os.path.join(self.datasetPath,Session,"dialog",
                                "EmoEvaluation/")
            eval_paths = eval_paths + glob.glob(path+"*.txt")

            path = os.path.join(self.datasetPath, Session,
                                "sentences", "wav/")
            wav_paths = wav_paths + glob.glob(path + "**/*.wav",
                                              recursive=True)

        return transcription_paths,wav_paths,eval_paths

    """
    This method takes all transcription paths and returns a nested  
    dictionary with keys:
    main_key:Utterance-Id
    partial_keys: Speaker-Id,  Gender, Transcription
    """
    def read_text_data(self,text_paths):
        all_Utterances =[]
        all_Genders = []
        all_Transcriptions = []
        all_Speakers = []

        for text_file in text_paths:
            utt_list,gender_list,trans_list = \
                self.read_trans_from_file(text_file)
            all_Utterances = utt_list + all_Utterances
            all_Genders = gender_list + all_Genders
            all_Transcriptions = trans_list + all_Transcriptions
            speaker_id = utt_list[0].split("_")[0]
            speakers_list = [speaker_id] *len(utt_list)
            all_Speakers = all_Speakers + speakers_list

        #create text_dict
        text_dict={}
        value_dict={}
        i=0
        for utt_id in all_Utterances:
            value_dict = { "Gender":all_Genders[i],
                           "Speaker-Id":all_Speakers[i],
                           "Transcription":all_Transcriptions[i] }
            text_dict[utt_id] = value_dict
            i+=1

        return text_dict


    """
    This method takes as input a file(containing utt and trans) and  
    returns 3 lists.
    The 1st list contains every utterance key.
    The 2nd list contains every gender.
    The 3rd list contains every transcription.
    The lists are sorted so as each utterance key has its relevant  
    transcription at the same index.
    """
    def read_trans_from_file(self,text_file):

        Utterance_id_list = []
        Gender_list = []
        Transcription_list =[]

        infile = open(text_file,"r")
        for line in infile.readlines():
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

                Utterance_id_list.append(utterance_key)
                Gender_list.append(gender)
                Transcription_list.append(transcription)

        return Utterance_id_list,Gender_list,Transcription_list


    """
    This method takes all text files paths containing the emotion  
    evaluation and returns a nested dictionary with keys:
    main_key: Utterance-Id
    partial_keys: Emotion,Valence,Action,Dominance.
    """
    def read_eval_data(self,eval_paths):
        Utt_id_list = []
        Emotion_list = []
        Valence_list = []
        Action_list = []
        Dominance_list = []

        for eval_file in eval_paths:
            utt_list,em_list,val_list,ac_list,dom_list =  \
                self.read_aff_from_file(eval_file)
            Utt_id_list = Utt_id_list + utt_list
            Emotion_list = Emotion_list + em_list
            Valence_list = Valence_list + val_list
            Action_list = Action_list + ac_list
            Dominance_list = Dominance_list + dom_list

        #create eval_dict
        eval_dict={}
        value_dict={}
        i=0
        for utt_id in Utt_id_list:
            value_dict = { "Emotion":Emotion_list[i],
                           "Valence":Valence_list[i],
                           "Action":Action_list[i],
                           "Dominance":Dominance_list[i] }
            eval_dict[utt_id] = value_dict
            i+=1

        return eval_dict


    """
    This method takes as input  the evaluation file.
    It also takes a list with utterance-Ids of the relevant
    """
    def read_aff_from_file(self,eval_file):
        Utterance_id_list = []
        Emotion_list = []
        Valence_list = []
        Action_list = []
        Dominance_list = []

        infile = open(eval_file,"r")
        for line in infile.readlines():
            if( ("Ses" in line) and ("MX"not in line)
                    and ("FX" not in line) ):
                line_splitted = line.split("\t")
                utt_id = line_splitted[1]
                emotion = line_splitted[2]
                if ( not emotion=="xxx" ):
                    split = line_splitted[3].split(",")
                    valence = split[0]
                    valence = valence[1:len(valence)]
                    action = split[1]
                    action = action[1:len(action)]
                    dominance = split[2]
                    dominance = dominance[1:len(dominance)-2]
                    Utterance_id_list.append(utt_id)
                    Emotion_list.append(emotion)
                    Valence_list.append(valence)
                    Action_list.append(action)
                    Dominance_list.append(dominance)

        return Utterance_id_list, Emotion_list, Valence_list,  \
               Action_list, Dominance_list


    """
    This method takes as input all wav files paths and returns a  
    nested dictionary with keys:
    main_key: Utterance-Id
    partial_keys: Wav,Freq
    """
    def read_audio_data(self,audio_paths):
        Utt_id_list = []
        Wav_list = []
        Freq_list = []

        for wav in audio_paths:
            utt_id = wav.split("/")[9]
            utt_id = utt_id[0:len(utt_id)-4]
            (wav_array, freq) = self.AudioParser.get_audio(wav)
            Utt_id_list.append(utt_id)
            Wav_list.append(wav_array)
            Freq_list.append(freq)

        #create audio_dict:
        audio_dict={}
        value_dict={}
        i=0
        for utt_id in Utt_id_list:
            value_dict = { "Wav":Wav_list[i], "Freq":Freq_list[i] }
            audio_dict[utt_id] = value_dict
            i+=1
        return audio_dict


    """
    This is a dummy audio reader!
    The original should take a wavfile and return normalized np-array 
    of wav and the relevant freq!
    
    def dummy_audio_reader(self,infile):
        import numpy as np
        return np.array([5,4,1,2]),260000


"""


if __name__ == '__main__':
    data_parser = IEMOCAP_Dataset_parser("/home/manzar/Desktop/IEMOCAP/")
    final_diction = data_parser.read_dataset()
