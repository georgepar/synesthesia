import os


#this method takes as argument the dataset_path and a script path.
#the utterance_collector method runs the script, which makes a file containing all utterances and the relevant transcription.
def utterance_collector(script_path,Dataset_path):

    os.system('{} {}'.format(script_path, Dataset_path))

#utterance_collector("../../scripts/collect_utt_text","/media/manzar/2E6F455C5D0B6310")
