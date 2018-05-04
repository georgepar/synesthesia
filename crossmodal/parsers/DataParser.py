from aff import *
from audio import *
from text import *
import os
import pandas as pd

class DataParse:

    def __init__(self, scripts = None,
                 dataset = None,
                 skip = False):

        if scripts is None:
            return None

        if skip:
            return None

        for script in scripts:
            os.system('{} {}'.format(script,dataset))

    def load(self,
             features = []):

        self.data = {}
        for feature in features:

            if feature == 'aff':
                self.data['aff'] = aff_parser.aff_parser('aff_data','./aff_data.pkl',True)
            elif feature == 'text':
                self.data['text'] = text_parser.utterances_parser(utt_file = 'text_data', save = True,savepath =  './text_data.pkl')
            elif feature == 'audio':
                self.data['audio'] = audio_collect.audio_parser('audio_data','./audio_data.pkl')
            else:
                pass

    def merge(self,
              amount = 'all',
              on = []):

        if on == []:
            first_taken = False
            for frame in self.data:

                if not first_taken:
                    temp = self.data[frame]
                    first_taken = not first_taken
                else:
                    temp = temp.join(self.data[frame], lsuffix = 'left', rsuffix = 'right')

        else:

            first_taken = False
            for frame in self.data:
                self.data[frame] = self.data[frame].set_index(on)

                if not first_taken:
                    temp = self.data[frame]
                    first_taken = not first_taken
                else:
                    temp = temp.join(self.data[frame])

            else:
                return temp
















if __name__ == '__main__':
    k = DataParse(['../scripts/collect_aff'],'/media/manzar/2E6F455C5D0B6310/IEMOCAP',skip = True)
    k.load(['audio','text','aff'])
    z = k.merge(amount = 'all', on = ['Utterance-Id','Gender'])
    print(z.head(2))
