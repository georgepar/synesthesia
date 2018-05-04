import os

'''
This method executes a script in order to find the audio files paths.
Parameters:
    1) The script name.
    2) The directory the dataset is at.
Returns:
    None
'''

def paths_generator(script_name,
                    script_arguments = None):

    os.system('{} {}'.format(script_name,script_arguments))
    return None


'''
def dummy():
    paths_generator('../../scripts/collect_wav_paths','~/Documents/IEMOCAP')

dummy()
'''

"""
def dummy():
    paths_generator('../../scripts/collect_audio','/media/manzar/2E6F455C5D0B6310/IEMOCAP')

dummy()
"""
