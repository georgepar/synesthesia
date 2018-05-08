from pydub import AudioSegment
import sedit

def load(path):

    '''
    Input:
        path: Path of data to search.

    Returns:
        Utterance-Id, Path, Data, Frequency
    '''

    # Split the path to find id and ignore .*.
    splitted_path = path.split('/')
    utt_id = splitted_path[9].split('.')[0]

    # Load sound.
    audio = AudioSegment.from_wav(path)

    # Return dictionary.
    return utt_id, path, sedit.normalize(audio), audio.frame_rate



if __name__ == '__main__':
    load('/Users/alexkafiris/Documents/IEMOCAP/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F011.wav')