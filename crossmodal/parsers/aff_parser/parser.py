def load(path):

    '''
    Input:
        path: the path of the file.

    Returns:
        Dictionary of utterance_id, affect, valence, action, dominace.
    '''

    # Set dictionary.
    info = {'Utterance-Id': [], 'Emotion': [], 'Valence': [], 'Action': [], 'Dominance': []}

    # Open file.
    with open(path,'r') as aff_file:
        for line in aff_file.readlines():

            # Check if we have the desired line.
            if not line.startswith('[',0,1):
                continue

            # Split line to get info.
            splitted_line = line.split('\t')

            # Split last element to get VAD values.
            vad = splitted_line[3].split(', ')
            v = float(vad[0][1:])
            a = float(vad[1])
            d = float(vad[2][:-2])

            # Fill dictionary.
            info['Utterance-Id'].append(splitted_line[1])
            info['Emotion'].append(splitted_line[2])
            info['Valence'].append(v)
            info['Action'].append(a)
            info['Dominance'].append(d)

    # Return dictionary.
    return info


if __name__ == '__main__':
    load('/Users/alexkafiris/Documents/IEMOCAP/Session4/dialog/EmoEvaluation/Ses04F_script02_1.txt')