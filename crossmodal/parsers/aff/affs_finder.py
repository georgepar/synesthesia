import os

'''
This function will search and create a file with all the data that has to do with affections.
Parameters:
    1) script_name: the name of the script to execute.
    2) dataset_path: the dataset to parse.
'''

def aff_prepare(script_name = None,
                dataset_path = None):
    
    # Chech if a script name is given.
    if script_name is None:
        print('No path given.')
        return None

    # Check if given script exists.
    assert os.path.exists(script_name), 'Script %s does not exist' \
    % script_name

    # Create a file with the data.
    os.system('{} {}'.format(script_name,dataset_path))

    return None

