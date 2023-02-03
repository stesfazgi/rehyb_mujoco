import numpy as np
import re
import os


def is_valid_xml_filename(file_name):
    '''
      file_name: string

      Returns True iff file_name is a valid XML file name
    '''
    return re.compile("^(?!.*\/)(\w|\s|-)+\.xml$").match(file_name) is not None


def is_xacro_or_xml_filename(file_name):
    '''
      file_name: string

      Returns True iff file_name is a valid XML or xacro file name
    '''
    return re.compile("^(?!.*\/)(\w|\s|-)+\.((xml)|(xacro))$").match(file_name) is not None


def round_to_3(x):
    '''
      Round x to 3 significant numbers
    '''
    if(x == 0.):
        return 0.
    else:
        return round(x, -int(np.floor(np.log10(abs(x)))) + 2)


round_to_3 = np.vectorize(round_to_3)


def round_to_n(x, n=3):
    '''
        Round x to n significant numbers
    '''
    if(x == 0.):
        return 0.
    else:
        return round(x, -int(np.floor(np.log10(abs(x)))) + n - 1)


round_to_n = np.vectorize(round_to_n)


def list_to_string(list, seperator=" "):
    return seperator.join(map(str, list))


def parse_float_seq(seq_string):
    return list(map(float, seq_string.split()))


def parse_int_seq(seq_string):
    return list(map(int, seq_string.split()))


def get_project_root():
    # TODO: instead of relying on an env variable, we could recompute this dynamically
    #       using the working directory path
    return os.environ['REHYB_MUJOCO_PATH']


def models_folder_path():
    return os.path.join(get_project_root(), "models")


def gen_models_folder_path():
    return os.path.join(get_project_root(), "bin", "models")


def remove_tmp_files(tmp_directory: str, file_name_ending: str = "_tmp"):
    '''
    Remove all files with file name having "_tmp" before extension
    in the tmp_directory
    '''
    m = re.compile(f".+{file_name_ending}\..+")
    tmp_files = [f for f in os.listdir(tmp_directory) if (
        os.path.isfile(os.path.join(tmp_directory, f)) and m.match(f) is not None)]

    for f in tmp_files:
        os.remove(os.path.join(tmp_directory, f))
