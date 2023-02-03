import os
from xml.etree import ElementTree as ET

from shared_utils.general import is_xacro_or_xml_filename, gen_models_folder_path


def save_xml_file(root: ET.Element, file_name: str, directory_path: str = None):
    '''
    this functions writes a XML file containing the XML tree steming from root

    file_name + directory_path is the path to the file that will be generated;
    default directory_path is the 'bin/models/default' folder
    '''

    assert is_xacro_or_xml_filename(
        file_name), f"{file_name} is not a correct xml/xacro file name"

    if directory_path is None:
        directory_path = os.path.join(gen_models_folder_path(), "default")
        assert os.path.isdir(directory_path)

    full_path = os.path.join(directory_path, file_name)

    with open(full_path, 'w') as f:
        # print to the file
        data = ET.tostring(root, encoding="unicode")
        f.write(data)


def wrap_save_xml_element(xml_element, file_name, wrapper_name="generated", directory_path=None):
    wrapper = ET.Element(wrapper_name)
    wrapper.append(xml_element)

    save_xml_file(wrapper, file_name, directory_path)


def insert_generated_comment(root):
    generated_comment = ET.Comment(
        "Warning: this is a generated file; it shouldn't be edited")
    generated_comment.tail = "\n    "
    root.insert(0, generated_comment)
