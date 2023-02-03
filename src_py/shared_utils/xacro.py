import subprocess
import os
import re

from xml.etree import ElementTree as ET

from shared_utils.general import is_xacro_or_xml_filename, get_project_root


def get_xacro_property(root: ET.ElementTree, property_name: str):
    '''
    Looks for a xacro:property element of name 'property_name' in root

    Returns the corresponding element if there is a unique match
    Otherwise returns an Assertion error
    '''

    xacro_namespace = {'xacro': 'http://www.ros.org/wiki/xacro'}
    property_regex = f"xacro:property[@name='{property_name}']"

    matching_elements = root.findall(property_regex, xacro_namespace)

    assert len(matching_elements) == 1, f"There should be exactly one match, \
        {len(matching_elements)} matching properties were found"

    return matching_elements[0]


def set_xacro_property(root: ET.ElementTree, property_name: str, value: str):
    '''
    Looks for a xacro:property element of name 'property_name' in root
    and set its value

    Returns the corresponding element if there is a unique match
    Otherwise returns an Assertion error
    '''

    property_element = get_xacro_property(root, property_name)
    property_element.attrib['value'] = value


def xml_name_from_xacro_name(xacro_filename: str):
    assert is_xacro_or_xml_filename(xacro_filename)

    return xacro_filename.split('.')[0]+'.xml'


def parse_xacro_to_xml(xacro_filename: str, xml_filename: str = None, tmp_relpath: str = "generated_xml/elbow_muscles_ranges/"):
    '''
    tmp_relpath is the relative path from the project root to
    the script tmp directory, which contains the xacro file and
    where the generated xml files are sent
    '''
    # this is legacy; TODO: modify code to use the new 'xacro_to_xml' function
    assert is_xacro_or_xml_filename(xacro_filename)

    if xml_filename is None:
        xml_filename = xml_name_from_xacro_name(xacro_filename)

    cd_project_root = f"cd {get_project_root()}"
    # generate .xml file with same name as xacro file in tmp_relpath directory
    xacro_cmd = f"{cd_project_root} && ./xacro.sh -d {tmp_relpath} -n {xml_filename} {os.path.join(tmp_relpath, xacro_filename)}"

    xacro_logs = subprocess.check_output(
        xacro_cmd, encoding='utf8', shell=True)
    # print(xacro_logs)


def is_xacro_filename(file_name):
    '''
      file_name: string

      Returns True iff file_name is a valid XML or xacro file name
    '''
    return re.compile("^(?!.*\/)(\w|\s|-)+\.xacro$").match(file_name) is not None


def xacro_to_xml(abspath_to_xacro: str, output_dir_relpath: str = None, output_filename: str = None):

    assert is_xacro_filename(abspath_to_xacro.split('/')[-1])

    cd_root = f"cd {get_project_root()}"

    xacro_cmd = "./xacro.sh"
    if(output_dir_relpath is not None):
        xacro_cmd += f" -d {output_dir_relpath}"
    if(output_filename is not None):
        xacro_cmd += f" -n {output_filename}"
    xacro_cmd += f" {os.path.relpath(abspath_to_xacro, get_project_root())}"
    print(xacro_cmd)

    full_cmd = f"{cd_root} && {xacro_cmd}"
    logs = subprocess.check_output(
        full_cmd, encoding='utf8', shell=True)
    # print(logs)


def register_xacro_namespace():
    ET.register_namespace('xacro', 'http://www.ros.org/wiki/xacro')
