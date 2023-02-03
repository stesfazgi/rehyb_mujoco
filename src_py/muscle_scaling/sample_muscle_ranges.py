'''
This script loads a .xacro file of the patient having scale_ua and scale_la attributes

Then it samples that file with different values for scale_ua and scale_la
For each sample, it is parsed into an xml file and MuJoCo's loader is employed
to compute muscle ranges

Muscle ranges are saved in an xml file (by default located 
at bin/outputs/elbow_muscles_ranges/muscle_ranges_sample.xml)

Those muscle ranges are used in other scripts to evaluate the impact of
upper/lower arm scales on muscle ranges
'''
import os

from mujoco_py import load_model_from_path
from mujoco_py.cymj import PyMjModel
from time import time
from itertools import product
from math import floor
from xml.etree import ElementTree as ET

from shared_utils.general import get_project_root, remove_tmp_files, models_folder_path
from shared_utils.xacro import set_xacro_property, xml_name_from_xacro_name, parse_xacro_to_xml
from xml_generation.utils import insert_generated_comment, save_xml_file


def set_xacro_scale(root: ET.Element, ua_scale: float, la_scale: float):
    '''
    Set the properties "scale_ua" and "scale_la" in the xacro file
    '''
    # set ua scale
    set_xacro_property(root, "scale_ua", str(ua_scale))

    # set la scale
    set_xacro_property(root, "scale_la", str(la_scale))


def add_model_element(root: ET.Element, model: PyMjModel, ua_scale: float, la_scale: float, muscle_names: list):
    model_element = ET.SubElement(root, "model_sample")
    model_element.attrib["ua_s"] = str(ua_scale)
    model_element.attrib["la_s"] = str(la_scale)

    for muscle_name in muscle_names:
        muscle = ET.SubElement(model_element, "muscle")

        muscle.attrib["name"] = muscle_name
        muscle_range = model.actuator_lengthrange[model.actuator_name2id(
            muscle_name)]
        muscle.attrib["lower_bound"], muscle.attrib["upper_bound"] = list(
            map(str, muscle_range))


if __name__ == "__main__":
    ET.register_namespace('xacro', 'http://www.ros.org/wiki/xacro')
    path_to_root = get_project_root()

    # ensure that we are able to run the xacro script
    abspath_to_orig_xacro = os.path.join(
        models_folder_path(), "muscles", "patient_elbow_alone.xacro")
    assert os.path.isfile(abspath_to_orig_xacro)

    # this script uses a tmp directory storing all generated files
    # this is the relative path from project root
    relpath_to_script_tmp = os.path.join(
        "bin", "outputs", "elbow_muscles_ranges")
    path_to_script_tmp = os.path.join(path_to_root, relpath_to_script_tmp)
    assert os.path.isdir(path_to_script_tmp)

    scales = [.9, .95, 1, 1.05, 1.1]
    muscle_list = ["tric_long_avg", "tric_med_avg", "tric_lat_avg",
                   "anconeus_avg", "bic_b_avg", "bic_l", "brachialis_avg", "brachiorad_avg"]

    DEBUG = True
    cached_model = None

    # initialize empty muscle ranges xml.etree file
    ranges_sample_root = ET.Element('root')
    insert_generated_comment(ranges_sample_root)
    ranges_sample_filename = "muscle_ranges_sample.xml"

    # initialize xacro model etree
    xacro_tree = ET.parse(abspath_to_orig_xacro)
    xacro_root = xacro_tree.getroot()
    insert_generated_comment(xacro_root)
    edited_xacro_filename = "patient_elbow_tmp.xacro"
    xml_filename = xml_name_from_xacro_name(edited_xacro_filename)

    for ua_scale, la_scale in product(scales, scales):
        t0 = time()

        print(f"ua_scale: {ua_scale: <4} ; la_scale: {la_scale: <4}")

        # modify scale of xacro model etree
        print("Setting xacro scale...")
        set_xacro_scale(xacro_root, ua_scale, la_scale)

        # generate xacro file with corresponding scales (use xml.etree)
        print("Saving xacro model...")
        save_xml_file(xacro_root, edited_xacro_filename, path_to_script_tmp)

        # parse newly generated xacro file into xml (uses xacro.sh)
        print("Generating xml model...")
        parse_xacro_to_xml(edited_xacro_filename,
                           xml_filename, relpath_to_script_tmp)

        # load model from file (takes a while)
        print("Loading xml model...")
        model = load_model_from_path(
            os.path.join(path_to_script_tmp, xml_filename))

        # extract muscle ranges and append them to xml.etree
        print("Extracting muscle ranges...")
        add_model_element(ranges_sample_root, model,
                          ua_scale, la_scale, muscle_list)

        print(f"Step completed in {floor(time() - t0)} seconds\n")

    # save the muscle ranges xml.etree file
    save_xml_file(ranges_sample_root,
                  ranges_sample_filename, path_to_script_tmp)

    # delete tmp files
    remove_tmp_files(path_to_script_tmp)
