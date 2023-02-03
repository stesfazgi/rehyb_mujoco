import numpy as np
from xml.etree import ElementTree as ET

from shared_utils.general import round_to_3


def compute_inertia_moment(rad_gyr, mass):
    return mass*rad_gyr**2


def compute_limb_features(body_mass, body_height, anthropometric_perc):
    '''
    body_mass: float, kg
    body_height: float, m

    Returns several upper arm features using anthropometric data:
        - the length (m)
        - the mass (kg)
        - the position of the center of gravity from proximal end (m)
        - the inertia matrix [sagittal axis, frontal axis, longitudinal axis]

    anthropometric_perc are percentages that stem from anthropometric studies
    '''

    limb_length = anthropometric_perc[0]*body_height
    limb_mass = anthropometric_perc[1]*body_mass
    limb_cg_pos = anthropometric_perc[2]*limb_length

    # radius of gyration
    limb_rad_gyr = np.array(anthropometric_perc[3:])*limb_length
    limb_I_mat = compute_inertia_moment(limb_rad_gyr, limb_mass)

    return limb_length, limb_mass, limb_cg_pos, limb_I_mat


def print_limb_features(limb_length, limb_mass, limb_cg_pos, limb_I_mat, limb_name="limb"):
    print(f"Length of the {limb_name}: {round_to_3(limb_length)} (m)\n")
    print(f"Mass of the {limb_name}: {round_to_3(limb_mass)} (kg)\n")
    print(f"CG pos of the {limb_name}: {round_to_3(limb_cg_pos)} (m)\n")
    print(f"Inertia of the {limb_name}: {round_to_3(limb_I_mat)} (m^2.kg)\n")


def inertia_to_XML(limb_mass, limb_cg_pos, limb_I_mat, xyz_slack, xyz_sign, color_seq, root_name="generated"):
    '''
    limb_cg_pos, limb_mass: float
    limb_I_mat: array, shape = (3,) 
    xyz_slack: array, shape = (3,)
    xyz_sign: array, shape = (3,), abs(xyz_sign) = [1, 1, 1]
    color_seq: array, shape = (3,), sort(color_seq) = ['b', 'g', 'r']
    root_name: string

    Returns an element tree object containing the inertial element defined by
    limb_cg_pos, limb_mass and limb_I_mat

    The final frame is the 3D frame defined by xyz_slack, xyz_sign and color_seq
    '''

    # apply transformation to compute pos of cg
    pos = np.array([limb_cg_pos, 0, 0])
    pos += xyz_slack
    pos *= xyz_sign
    pos = round_to_3(pos)

    limb_mass = round_to_3(limb_mass)

    limb_I_mat = round_to_3(limb_I_mat)

    # build XML element
    root = ET.Element(root_name)

    inertial = ET.SubElement(root, "inertial")

    inertial.set(
        "pos", f"{pos[color_seq.index('r')]} {pos[color_seq.index('g')]} {pos[color_seq.index('b')]}")

    inertial.set("mass", limb_mass)

    inertial.set(
        "diaginertia", f"{limb_I_mat[color_seq.index('b')]} {limb_I_mat[color_seq.index('g')]} {limb_I_mat[color_seq.index('r')]}")

    return root
