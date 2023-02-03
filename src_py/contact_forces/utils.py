import numpy as np
import re
from mujoco_py import functions as mjf


def get_contact_force(model, sim_data, contact_idx: int) -> np.ndarray:
    '''
    Given a 3D contact (without torque), returns the force vector in world frame

    !! Important: the result is the force applied by geom1 on geom2 !!
    '''
    # buffer has to be of len 6 to prevent seg fault
    force_buffer = np.zeros(6)

    mjf.mj_contactForce(model, sim_data, contact_idx, force_buffer)
    return sim_data.contact[contact_idx].frame.reshape((3, 3)).T @ force_buffer[:3]


def get_contact_ft(model, sim_data, contact_idx: int) -> np.ndarray:
    '''
    Given a 3D contact, returns the force and torque vector in world frame

    !! Important: the result is the force applied by geom1 on geom2 !!
    '''
    # buffer has to be of len 6 to prevent seg fault
    force_buffer = np.zeros(6)

    mjf.mj_contactForce(model, sim_data, contact_idx, force_buffer)
    return sim_data.contact[contact_idx].frame.reshape((3, 3)).T @ force_buffer.reshape((3, 2))


def get_arm_support_geoms(geom_names, prefix):
    pattern = f"{prefix}_[0-9]+"

    return [name for name in geom_names if re.match(pattern, name)]


def get_strap_geoms_names(geom_names, prefix):
    pattern = f"{prefix}G[0-9]+_[0-9]+"

    return [name for name in geom_names if re.match(pattern, name)]
