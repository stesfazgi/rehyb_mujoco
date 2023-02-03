from shared_utils.mujoco import n_step_forward
from shared_utils.general import gen_models_folder_path
from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
import os
# import matplotlib.pyplot as plt


def set_angular_range(model, joint_index, range, in_degree=False):
    '''
    Works on the fly
    in_degree is True iff the range is expressed in degree
    '''
    rad_range = np.deg2rad(range) if in_degree else range
    model.jnt_range[joint_index] = rad_range


if __name__ == "__main__":
    # initialisation
    PATH_TO_MODEL = os.path.join(
        gen_models_folder_path(), "muscles", "eval_elbow_flex_ext.xml")
    model = load_model_from_path(PATH_TO_MODEL)

    sim = MjSim(model)
    viewer = MjViewer(sim)

    # get index and names of flexors/extensors/joint
    elbow_flexors_names = ["bic_b_avg", "bic_l",
                           "brachialis_avg", "brachiorad_avg"]
    elbow_flexors_index = [model.actuator_name2id(
        muscle_name) for muscle_name in elbow_flexors_names]

    elbow_extensors_names = ["tric_long_avg",
                             "tric_med_avg", "tric_lat_avg", "anconeus_avg"]
    elbow_extensors_index = [model.actuator_name2id(
        muscle_name) for muscle_name in elbow_extensors_names]

    elbow_joint_name = "el_x"
    elbow_joint_index = model.joint_name2id(elbow_joint_name)

    # parameters of the experiment
    n_points = 20
    micro_stage_steps = 150

    # compute sequential ranges
    init_lower_bound, init_upper_bound = model.jnt_range[elbow_joint_index]
    delta_n = (init_upper_bound-init_lower_bound)/n_points
    flex_angle_ranges = np.vstack((np.ones(n_points)*init_lower_bound,
                                   np.arange(1, n_points+1)*delta_n)).T
    ext_angle_ranges = np.vstack(
        (np.arange(n_points, 0, -1)*delta_n, np.ones(n_points)*init_upper_bound)).T

    # set full flexor actuation
    for flexor_index in elbow_flexors_index:
        sim.data.ctrl[flexor_index] = 1.

    for angle_range in flex_angle_ranges:
        set_angular_range(model, elbow_joint_index, angle_range)
        n_step_forward(micro_stage_steps, sim, viewer)

    # remove flexor actuation and set full extensor actuation
    for flexor_index, extensor_index in zip(elbow_flexors_index, elbow_extensors_index):
        sim.data.ctrl[flexor_index] = 0.
        sim.data.ctrl[extensor_index] = 1.

    for angle_range in ext_angle_ranges:
        set_angular_range(model, elbow_joint_index, angle_range)
        n_step_forward(micro_stage_steps, sim, viewer)
