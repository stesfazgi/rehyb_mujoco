'''
This file is a WIP
'''

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib

from exo_eval.read_test_data_from_csv import read_test_data_from_csv
from mujoco_py import load_model_from_path, MjSim, MjViewer
from shared_utils.general import get_project_root, gen_models_folder_path
from tqdm import tqdm

matplotlib.use('TkAgg')

if __name__ == "__main__":
    TOGGLE_VIEW = True

    test_name = "20210208_EFE_TorqueHIL"
    test_number = 4

    time, joint_theta, _, motor_torques, _, _, _ = read_test_data_from_csv(
        test_name, test_number)

    joint_theta = np.deg2rad(joint_theta)

    # we only study the elbow in the end
    joint_theta = joint_theta[-1]
    motor_torques = motor_torques[-1]
    assert len(joint_theta) > 100

    dt = 0.01  # torque update every hundredth of second

    ''' Use inverse dynamics to get the human torque'''
    joint_vel, joint_acc = np.zeros_like(
        joint_theta), np.zeros_like(joint_theta)

    joint_vel[1:] = (joint_theta[1:] - joint_theta[:-1]) / dt
    joint_acc[1:] = (joint_vel[1:] - joint_vel[:-1]) / dt

    # load id model
    model_path = os.path.join(
        gen_models_folder_path(), "patient", "simple_patient_alone.xml")
    assert os.path.isfile(model_path)

    model = load_model_from_path(model_path)

    sim = MjSim(model)
    sim.forward()

    ''' Evaluation on Nesm + Human model '''
    # # load mujoco model
    # model_path = os.path.join(
    # gen_models_folder_path(), "exo_with_patient", "nesm_with_simple_patient.xml")
    # assert os.path.isfile(model_path)

    # model = load_model_from_path(model_path)

    # # count number of simulation steps per torque update

    # n_steps = dt / model.opt.timestep
    # assert (n_steps).is_integer()
    # n_steps = int(n_steps)

    # # lock all dofs except elbow
    # d_theta = 1e-6
    # locked_dof_indexes = np.vectorize(model.actuator_name2id)(
    #     ["sAA", "sFE", "sIE"])

    # lock_angle = np.zeros(3)
    # assert lock_angle.shape == (3,)

    # model.jnt_range[locked_dof_indexes] = np.vstack(
    #     (lock_angle - d_theta, lock_angle + d_theta)).T

    # rough_ground_truth = 1.6*np.sin(joint_theta)*9.81*0.15 - motor_torques

    # # initialize the sim
    # sim = MjSim(model)
    # sim.forward()

    # viewer = MjViewer(sim)

    # # initialize the angle?
    # init_config_human = np.array([.05, 0., 0., joint_theta[0]])
    # init_config_exo = np.array([0., 0., 0., joint_theta[0]])

    # _ = np.vectorize(sim.data.set_joint_qpos)(
    #     ["gh_x", "gh_z", "gh_y", "el_x"], init_config_human)
    # _ = np.vectorize(sim.data.set_joint_qpos)(
    #     ["J1", "J2", "J3", "J4"], init_config_exo)

    # sim.forward()

    # human_elbow_index, exo_elbow_act_index = np.vectorize(
    #     model.joint_name2id)(["el_x", "J4"])

    # # save the angles
    # simulated_elbow_angles = np.zeros(0)

    # for human_torque, exo_torque in tqdm(zip(rough_ground_truth, motor_torques)):
    #     sim.data.qfrc_applied[human_elbow_index] = human_torque
    #     sim.data.qfrc_applied[exo_elbow_act_index] = exo_torque

    #     simulated_elbow_angles = np.hstack(
    #         (simulated_elbow_angles, np.array([sim.data.get_joint_qpos("J4")])))

    #     for _ in range(n_steps):
    #         sim.step()

    #         if viewer is not None:
    #             viewer.render()

    # plt.plot(time, np.rad2deg(simulated_elbow_angles), label="Simulated")
    # plt.plot(time, np.rad2deg(joint_theta), label="Reality")

    # plt.xlabel("Time (s)")
    # plt.ylabel("Angle (deg)")

    # plt.legend()

    # plt.show()
