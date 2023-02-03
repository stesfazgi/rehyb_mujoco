from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
import matplotlib.pyplot as plt
import os

from shared_utils.general import gen_models_folder_path
from shared_utils.mujoco import n_step_forward
from elbow_muscles_eval.utils import add_musc_force

import matplotlib
matplotlib.use('TkAgg')


def play_cocontraction_exp(joint_name, model_path):
    '''
    Plays the experience used to evaluate elbw passive torques
    '''
    # instantiate model
    model = load_model_from_path(model_path)
    sim = MjSim(model)
    viewer = MjViewer(sim)

    joint_index = model.joint_name2id(joint_name)

    # initialisation of the elbow joint position
    init_torque = 10
    sim.data.qfrc_applied[joint_index] = init_torque

    print("Initialising position...")
    init_pos_steps = 250
    n_step_forward(init_pos_steps, sim, viewer)

    sim.data.qfrc_applied[joint_index] = 0

    # set muscles actuation to start cocontraction
    for i in range(len(sim.data.ctrl)):
        sim.data.ctrl[i] = 1.

    cocontraction_steps = 200
    print("Start cocontraction...")
    n_step_forward(cocontraction_steps, sim, viewer)

    # activate external extension torque
    extension_torque = -50
    sim.data.qfrc_applied[joint_index] = extension_torque

    # store muscular torque values
    musc_torque = []

    extension_steps = 2750
    print("Start extension...")
    for i in range(extension_steps):
        sim.step()
        viewer.render()

        add_musc_force(sim.data, musc_torque, joint_index)

    # compute time array
    time = model.opt.timestep * np.arange(0, extension_steps)

    # plot time x muscular_torque
    _, axes = plt.subplots(1, 1)
    axes.scatter(time, musc_torque)
    axes.set_xlabel("Time (s)")
    axes.set_ylabel("Muscular torque (Nm)")

    axes.set_title(
        f"Extension of cocontracted muscle (extension torque: {abs(extension_torque)})")

    plt.show()


if __name__ == "__main__":
    # play
    PATH_TO_MODEL = os.path.join(
        gen_models_folder_path(), "muscles", "eval_elbow_flex_ext.xml")
    joint_name = "el_x"

    FULL_PATH_TO_MODEL = os.environ.get('REHYB_MUJOCO_PATH') + PATH_TO_MODEL
    print(FULL_PATH_TO_MODEL)
    play_cocontraction_exp(joint_name, FULL_PATH_TO_MODEL)
