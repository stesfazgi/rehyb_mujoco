import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from time import time

from shared_utils.general import get_project_root, gen_models_folder_path, models_folder_path
from mujoco_py import load_model_from_path, MjSim, MjViewer
from shared_utils.xacro import xacro_to_xml

matplotlib.use('TkAgg')


def lock_joint(model, joint_name, lock_angle, margin=0.01):
    '''
    The angle must be in rad
    '''
    model.jnt_range[model.joint_name2id(joint_name)] = np.array(
        [lock_angle - margin, lock_angle+margin])


if __name__ == "__main__":
    # VISUAL CONFIG BOOLEANS
    TOGGLE_VIEW = False
    SAVE_PLOT = False

    # MODEL CONFIG BOOLEANS
    # do we use the whole time interval, or only a subsection?
    REMOVE_EARLY_DATA = True
    LOCK_SIE = False

    # title of the figure (find config easily)
    config_string = f"RED_{REMOVE_EARLY_DATA}_LS_{LOCK_SIE}"

    # load experimental data
    data_filename = "run1.csv"
    data_path = os.path.join(get_project_root(), "data",
                             "4_dofs_eval", data_filename)
    assert os.path.isfile(data_path)

    df = pd.read_csv(data_path, sep=",")
    angle_data = np.deg2rad(df.filter(regex="Angle$").to_numpy())
    torque_data = df.filter(regex="Torque$").to_numpy()

    assert angle_data.shape[1] == torque_data.shape[1] == 4

    # to ensure that the human is "in the straps"
    # its position is hardcoded
    # TODO: maybe possible to use a cleaner procedure
    HUMAN_INIT_CONFIG = np.array([-1.19, -0.66, -1.11, np.deg2rad(120)])

    # should we filter some data
    if(REMOVE_EARLY_DATA):
        # somehow empirical starting point (seconds)
        starting_point = 82.
        angle_data = angle_data[int(starting_point/0.01):, :]
        torque_data = torque_data[int(starting_point/0.01):, :]
        HUMAN_INIT_CONFIG = np.array([-0.691, -0.565, -0.59, 2.02])

    # parse xacro model
    xacro_path = os.path.join(models_folder_path(
    ), "exo_with_patient", "nesm_with_simple_patient.xacro")
    assert os.path.isfile(xacro_path)

    xacro_to_xml(xacro_path)

    # load mujoco model
    model_path = os.path.join(
        gen_models_folder_path(), "exo_with_patient", "nesm_with_simple_patient.xml")
    assert os.path.isfile(model_path)

    model = load_model_from_path(model_path)

    # hardcode joint and actuator names
    exo_act_names = ["sAA", "sFE", "sIE", "eFE"]
    exo_joint_names = ["J1", "J2", "J3", "J4"]
    hum_joint_names = ["gh_z", "gh_x", "gh_y", "el_x"]

    assert set(exo_act_names) <= set(model.actuator_names)
    assert set(exo_joint_names+hum_joint_names) <= set(model.joint_names)

    act_indexes = np.vectorize(model.actuator_name2id)(exo_act_names)

    # in case some joint are locked, we may want to clear some joint torques
    torque_mask = [True]*torque_data.shape[1]

    if(LOCK_SIE):
        # lock shoulder intra extra rotation to mean data pos
        lock_joint(model, "J3", np.mean(angle_data[:, 2]))
        # disable torque input for sIE
        torque_mask[2] = False

    # potentially filter out some torque data
    act_indexes = act_indexes[torque_mask]
    torque_data = torque_data[:, torque_mask]

    # the torque are divided by the actuator gears, to set correct torque
    # TODO: set gear to 1 in model?
    torque_data /= model.actuator_gear[act_indexes, 0]

    # initialize simulation
    sim = MjSim(model)
    viewer = MjViewer(sim) if TOGGLE_VIEW else None
    sim.forward()

    # vectorize joint functions
    vec_set_joint_qpos = np.vectorize(sim.data.set_joint_qpos)
    vec_get_joint_pos = np.vectorize(sim.data.get_joint_qpos)

    # set start angles
    _ = vec_set_joint_qpos(exo_joint_names, angle_data[0, :])
    # TODO: compute proper conversion (done by hand right now)
    _ = vec_set_joint_qpos(hum_joint_names, HUMAN_INIT_CONFIG)
    sim.forward()

    # count number of simulation steps per torque update
    dt = 0.01  # torque update every hundredth of second
    n_steps = dt / sim.model.opt.timestep
    assert (n_steps).is_integer()
    n_steps = int(n_steps)

    # simulated joint pos are saved
    simulated_joint_pos = np.zeros((0, 4))

    for torque_config in tqdm(torque_data):
        # set torque
        sim.data.ctrl[act_indexes] = torque_config

        # save joint pos
        simulated_joint_pos = np.vstack(
            (simulated_joint_pos, vec_get_joint_pos(exo_joint_names)))

        for _ in range(n_steps):
            sim.step()

            if viewer is not None:
                viewer.render()

    # 10 ms frequency
    time_array = dt*np.arange(0, simulated_joint_pos.shape[0])

    # just plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    # size of dots
    s = np.full_like(simulated_joint_pos[:, 0], 5)

    for i, ax in enumerate(axes.reshape(-1)):
        # plot reference
        ax.scatter(time_array, np.rad2deg(
            angle_data.T[i]), label="Ground Truth", s=s)

        # plot simulation
        ax.scatter(
            time_array, np.rad2deg(simulated_joint_pos[:, i]), label="Simulation", s=s)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Joint position (deg)")
        ax.set_title(model.actuator_names[i])
        ax.legend()

    # set same scale to all plots, not necessarily in the same limits
    ax_with_lim = [(ax, ax.get_ylim()) for ax in axes.reshape(-1)]
    max_y_delta = np.max(
        [lmax-lmin for _, (lmin, lmax) in ax_with_lim])

    for ax, (lmin, lmax) in ax_with_lim:
        ax_delta = (max_y_delta - (lmax-lmin))/2
        ax.set_ylim(lmin-ax_delta, lmax+ax_delta)

    plt.subplots_adjust(wspace=.2, hspace=.3)
    fig.suptitle(config_string, fontsize=16)

    # save image
    if(SAVE_PLOT):
        PICS_DIR = os.path.join(
            get_project_root(), "bin", "pics", "4_dofs_eval")
        if not os.path.isdir(PICS_DIR):
            os.mkdir(PICS_DIR)

        plt.savefig(os.path.join(
            PICS_DIR, f"{int(time())}_{config_string}.png"))
    plt.show()
