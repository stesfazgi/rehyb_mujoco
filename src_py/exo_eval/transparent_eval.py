import tikzplotlib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib

from shared_utils.general import get_project_root, models_folder_path
from mujoco_py import load_model_from_path, MjSim, MjViewer

matplotlib.use('TkAgg')


if __name__ == "__main__":
    TOGGLE_VIEW = False
    # if False just show plot
    SAVE_TIKZ = False

    # load data
    data_filename = "run1_transparent.csv"
    data_path = os.path.join(get_project_root(), "data",
                             "4_dofs_eval", data_filename)
    assert os.path.isfile(data_path)

    # load data
    df = pd.read_csv(data_path, sep=",")
    angle_data = np.deg2rad(df.filter(regex="Angle$").to_numpy())
    torque_data = df.filter(regex="Torque$").to_numpy()

    # angle_data[:,0] =

    dt = 0.01
    velocity = np.zeros_like(angle_data)
    velocity[1:, :] = (angle_data[1:, :] - angle_data[:-1, :]) / dt

    assert angle_data.shape[1] == torque_data.shape[1] == 4

    # load MJC model
    model_path = os.path.join(
        models_folder_path(), "exoskeleton", "exoskeleton_alone.xml")
    assert os.path.isfile(model_path)

    model = load_model_from_path(model_path)
    assert len(model.joint_names) == 4

    sim = MjSim(model)
    sim.forward()
    viewer = MjViewer(sim) if TOGGLE_VIEW else None

    # measure gravity torques along the simulation
    gravity_torques = np.zeros((0, 4))
    position = np.zeros((0, 4))

    for angle_config, velocity_config in zip(angle_data, velocity):
        _ = np.vectorize(sim.data.set_joint_qpos)(
            model.joint_names, angle_config)

        _ = np.vectorize(sim.data.set_joint_qvel)(
            model.joint_names, velocity_config
        )

        position = np.vstack((position, np.vectorize(
            sim.data.get_joint_qpos)(model.joint_names)))

        sim.forward()

        gravity_torques = np.vstack((gravity_torques, sim.data.qfrc_bias))
        if viewer is not None:
            viewer.render()

    # plot the simulated torques against the ground truth
    time = np.arange(len(gravity_torques)) * 0.01

    generated_pgfs_dir = os.path.join(
        get_project_root(), "bin", "pgf_plots")
    assert os.path.isdir(generated_pgfs_dir)

    joint_names = ["sAA", "sFE", "sIE", "eFE"]

    for i in range(4):
        # plot reference
        plt.figure(figsize=(15, 15))
        plt.scatter(time, torque_data[:, i], label="Ground Truth")

        # plot simulation
        plt.scatter(
            time, gravity_torques[:, i], label="Torque bias Simulation")

        plt.title(f"{joint_names[i]}")

        plt.xlabel("Time (s)")
        plt.ylabel("Torque (N.m)")
        plt.legend()

        if SAVE_TIKZ:
            tikzplotlib.save(os.path.join(
                generated_pgfs_dir, f"4_dof_transparent_eval_{model.actuator_names[i]}.pgf"))
        else:
            plt.show()
