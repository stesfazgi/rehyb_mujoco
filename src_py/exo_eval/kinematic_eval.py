import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import tikzplotlib

from shared_utils.general import get_project_root, models_folder_path
from mujoco_py import load_model_from_path, MjSim

matplotlib.use('TkAgg')


if __name__ == "__main__":
    SAVE_PGF = False

    # load the model
    model_path = os.path.join(
        models_folder_path(), "exoskeleton", "exoskeleton_with_weight.xml")
    assert os.path.isfile(model_path)
    model = load_model_from_path(model_path)

    # load trajectory data
    data_filename = "run1.csv"
    data_path = os.path.join(get_project_root(), "data",
                             "4_dofs_eval", data_filename)
    assert os.path.isfile(data_path)

    # load data
    df = pd.read_csv(data_path, sep=",")
    angle_data = np.deg2rad(df.filter(regex="Angle$").to_numpy())
    angle_data = angle_data[2*len(angle_data)//3:]

    # lengthen upper arm in the case of run2
    if data_filename.endswith("2.csv"):
        model.body_pos[model.body_name2id("Link3")][0] = .5

    sim = MjSim(model)
    sim.forward()

    vec_set_joint = np.vectorize(sim.data.set_joint_qpos)

    # save simulated end effector pos
    end_effector_pos = np.zeros((0, 3))

    for angle_config in angle_data:
        vec_set_joint(model.joint_names, angle_config)
        sim.forward()

        end_effector_pos = np.vstack(
            (end_effector_pos, sim.data.get_body_xpos("weight")))

    # use color gradient to draw trajectory
    RED = np.array([1., 0., 0., 1.])
    YELLOW = np.array([1., 1., 0., 1.])
    GREEN = np.array([0., 1., 0., 1.])
    BLUE = np.array([0., 0., 1., 1.])

    shade1 = np.linspace(RED, YELLOW, num=len(end_effector_pos))

    # artificial scaling and offsetting
    end_effector_pos = end_effector_pos*1.32 + np.array([0., .025, .128])

    color = [tuple(rgba) for rgba in shade1]
    plt.figure(figsize=(10, 10))
    plt.scatter(end_effector_pos[:, 1], end_effector_pos[:, 2], color=color)
    plt.xlabel("Simulation Y axis (m)")
    plt.ylabel("Simulation Z axis (m)")

    # compare with ref data
    ground_truth_file = "run1_end_effector.csv"
    ground_truth_path = os.path.join(
        get_project_root(), "data", "4_dofs_eval", ground_truth_file)
    assert os.path.isfile(ground_truth_path)

    xyz_data = pd.read_csv(ground_truth_path, sep=",").filter(
        regex="position$").to_numpy().T
    assert xyz_data.shape[0] == 3

    xyz_data = xyz_data[:, 2*xyz_data.shape[1]//3:]
    # print(xyz_data[2, :20])

    # convert to meter
    xyz_data /= 100.

    shade2 = np.linspace(GREEN, BLUE, num=len(xyz_data[0]))
    color = [tuple(rgba) for rgba in shade2]

    plt.scatter(xyz_data[0, :], xyz_data[1, :], color=color)

    if(SAVE_PGF):
        generated_pgfs_dir = os.path.join(
            get_project_root(), "bin", "pgf_plots")
        assert os.path.isdir(generated_pgfs_dir)

        tikzplotlib.save(os.path.join(
            generated_pgfs_dir, f"kinematic_eval.pgf"))

    plt.show()
