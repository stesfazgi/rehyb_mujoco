from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
from shared_utils.mujoco import n_step_forward
from shared_utils.general import gen_models_folder_path
import numpy as np
import matplotlib.pyplot as plt


import matplotlib
matplotlib.use('TkAgg')


def quaternion_multiply(quaternion1, quaternion0):
    '''
    Source: https://stackoverflow.com/questions/39000758/how-to-multiply-two-quaternions-by-python-or-numpy
    '''
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)


if __name__ == "__main__":
    TOGGLE_VIEWER = False

    nesm_abspath = os.path.join(
        gen_models_folder_path(), "exo_with_patient",
        "nesm_with_patient.xml"
    )

    assert os.path.isfile(nesm_abspath)

    model = load_model_from_path(nesm_abspath)
    sim = MjSim(model)
    sim.forward()

    viewer = MjViewer(sim) if TOGGLE_VIEWER else None

    # get human elbow joint index
    elbow_joint_name = "el_x"
    elbow_joint_index = model.joint_name2id(elbow_joint_name)
    # deduce position in local parent body frame
    elbow_joint_lpos = model.jnt_pos[elbow_joint_index]
    # get parent body index
    elbow_body_name = "ulna_r"
    elbow_body_index = model.body_name2id(elbow_body_name)

    # get elbow flexion actuator index
    act_joint_name = "J4"
    act_joint_index = model.joint_name2id(act_joint_name)
    # deduce position in local parent body frame
    act_joint_lpos = model.jnt_pos[act_joint_index]
    # get parent body index
    actjoint_body_name = "Link4"
    actjoint_body_index = model.body_name2id(actjoint_body_name)

    # get actuator index
    actuator_name = "eFE"
    actuator_index = model.actuator_name2id(actuator_name)

    # init steps
    n_init_steps = 300
    n_step_forward(n_init_steps, sim, viewer)

    # apply flexion
    sim.data.ctrl[actuator_index] = .3

    n_flexion_steps = 150
    # list of cor offsets = pos_el_x - pos_J4 (dim: n x 3)
    cor_offsets = []
    # list of angular positions of J4 (dim: n)
    flexion_angles = []

    for _ in range(n_flexion_steps):
        # get flexion angle
        flexion_angles.append(sim.data.get_joint_qpos(act_joint_name))

        # get cor offset
        act_joint_pos = sim.data.get_body_xpos(
            actjoint_body_name) + act_joint_lpos
        elbow_joint_pos = sim.data.get_body_xpos(
            elbow_body_name) + elbow_joint_lpos
        abs_cor_offset = elbow_joint_pos - act_joint_pos

        # compute quat
        quat_cor_offset = np.concatenate((np.zeros(1), abs_cor_offset))
        quat_actjoint_body = sim.data.get_body_xquat(actjoint_body_name)

        # rotate in local frame
        local_cor_offset = quaternion_multiply(
            quat_actjoint_body, quat_cor_offset)[1:]

        cor_offsets.append(local_cor_offset)

        # step
        sim.step()
        if viewer is not None:
            viewer.render()

    flexion_angles = np.rad2deg(flexion_angles)
    cor_offsets = np.array(cor_offsets)*1000.
    axes_labels = ["x", "y", "z"]
    fig, axes = plt.subplots(1, 3)

    for i, label in enumerate(axes_labels):
        axe = axes[i]
        axe.scatter(flexion_angles, cor_offsets[:, i])
        axe.set_xlabel("Flexion angle (deg)")
        axe.set_ylabel(f"{label} cor offset (mm)")
        axe.set_title(f"Evolution of {label} cor offset")

    plt.show()
