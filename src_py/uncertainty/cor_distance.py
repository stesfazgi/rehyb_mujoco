'''
Simple experiment showing computation of center of rotation offset
'''

from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
from shared_utils.mujoco import n_step_forward
from shared_utils.general import models_folder_path
from numpy.linalg import norm

if __name__ == "__main__":
    two_arms_path = os.path.join(
        models_folder_path(), "uncertainty", "two_hanging_arms.xml")

    model = load_model_from_path(two_arms_path)
    sim = MjSim(model)
    viewer = MjViewer(sim)

    l_shoulder_index = model.joint_name2id("shoulder_l")
    r_shoulder_index = model.joint_name2id("shoulder_r")

    l_elbow_index = model.joint_name2id("elbow_l")
    r_elbow_index = model.joint_name2id("elbow_r")

    ll_arm_index = model.body_name2id("l_arm_l")
    rl_arm_index = model.body_name2id("l_arm_r")

    shoulder_torque = 50

    # measure initial distance
    n_step_forward(200, sim, viewer)
    # in more complex cases it may be necessary to compute a rotation of frame
    l_elbow_pos = sim.data.body_xpos[ll_arm_index]+model.jnt_pos[l_elbow_index]
    r_elbow_pos = sim.data.body_xpos[rl_arm_index]+model.jnt_pos[r_elbow_index]
    elbow_distance = norm(r_elbow_pos - l_elbow_pos)
    print(f"Initial elbow cor distance: {elbow_distance} m")

    # Open the shoulders
    sim.data.qfrc_applied[l_shoulder_index] = shoulder_torque
    sim.data.qfrc_applied[r_shoulder_index] = -shoulder_torque
    print("Opening the shoulders")
    n_step_forward(1500, sim, viewer)

    # measure new distance
    l_elbow_pos = sim.data.body_xpos[ll_arm_index]+model.jnt_pos[l_elbow_index]
    r_elbow_pos = sim.data.body_xpos[rl_arm_index]+model.jnt_pos[r_elbow_index]
    elbow_distance = norm(r_elbow_pos - l_elbow_pos)
    print(f"New elbow cor distance: {elbow_distance} m")
