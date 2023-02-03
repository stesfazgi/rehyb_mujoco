'''
Simple pendulum swing experiment proving that the inertial properties
of a MuJoCo body can be modified on the fly

WARNING: It doesn't prove that it doesn't cause unstabilities
'''

from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
from shared_utils.mujoco import n_step_forward
from shared_utils import models_folder_path

if __name__ == "__main__":
    single_mass_path = os.path.join(
        models_folder_path(), "uncertainty", "inertia_pendulum.xml")

    model = load_model_from_path(single_mass_path)
    sim = MjSim(model)
    viewer = MjViewer(sim)

    target_body_index = model.body_name2id("pendulum")
    rotation_axis_index = model.joint_name2id("rot_axis")

    original_inertia = model.body_inertia[target_body_index]
    new_inertia = 100*original_inertia

    rot_torque = 50

    # bring pendulum to horizontal position
    sim.data.qfrc_applied[rotation_axis_index] = rot_torque
    print(f"Original inertia moment: {original_inertia[1]} kg.m2")
    n_step_forward(1000, sim, viewer)

    # relaxation with original inertia
    sim.data.qfrc_applied[rotation_axis_index] = 0
    print("Start first relaxation")
    n_step_forward(3000, sim, viewer)

    # bring back pendulum to horizontal position
    sim.data.qfrc_applied[rotation_axis_index] = rot_torque
    n_step_forward(1000, sim, viewer)

    # relaxation with new inertia
    print(f"New inertial moment: {new_inertia[1]} kg.m2")
    for i in range(len(new_inertia)):
        model.body_inertia[target_body_index][i] = new_inertia[i]
    sim.data.qfrc_applied[rotation_axis_index] = 0
    print("Start second relaxation")
    n_step_forward(5000, sim, viewer)
