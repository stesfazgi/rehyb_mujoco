'''
Simple lift experiment proving that the mass of a MuJoCo body can be modified on the fly

WARNING: It doesn't prove that it doesn't cause unstabilities
'''

from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
from shared_utils.mujoco import n_step_forward
from shared_utils.general import models_folder_path

if __name__ == "__main__":
    single_mass_path = os.path.join(
        models_folder_path(), "uncertainty", "mass_lift.xml")

    model = load_model_from_path(single_mass_path)
    sim = MjSim(model)
    viewer = MjViewer(sim)

    target_body_index = model.body_name2id("single_mass")
    vertical_slide_index = model.joint_name2id("vertical_slide")

    original_mass = model.body_mass[target_body_index]
    lift_force = 12*original_mass  # slightly more than the weight
    new_mass = 10*original_mass

    # first lift
    sim.data.qfrc_applied[vertical_slide_index] = lift_force
    print(f"Lifting the original mass: {original_mass} kg")
    n_step_forward(250, sim, viewer)

    # fall
    sim.data.qfrc_applied[vertical_slide_index] = 0
    n_step_forward(1000, sim, viewer)

    # lift attempt on big mass; shouldn't work
    model.body_mass[target_body_index] = new_mass
    print(f"Trying to lift new mass: {new_mass} kg")
    sim.data.qfrc_applied[vertical_slide_index] = lift_force
    n_step_forward(1000, sim, viewer)
