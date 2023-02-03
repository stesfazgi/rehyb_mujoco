import os

from mujoco_py import load_model_from_path, MjSim, MjViewer
from shared_utils.mujoco import n_step_forward
from shared_utils.general import gen_models_folder_path


def play_passive_torque_exp(joint_name, model_path, phases_nb_steps, control_forces, init_steps=200):
    '''
    Plays the experience used to evaluate elbw passive torques
    '''
    model = load_model_from_path(model_path)
    sim = MjSim(model)
    viewer = MjViewer(sim)

    joint_index = model.joint_name2id(joint_name)

    n_step_forward(init_steps, sim, viewer)

    for control_force, phase_nb_step in zip(control_forces, phases_nb_steps):
        sim.data.qfrc_applied[joint_index] = control_force
        n_step_forward(phase_nb_step, sim, viewer)


if __name__ == "__main__":
    # play
    PATH_TO_MODEL = os.path.join(
        gen_models_folder_path(), "muscles", "eval_elbow_flex_ext.xml")
    joint_name = "el_x"

    play_passive_torque_exp(joint_name, PATH_TO_MODEL, [400, 400], [2.5, -1.5])
