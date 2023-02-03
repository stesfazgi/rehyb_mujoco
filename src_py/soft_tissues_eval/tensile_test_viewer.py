import matplotlib.pyplot as plt
from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np

from shared_utils.mujoco import n_step_forward
from shared_utils.general import models_folder_path

import os

import matplotlib
matplotlib.use('TkAgg')

if __name__ == "__main__":
    TOGGLE_VIEW = False

    tensile_abs_path = os.path.join(
        models_folder_path(), "soft_tissues", "generated_tensile_test.xml")
    assert os.path.isfile(tensile_abs_path)

    model = load_model_from_path(tensile_abs_path)
    sim = MjSim(model)

    viewer = MjViewer(sim) if TOGGLE_VIEW else None

    collider_name = "test_collider"
    collider_index = model.body_names.index(collider_name)

    ctrls = .1*np.arange(0, 10)

    pos = []

    for ctrl in ctrls:
        sim.data.ctrl[0] = ctrl

        n_step_forward(1000, sim, viewer)

        pos.append(sim.data.body_xpos[collider_index][0])

    plt.scatter(ctrls, pos)

    plt.xlabel("Force (normalized)")
    plt.ylabel("Collider center position (m)")

    plt.show()
