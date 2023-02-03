'''
Small demo plotting the contact forces of two boxes, starting at the contact point
'''

from mujoco_py import load_model_from_path, MjSim
from shared_utils import models_folder_path
from contact_forces.utils import get_contact_force
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


if __name__ == "__main__":
    model_path = os.path.join(models_folder_path(),
                              "contact_eval", "contact_test.xml")
    assert os.path.isfile(model_path)

    # load model and sim
    model = load_model_from_path(model_path)
    sim = MjSim(model)
    sim.forward()

    # reach contact
    for _ in range(100):
        sim.step()

    assert sim.data.ncon == 4

    # retrieve contact forces
    top_box_name = "top_box"
    bottom_box_name = "bottom_box"

    top_box_id = model.geom_name2id(top_box_name)
    bottom_box_id = model.geom_name2id(bottom_box_name)

    contact_arrows = []

    # 6 is necessary too prevent seg fault, even though we only need 3
    force_tmp = np.zeros(6)

    for contact_idx in range(sim.data.ncon):
        contact = sim.data.contact[contact_idx]

        if contact.geom1 != bottom_box_id or contact.geom2 != top_box_id:
            continue

        # compute force in world frame
        force_array = get_contact_force(model, sim.data, contact_idx)

        # contact_arrows = [(*)]
        contact_arrows.append((*contact.pos, *force_array))

    # plot everything
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    for contact_arrow in contact_arrows:
        ax.quiver(*contact_arrow, length=0.01, normalize=True)

    plt.show()
