import os
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib

from mpl_toolkits.mplot3d import Axes3D
from mujoco_py import load_model_from_path, MjSim
from shared_utils import gen_models_folder_path
from soft_tissues_eval.utils import get_micro_bodies
from contact_forces.utils import get_contact_force, get_arm_support_geoms, get_strap_geoms_names

matplotlib.use('TkAgg')


if __name__ == "__main__":
    # init model and sim
    model_path = os.path.join(gen_models_folder_path(
    ), "exo_with_patient", "nesm_with_patient.xml")
    assert os.path.isfile(model_path)

    model = load_model_from_path(model_path)

    sim = MjSim(model)
    sim.forward()

    # run simulation for a few steps
    for _ in range(100):
        sim.step()

    # extract idx from geom of interest
    ua_prefix = "ua"

    # upper arm support geoms
    ua_sup_names = get_arm_support_geoms(model.geom_names, ua_prefix)
    ua_sup_ids = np.vectorize(model.geom_name2id)(ua_sup_names)

    # upper arm strap geoms
    ua_strap_names = get_strap_geoms_names(model.geom_names, "uarm_strap_")
    ua_strap_ids = np.vectorize(model.geom_name2id)(ua_strap_names)

    # upper arm soft colliding body geoms
    ua_soft_names = get_micro_bodies(model.geom_names, ua_prefix, "G")
    ua_soft_ids = np.vectorize(model.geom_name2id)(ua_soft_names)

    # ensures that for all soft body - arm strap contact, geom1 belongs to soft collider
    # assert all(
    #     [soft_id < strap_id for soft_id in ua_soft_ids for strap_id in ua_strap_ids])
    # ensures that for all soft body - arm support contact, geom1 belongs to soft collider
    # assert all(
    #     [soft_id < sup_id for soft_id in ua_soft_ids for sup_id in ua_sup_ids])

    softGeom_to_supForce = {}
    softGeom_to_strapForce = {}

    # extract contacts between (support or strap) and soft body
    for contact_idx in range(sim.data.ncon):
        contact = sim.data.contact[contact_idx]

        if contact.geom1 not in ua_soft_ids:
            continue

        dict_to_update = softGeom_to_strapForce if contact.geom2 in ua_strap_ids \
            else softGeom_to_supForce if contact.geom2 in ua_sup_ids else None

        if dict_to_update is None:
            continue

        # TODO handle 6D contacts

        if contact.geom1 not in dict_to_update:
            # we want the force applied by the support to geom1 -> we have a minus
            dict_to_update[contact.geom1] = -get_contact_force(
                model, sim.data, contact_idx)
        else:
            # we want the force applied by the support to geom1 -> we have a minus
            dict_to_update[contact.geom1] += -get_contact_force(
                model, sim.data, contact_idx)

    # plot everything
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # colors
    GREY = np.array([.1, .1, .1, .7])
    BLUE = np.array([.1, .1, .9, 1.])
    RED = np.array([.9, .1, .1, 1.])

    # draw micro geoms (in grey)
    ua_soft_pos = sim.data.geom_xpos[ua_soft_ids].T
    colors = np.full((len(ua_soft_ids), 4), GREY)

    Axes3D.scatter(ax, *ua_soft_pos, c=colors)

    # draw support arrow forces
    for soft_geom_id, force_arrow in softGeom_to_supForce.items():
        force_pos = sim.data.geom_xpos[soft_geom_id]
        ax.quiver(*force_pos, *force_arrow, length=0.01,
                  normalize=True, colors=BLUE)

    # draw strap arrow forces
    for soft_geom_id, force_arrow in softGeom_to_strapForce.items():
        force_pos = sim.data.geom_xpos[soft_geom_id]
        ax.quiver(*force_pos, *force_arrow, length=0.01,
                  normalize=True, colors=RED)

    plt.show()
