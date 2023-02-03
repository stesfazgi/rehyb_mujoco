import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from mujoco_py import load_model_from_path, MjSim, MjViewer
from shared_utils import models_folder_path
from soft_tissues_eval.utils import get_micro_bodies
from contact_forces.utils import get_contact_force
from contact_forces.ua_contact_analysis import get_arm_support_geoms, get_strap_geoms_names

matplotlib.use('TkAgg')

if __name__ == "__main__":
    TOGGLE_VIEWER = False

    # init model and sim
    model_path = os.path.join(models_folder_path(
    ), "contact_eval", "strap_lying_on_soft.xml")
    assert os.path.isfile(model_path)

    model = load_model_from_path(model_path)

    sim = MjSim(model)
    viewer = MjViewer(sim) if TOGGLE_VIEWER else None
    sim.forward()

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

    # simulation length / measurement frequency
    sim_delta_t = 500
    meas_freq = 10
    sim_rel_delta = sim_delta_t // meas_freq

    # extract s_FE joint id

    # impose horizontal arm

    # save data
    strap_conforce_save = np.zeros((3, sim_rel_delta))
    sup_conforce_save = np.zeros((3, sim_rel_delta))

    for cf_idx in range(sim_rel_delta):
        # extract joint pos

        # compute and merge contact forces
        strap_conforce = np.zeros(3)
        sup_conforce = np.zeros(3)

        for con_idx in range(sim.data.ncon):
            contact = sim.data.contact[con_idx]

            if contact.geom1 not in ua_soft_ids:
                continue

            # we want the force applied by the support to geom1 -> we have a minus
            if contact.geom2 in ua_strap_ids:
                force_torque = get_contact_force(
                    model, sim.data, con_idx, True)
                strap_conforce -= force_torque[:, 0]

            elif contact.geom2 in ua_sup_ids:
                sup_conforce -= get_contact_force(model, sim.data, con_idx)

        # append to cache
        strap_conforce_save[:, cf_idx] = strap_conforce
        sup_conforce_save[:, cf_idx] = sup_conforce

        # simulate further
        for _ in range(meas_freq):
            sim.step()
            if TOGGLE_VIEWER:
                viewer.render()

    # convert pos to deg

    # plot results
    time_steps = np.arange(sim_rel_delta)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(64, 8))
    # plt.subplots_adjust(hspace=.5)
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

    # plot sFE and sIE position

    # plot support forces
    axis_names = ["Sup Con X", "Sup Con Y", "Sup Con Z"]
    for idx, name in enumerate(axis_names):
        axes[0, idx].plot(time_steps, sup_conforce_save[idx])
        axes[0, idx].set(xlabel='time step', ylabel='force (N)', title=name)

    # plot strap forces
    axis_names = ["Strap Con X", "Strap Con Y", "Strap Con Z"]
    for idx, name in enumerate(axis_names):
        axes[1, idx].plot(time_steps, strap_conforce_save[idx])
        axes[1, idx].set(xlabel='time step', ylabel='force (N)', title=name)

    # fig.tight_layout()

    plt.show()

    # basic validation with exoskeleton maintaining horizontal arm -> evaluate contact forces
    # decomposition strap / support contact forces
