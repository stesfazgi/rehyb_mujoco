'''
Analysing the evoltion of contact forces on lower arm during a basic flexion movement
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from mujoco_py import load_model_from_path, MjSim, MjViewer
from shared_utils import gen_models_folder_path
from soft_tissues_eval.utils import get_micro_bodies
from contact_forces.utils import get_contact_force
from contact_forces.ua_contact_analysis import get_arm_support_geoms, get_strap_geoms_names

matplotlib.use('TkAgg')

if __name__ == "__main__":
    TOGGLE_VIEWER = True

    # init model and sim
    model_path = os.path.join(gen_models_folder_path(
    ), "exo_with_patient", "nesm_with_patient.xml")
    assert os.path.isfile(model_path)

    model = load_model_from_path(model_path)

    sim = MjSim(model)
    viewer = MjViewer(sim) if TOGGLE_VIEWER else None
    sim.forward()

    # extract idx from geom of interest
    la_prefix = "la"

    # upper arm support geoms
    la_sup_names = get_arm_support_geoms(model.geom_names, la_prefix)
    la_sup_ids = np.vectorize(model.geom_name2id)(la_sup_names)

    # upper arm strap geoms
    la_strap_names = get_strap_geoms_names(model.geom_names, "larm_strap_")
    la_strap_ids = np.vectorize(model.geom_name2id)(la_strap_names)

    # upper arm soft colliding body geoms
    la_soft_names = get_micro_bodies(model.geom_names, la_prefix, "G")
    la_soft_ids = np.vectorize(model.geom_name2id)(la_soft_names)

    # simulation length / measurement frequency
    sim_delta_t = 500
    meas_freq = 10
    sim_rel_delta = sim_delta_t // meas_freq

    # extract s_FE joint id
    sFE_joint_name = 'J2'
    sIE_joint_name = 'J3'
    # eFE_joint_name = 'J4'

    # impose horizontal arm
    sim.data.ctrl[model.actuator_name2id('sFE')] = -10.
    # sim.data.ctrl[model.actuator_name2id('eFE')] = -.7

    # save data
    sFE_pos_save = np.zeros(sim_rel_delta)
    sIE_pos_save = np.zeros(sim_rel_delta)
    strap_conforce_save = np.zeros((3, sim_rel_delta))
    sup_conforce_save = np.zeros((3, sim_rel_delta))

    for cf_idx in range(sim_rel_delta):
        # extract joint pos
        sFE_pos_save[cf_idx] = sim.data.get_joint_qpos(sFE_joint_name)
        sIE_pos_save[cf_idx] = sim.data.get_joint_qpos(sIE_joint_name)

        # compute and merge contact forces
        strap_conforce = np.zeros(3)
        sup_conforce = np.zeros(3)

        for con_idx in range(sim.data.ncon):
            contact = sim.data.contact[con_idx]

            # con.geom1 < con.geom2
            # and la_soft_ids < la_strap_ids, la_sup_ids
            # implies for arm - soft body and arm
            if contact.geom1 not in la_soft_ids:
                continue

            # we want the force applied by the support to geom1 -> we have a minus
            if contact.geom2 in la_strap_ids:
                strap_conforce -= get_contact_force(
                    model, sim.data, con_idx)
            elif contact.geom2 in la_sup_ids:
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
    sFE_pos_save = np.rad2deg(sFE_pos_save)
    sIE_pos_save = np.rad2deg(sIE_pos_save)

    # plot results
    time_steps = np.arange(sim_rel_delta)

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(64, 8))
    # plt.subplots_adjust(hspace=.5)
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

    # plot sFE and sIE position
    axes[0, 0].plot(time_steps, sFE_pos_save)
    axes[0, 0].set(xlabel='time step', ylabel='pos (deg)', title='Exo sFE')

    axes[1, 0].plot(time_steps, sIE_pos_save)
    axes[1, 0].set(xlabel='time step', ylabel='pos (deg)', title='Exo sIE')

    # plot support forces
    axis_names = ["Sup Con X", "Sup Con Y", "Sup Con Z"]
    for idx, name in enumerate(axis_names):
        axes[0, idx+1].plot(time_steps, sup_conforce_save[idx])
        axes[0, idx+1].set(xlabel='time step', ylabel='force (N)', title=name)

    # plot strap forces
    axis_names = ["Strap Con X", "Strap Con Y", "Strap Con Z"]
    for idx, name in enumerate(axis_names):
        axes[1, idx+1].plot(time_steps, strap_conforce_save[idx])
        axes[1, idx+1].set(xlabel='time step', ylabel='force (N)', title=name)

    # fig.tight_layout()

    plt.show()

    # basic validation with exoskeleton maintaining horizontal arm -> evaluate contact forces
    # decomposition strap / support contact forces
