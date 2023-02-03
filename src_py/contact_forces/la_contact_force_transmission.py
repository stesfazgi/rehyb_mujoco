'''
Analysing the evoltion of contact forces on lower arm during a basic flexion movement
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from mujoco_py import load_model_from_path, MjSim, MjViewer
from shared_utils import gen_models_folder_path, n_step_forward
from soft_tissues_eval.utils import get_micro_bodies
from contact_forces.utils import get_contact_force
from contact_forces.ua_contact_analysis import get_arm_support_geoms, get_strap_geoms_names

matplotlib.use('TkAgg')

if __name__ == "__main__":
    TOGGLE_VIEWER = False

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
    sim_delta_t = 300
    meas_freq = 10
    sim_rel_delta = sim_delta_t // meas_freq

    # extract s_FE joint id
    sFE_joint_name = 'J2'
    sIE_joint_name = 'J3'

    # index of lower arm body
    larm_body_id = model.body_name2id("ulna_r")

    # impose horizontal arm via joint contraints
    model.jnt_range[model.joint_name2id(sIE_joint_name)] = [-.01, .01]
    model.jnt_range[model.joint_name2id(
        sFE_joint_name)] = [-np.pi/2, -np.pi/2+.01]

    # initialize torque on elbow from exo side
    initial_torque = -.7
    sim.data.ctrl[model.actuator_name2id('eFE')] = initial_torque
    torque_array = initial_torque - np.arange(sim_rel_delta)*.1

    # initialize simulation
    n_step_forward(100, sim, viewer)

    # save data
    strap_conforce_save = np.zeros((3, sim_rel_delta))
    sup_conforce_save = np.zeros((3, sim_rel_delta))
    lever_arm_axis = np.zeros((3, sim_rel_delta))

    for cf_idx in range(sim_rel_delta):
        # increase exo elbow torque
        sim.data.ctrl[model.actuator_name2id('eFE')] = torque_array[cf_idx]

        # compute and merge contact forces
        for con_idx in range(sim.data.ncon):
            contact = sim.data.contact[con_idx]

            # for all contact con: con.geom1 < con.geom2
            # from model: la_soft_ids < la_strap_ids and la_soft_ids < la_sup_ids
            if contact.geom1 not in la_soft_ids:
                continue

            # we want the force applied by the support to geom1 -> we have a minus
            if contact.geom2 in la_strap_ids:
                strap_conforce_save[:, cf_idx] -= get_contact_force(
                    model, sim.data, con_idx)
            elif contact.geom2 in la_sup_ids:
                sup_conforce_save[:,
                                  cf_idx] -= get_contact_force(model, sim.data, con_idx)

        # save axis of lever arm (ie z of lower arm)
        lever_arm_axis[:, cf_idx] = sim.data.body_xmat[larm_body_id, 6:9]

        # simulate further
        for _ in range(meas_freq):
            sim.step()
            if TOGGLE_VIEWER:
                viewer.render()

    # offset strap and support contact forces
    sup_conforce_save = sup_conforce_save - \
        (sup_conforce_save[:, 0]).reshape(3, 1)
    strap_conforce_save = strap_conforce_save - \
        (strap_conforce_save[:, 0]).reshape(3, 1)

    # plot results
    time_steps = np.arange(sim_rel_delta)

    # compute norm of contact forces projected on lower arm z
    con_force_tot = strap_conforce_save+sup_conforce_save
    con_force_tot = np.sum(con_force_tot*lever_arm_axis,
                           axis=0).reshape(1, -1) * lever_arm_axis

    con_force_norm = np.linalg.norm(
        con_force_tot, axis=0)

    # compute reference for contact force
    estimated_lever_arm = .13
    theoretical_contact_force = np.abs(
        torque_array - initial_torque) / estimated_lever_arm

    plt.figure(figsize=(12, 8))
    plt.plot(time_steps, con_force_norm, label="Empirical contact force")
    plt.plot(time_steps, theoretical_contact_force,
             label="Theoretical contact force")
    plt.xlabel('time step')
    plt.ylabel('contact force norm (N)')
    plt.legend()

    plt.show()
