import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from mujoco_py import load_model_from_path, MjSim, MjViewer
from tqdm import tqdm
from numpy import linalg as LA

from shared_utils import get_project_root, models_folder_path, gen_models_folder_path
from shared_utils.xacro import xacro_to_xml
from soft_tissues_eval.utils import edit_MjModel_cmp_smoothness

matplotlib.use('TkAgg')


def PID_controller(pos_error, int_error, vel_error):
    '''
    pos_error.shape == int_error.shape == vel_error.shape == (4,)
    '''

    # compute PID-control output
    # TODO: use 4 dimensional coeffs?
    K = 50.0
    I = 25.0
    D = 15.0

    return K*pos_error + D*vel_error + I*int_error


def init_experiment_controller(sim, des_start_pos, des_start_vel, exo_joint_names, hum_joint_names, act_indexes):
    # # init simulation
    # sim = MjSim(model)
    # viewer = MjViewer(sim) if TOGGLE_VIEW else None 
    # for _ in range(250):
    #     sim.step()
    #     if viewer is not None:
    #         viewer.render()

    # vectorize joint functions
    vec_set_joint_qpos = np.vectorize(sim.data.set_joint_qpos)
    vec_get_joint_qpos = np.vectorize(sim.data.get_joint_qpos)
    vec_get_joint_qvel = np.vectorize(sim.data.get_joint_qvel)

    # HUMAN_INIT_CONFIG = np.array([0.01, 0.01, 0.01, vec_get_joint_qpos("el_x")])
    # HUMAN_INIT_CONFIG = np.array([vec_get_joint_qpos("J1"), vec_get_joint_qpos("J2"), vec_get_joint_qpos("J3"), vec_get_joint_qpos("el_x")])
    # _ = vec_set_joint_qpos(hum_joint_names, HUMAN_INIT_CONFIG)
    # sim.forward()

    # run control until close to desired starting position/velocity
    start_reached = False
    cancel = False
    pos_epsilon = np.deg2rad(1)
    vel_epsilon = np.deg2rad(0.25)
    
    # simulated joint pos, vel and integrated error are saved
    sim_joint_pos = np.zeros((0, 4))
    sim_joint_vel = np.zeros((0, 4))
    int_error = np.zeros(4)
    # simulated human joint positions
    hum_joint_pos = np.zeros((0, 4))
    
    iter = 0
    while not start_reached:
        # collect pos and vel
        sim_joint_pos = np.vstack(
            (sim_joint_pos, vec_get_joint_qpos(exo_joint_names)))
        sim_joint_vel = np.vstack(
            (sim_joint_vel, vec_get_joint_qvel(exo_joint_names)))
        hum_joint_pos = np.vstack(
            (hum_joint_pos, vec_get_joint_qpos(hum_joint_names)))

        # update integrated error
        # TODO: check what is the value of qvel at step 0
        pos_error = des_start_pos - sim_joint_pos[-1, :]
        int_error += pos_error*dt
        vel_error = des_start_vel - sim_joint_vel[-1, :]

        # compute torque given pid controller
        pid_tau = PID_controller(pos_error, int_error, vel_error)
        ctc_tau = sim.data.qfrc_bias[act_indexes]

        pid_tau = np.minimum(np.maximum(pid_tau, -60.0), 60.0)

        # set pid controller torque
        sim.data.ctrl[act_indexes] = (pid_tau + ctc_tau)/60

        for _ in range(n_steps):
            # HUMAN_INIT_CONFIG = np.array([vec_get_joint_qpos("gh_z"), vec_get_joint_qpos("gh_x"), vec_get_joint_qpos("J3"), vec_get_joint_qpos("el_x")]) #["gh_z", "gh_x", "gh_y", "el_x"]
            # _ = vec_set_joint_qpos(hum_joint_names, HUMAN_INIT_CONFIG)
            sim.step()
            iter += 1
            if viewer is not None:
                viewer.render()
        pos_close = np.abs(des_start_pos - sim_joint_pos[-1, :]) < pos_epsilon
        vel_close = np.abs(des_start_vel - sim_joint_vel[-1, :]) < vel_epsilon

        if iter > 2000:
            cancel = True

        if (all(pos_close) and all(vel_close)) or cancel:
            start_reached = True
            np.disp(all(pos_close) and all(vel_close))

    return sim, sim_joint_pos, sim_joint_vel, int_error, hum_joint_pos



def pid_controlled_experiment(sim, ref_joint_pos, ref_joint_vel, exo_joint_names, hum_joint_names, act_indexes, int_error):
    # # init simulation
    # sim = MjSim(model)
    # viewer = MjViewer(sim) if TOGGLE_VIEW else None
    # sim.forward()

    # vectorize joint functions
    vec_set_joint_qpos = np.vectorize(sim.data.set_joint_qpos)
    vec_get_joint_qpos = np.vectorize(sim.data.get_joint_qpos)
    vec_get_joint_qvel = np.vectorize(sim.data.get_joint_qvel)

    # # set start angles
    # _ = vec_set_joint_qpos(exo_joint_names, ref_joint_pos[0, :])
    # # TODO: properly compute human init pos (hardcoded right now)
    # _ = vec_set_joint_qpos(hum_joint_names, HUMAN_INIT_CONFIG)
    # sim.forward()

    # simulated joint pos, vel and integrated error are saved
    sim_joint_pos = np.zeros((0, 4))
    sim_joint_vel = np.zeros((0, 4))
    # int_error = np.zeros(4)

    # simulated human joint positions
    hum_joint_pos = np.zeros((0, 4))

    # debug/plot saved control outputs
    sim_applied_ctrl = np.zeros((0, 4))

    for des_joint_pos, des_joint_vel in tqdm(zip(ref_joint_pos, ref_joint_vel)):
        # collect pos and vel
        sim_joint_pos = np.vstack(
            (sim_joint_pos, vec_get_joint_qpos(exo_joint_names)))
        sim_joint_vel = np.vstack(
            (sim_joint_vel, vec_get_joint_qvel(exo_joint_names)))

        hum_joint_pos = np.vstack(
            (hum_joint_pos, vec_get_joint_qpos(hum_joint_names)))

        # update integrated error
        # TODO: check what is the value of qvel at step 0
        pos_error = des_joint_pos - sim_joint_pos[-1, :]
        int_error += pos_error*dt
        vel_error = des_joint_vel - sim_joint_vel[-1, :]

        # compute torque given pid controller
        pid_tau = PID_controller(pos_error, int_error, vel_error)
        pid_tau = np.minimum(np.maximum(pid_tau, -60.0), 60.0)
        ctc_tau = sim.data.qfrc_bias[act_indexes]

        # set pid controller torque
        sim.data.ctrl[act_indexes] = (pid_tau + ctc_tau) / 60
        sim_applied_ctrl = np.vstack((sim_applied_ctrl, ((pid_tau + ctc_tau)) ))


        for _ in range(n_steps):
            sim.step()

            if viewer is not None:
                viewer.render()

    return sim_joint_pos, sim_joint_vel, sim_applied_ctrl, hum_joint_pos


if __name__ == "__main__":
    TOGGLE_VIEW = False

    # load experimental data
    data_filename = "run1.csv"
    data_path = os.path.join(get_project_root(), "data",
                             "4_dofs_eval", data_filename)
    assert os.path.isfile(data_path)

    df = pd.read_csv(data_path, sep=",")
    ref_joint_pos = np.deg2rad(df.filter(regex="Angle$").to_numpy())

    # approximative start of the experiment (seconds)
    starting_point = 82.
    ref_joint_pos = ref_joint_pos[int(
        starting_point/0.01):, :]
    HUMAN_INIT_CONFIG = np.array([-0.691, -0.565, -0.59, 2.02])

    # measurements time step
    dt = 0.01  # torque update every hundredth of second

    # differentiate to get ref velocity
    ref_joint_vel = np.zeros_like(ref_joint_pos)
    ref_joint_vel[1:] = (ref_joint_pos[1:] - ref_joint_pos[:-1])/dt

    # debug store control vector over time
    # applied_control = np.zeros_like(ref_joint_pos)

    # parse xacro model
    xacro_path = os.path.join(models_folder_path(
    ), "exo_with_patient", "nesm_with_simple_patient.xacro")
    assert os.path.isfile(xacro_path)

    xacro_to_xml(xacro_path)

    # load mujoco model
    model_path = os.path.join(
        gen_models_folder_path(), "exo_with_patient", "nesm_with_simple_patient.xml")
    assert os.path.isfile(model_path)

    model = load_model_from_path(model_path)

    # hardcode joint and actuator names
    exo_act_names = ["sAA", "sFE", "sIE", "eFE"]
    exo_joint_names = ["J1", "J2", "J3", "J4"]
    hum_joint_names = ["gh_z", "gh_x", "gh_y", "el_x"]

    assert set(exo_act_names) <= set(model.actuator_names)
    assert set(exo_joint_names+hum_joint_names) <= set(model.joint_names)

    act_indexes = np.vectorize(model.actuator_name2id)(exo_act_names)

    # set actuator gears to one (easier torque setting later)
    # model.actuator_gear[act_indexes, 0] = np.ones(4)

    # count number of simulation steps per torque update
    n_steps = dt / model.opt.timestep
    assert (n_steps).is_integer()
    n_steps = int(n_steps)

    # define extreme solrefs
    HARD_SOLREF = np.array([-975, -111])
    SOFT_SOLREF = np.array([-100, -111])

    # upper arm and lower arm soft tissues prefix
    ua_cmp_prefix = "ua"
    la_cmp_prefix = "la"

    '''
    Experience in hard setting
    '''
    # set hard ua and la
    edit_MjModel_cmp_smoothness(model, ua_cmp_prefix, HARD_SOLREF)
    edit_MjModel_cmp_smoothness(model, la_cmp_prefix, HARD_SOLREF)

    # drive smoothly to start position
    start_pos = ref_joint_pos[0]
    start_vel = ref_joint_vel[0]

    # init simulation
    sim_hard = MjSim(model)
    viewer = MjViewer(sim_hard) if TOGGLE_VIEW else None 
    for _ in range(500):
        sim_hard.step()
        if viewer is not None:
            viewer.render()

    # drive system into the desired starting position and velocity 
    sim_hard, sim_init_pos, sim_init_vel, int_error, hum_hard_init = init_experiment_controller(sim_hard, start_pos, start_vel, exo_joint_names, hum_joint_names, act_indexes)
      
    #  # compare ref and real pos
    # fig4, axes4 = plt.subplots(2, 2, figsize=(15, 15))
    # time_array = dt*np.arange(0, sim_init_pos.shape[0])

    # # size of dots
    # s = np.full_like(sim_init_pos[:, 0], 5)

    # for i, ax in enumerate(axes4.reshape(-1)):
    #     # plot reference
    #     #ax.scatter(time_array, np.rad2deg(
    #     #    ref_joint_pos.T[i]), label="Ground Truth", s=s)
    #     ax.scatter(time_array, (np.ones_like(sim_init_pos)*np.rad2deg(start_pos)).T[i], label="Desired Start", s=s)

    #     # plot simulation
    #     ax.scatter(
    #         time_array, np.rad2deg(sim_init_pos[:, i]), label="Exo init", s=s)

    #     # plot human joint position
    #     ax.scatter(
    #         time_array, np.rad2deg(hum_hard_init[:, i]), label="Human init", s=s)
    #     ax.set_xlabel("Time (s)")
    #     ax.set_ylabel("Joint position (deg)")
    #     ax.set_title(model.actuator_names[i])
    #     ax.legend()

    # plt.subplots_adjust(wspace=.2, hspace=.3)
    # plt.show()

    # carry out pid controlled experiment, based on ref data
    sim_hard_pos, sim_hard_vel, sim_applied_ctrl, hum_hard_pos = pid_controlled_experiment(
    sim_hard, ref_joint_pos, ref_joint_vel, exo_joint_names, hum_joint_names, act_indexes, int_error)


    plt.rcParams.update({'font.size': 32})
    font = {'family' : 'normal',
            'size'   : 32}

    matplotlib.rc('font', **font)

    # compare ref and real pos
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    time_array = dt*np.arange(0, sim_hard_pos.shape[0])

    # size of dots
    s = np.full_like(sim_hard_pos[:, 0], 5)

    # Placeholder RMSE per joint
    rmse_hard = np.zeros(4)
    misalignment_rmse_hard = np.zeros(4)
    misalignment_hard = np.zeros_like(ref_joint_pos)


    for i, ax in enumerate(axes.reshape(-1)):
        # plot reference
        # ax.scatter(time_array, np.rad2deg(
        #     ref_joint_pos.T[i]), label="Ground Truth", s=s)

        # plot simulation
        ax.scatter(
            time_array, np.rad2deg(sim_hard_pos[:, i]), label="Exoskeleton ", s=s)

        # plot human joint position
        ax.scatter(
            time_array, np.rad2deg(hum_hard_pos[:, i]), label="Human (soft)", s=s)

        ax.set_xlabel("Time (s)") #, fontsize = 20.0
        ax.set_ylabel("Joint position (deg)") #, fontsize = 20.0
        ax.set_title("Experiment results - soft configuration")# #, size=20
        lgnd = ax.legend()
        lgnd.legendHandles[0]._sizes = [30]
        lgnd.legendHandles[1]._sizes = [30]
        # lgnd.legendHandles[2]._sizes = [30]

        rmse = np.rad2deg(ref_joint_pos.T[i]) - np.rad2deg(sim_hard_pos[:, i])
        misalignment = np.rad2deg(hum_hard_pos[:, i]) - np.rad2deg(sim_hard_pos[:, i])
        rmse_hard[i] = LA.norm(rmse)
        misalignment_rmse_hard[i] = LA.norm(misalignment)
        misalignment_hard.T[i] = np.sqrt(misalignment*misalignment)

    # # compare ref and real velocity
    # fig3, axes3 = plt.subplots(2, 2, figsize=(15, 15))
    # time_array = dt*np.arange(0, sim_joint_vel.shape[0])

    # # size of dots
    # s = np.full_like(sim_joint_vel[:, 0], 5)

    # for i, ax in enumerate(axes3.reshape(-1)):
    #     # plot reference
    #     ax.scatter(time_array, np.rad2deg(
    #         ref_joint_vel.T[i]), label="Ground Truth", s=s)

    #     # plot simulation
    #     ax.scatter(
    #         time_array, np.rad2deg(sim_joint_vel[:, i]), label="Simulation", s=s)

    #     ax.set_xlabel("Time (s)")
    #     ax.set_ylabel("Joint velocity (deg/s)")
    #     ax.set_title(model.actuator_names[i])
    #     ax.legend()

    # # show control output (applied ctrl to actuators)
    # fig2, axes2 = plt.subplots(2, 2, figsize=(15, 15))
    # time_array = dt*np.arange(0, sim_applied_ctrl.shape[0])

    # # size of dots
    # s = np.full_like(sim_applied_ctrl[:, 0], 5)

    # for i, ax in enumerate(axes2.reshape(-1)):
    #     # plot simulation
    #     ax.scatter(
    #         time_array, sim_applied_ctrl[:, i], label="PID control", s=s)

    #     ax.set_xlabel("Time (s)")
    #     ax.set_ylabel("Control output (Nm/rad)")
    #     ax.set_title(model.actuator_names[i])
    #     ax.legend()

    # # set same scale to all plots, not necessarily in the same limits
    # ax_with_lim = [(ax, ax.get_ylim()) for ax in axes.reshape(-1)]
    # max_y_delta = np.max(
    #     [lmax-lmin for _, (lmin, lmax) in ax_with_lim])

    # for ax, (lmin, lmax) in ax_with_lim:
    #     ax_delta = (max_y_delta - (lmax-lmin))/2
    #     ax.set_ylim(lmin-ax_delta, lmax+ax_delta)


    # Compute tracking error (exoskeleton)
    # plt.subplots_adjust(wspace=.2, hspace=.3)
    # plt.draw()
    print("RMSE (hard)", rmse_hard)
    print("Misalignment (hard)", misalignment_rmse_hard)


    '''
    Experience in soft setting
    '''
    # set soft ua and la
    edit_MjModel_cmp_smoothness(model, ua_cmp_prefix, SOFT_SOLREF)
    edit_MjModel_cmp_smoothness(model, la_cmp_prefix, SOFT_SOLREF)

    # init simulation
    sim_soft = MjSim(model)
    viewer = MjViewer(sim_soft) if TOGGLE_VIEW else None 
    for _ in range(500):
        sim_soft.step()
        if viewer is not None:
            viewer.render()

    # Placeholder RMSE per joint
    rmse_soft = np.zeros(4)
    misalignment_rmse_soft = np.zeros(4)
    misalignment_soft = np.zeros_like(ref_joint_pos)


    # drive system into the desired starting position and velocity 
    sim_soft, sim_init_pos, sim_init_vel, int_error, hum_soft_init = init_experiment_controller(sim_soft, start_pos, start_vel, exo_joint_names, hum_joint_names, act_indexes)
    # carry out pid controlled experiment, based on ref data
    sim_soft_pos, sim_soft_vel, sim_applied_ctrl, hum_soft_pos = pid_controlled_experiment(
        sim_soft, ref_joint_pos, ref_joint_vel, exo_joint_names, hum_joint_names, act_indexes, int_error)

    # compare ref and real pos
    fig, axes_soft = plt.subplots(2, 2, figsize=(15, 15))
    time_array = dt*np.arange(0, sim_soft_pos.shape[0])

    # size of dots
    s = np.full_like(sim_soft_pos[:, 0], 5)

    for i, ax in enumerate(axes_soft.reshape(-1)):
        # plot reference
        # ax.scatter(time_array, np.rad2deg(
        #     ref_joint_pos.T[i]), label="Desired Position", s=s)

        # plot simulation
        ax.scatter(
            time_array, np.rad2deg(sim_soft_pos[:, i]), label="Exoskeleton", s=s)
        # plot human joint position
        ax.scatter(
            time_array, np.rad2deg(hum_soft_pos[:, i]), label="Human (hard)", s=s)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Joint position (deg)")
        ax.set_title("Experiment results - hard configuration")#(model.actuator_names[i])
        lgnd = ax.legend()
        lgnd.legendHandles[0]._sizes = [30]
        lgnd.legendHandles[1]._sizes = [30]
        # lgnd.legendHandles[2]._sizes = [30]

        rmse = np.rad2deg(ref_joint_pos.T[i]) - np.rad2deg(sim_soft_pos[:, i])
        rmse_soft[i] = LA.norm(rmse)
        misalignment = np.rad2deg(hum_soft_pos[:, i]) - np.rad2deg(sim_soft_pos[:, i])
        misalignment_rmse_soft[i] = LA.norm(misalignment)
        misalignment_soft.T[i] = np.sqrt(misalignment*misalignment)


    # # set same scale to all plots, not necessarily in the same limits
    # ax_with_lim = [(ax, ax.get_ylim()) for ax in axes_soft.reshape(-1)]
    # max_y_delta = np.max(
    #     [lmax-lmin for _, (lmin, lmax) in ax_with_lim])

    # for ax, (lmin, lmax) in ax_with_lim:
    #     ax_delta = (max_y_delta - (lmax-lmin))/2
    #     ax.set_ylim(lmin-ax_delta, lmax+ax_delta)

    print("RMSE (soft)", rmse_soft)
    print("Misalignment (soft)", misalignment_rmse_soft)




    # compare kinematic misalignment for hard and soft configuration
    fig, axes_misalign = plt.subplots(2, 2, figsize=(15, 15))
    time_array = dt*np.arange(0, sim_soft_pos.shape[0])

    # size of dots
    s = np.full_like(sim_soft_pos[:, 0], 5)

    for i, ax in enumerate(axes_misalign.reshape(-1)):
        # plot hard kinematic misalignment
        ax.scatter(
            time_array, misalignment_hard.T[i], label="Hard", s=s)
        # plot soft kinematic misalignment
        ax.scatter(
            time_array, misalignment_soft.T[i], label="Soft", s=s)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Joint position error (deg)")
        ax.set_title(model.actuator_names[i]+hum_joint_names[i])
        lgnd = ax.legend()
        lgnd.legendHandles[0]._sizes = [30]
        lgnd.legendHandles[1]._sizes = [30]

    plt.subplots_adjust(wspace=.2, hspace=.3)
    plt.show()
