from mujoco_py import load_model_from_path, MjSim, MjViewer, MjViewerBasic, functions
import numpy as np
import matplotlib.pyplot as plt
import os

from shared_utils.mujoco import n_step_forward
from elbow_muscles_eval.utils import add_musc_force, add_joint_pos, add_joint_vel
from tqdm import tqdm
from sklearn.linear_model import LinearRegression


from shared_utils import get_project_root, models_folder_path, gen_models_folder_path
from shared_utils.xacro import xacro_to_xml
from contact_forces.ua_contact_analysis import get_arm_support_geoms, get_strap_geoms_names
from contact_forces.utils import get_contact_force
from soft_tissues_eval.utils import get_micro_bodies


import matplotlib
matplotlib.use('TkAgg')


def trajectory(t, x_0):
    '''
    Return the value of the trajectory at provided time step
    '''
    # time offset
    t_0 = 5

    # Sigmoid function for desired trajectory
    f = x_0 - ((x_0/1.1) / (1 + np.exp(-((t/.5) - t_0)))) - \
        ((x_0/1.1) / (1 + np.exp(-((0.0/.5) - t_0))))

    # Analytical differentiation for desired velocity
    df = -(2*(x_0/1.1)*np.exp(t_0-(t/.5))) / (np.exp(t_0-(t/.5)) + 1)**2 + \
        (2*(x_0/1.1)*np.exp(t_0-(0.0/.5))) / (np.exp(t_0-(0.0/.5)) + 1)**2

    # Analytical differentitation for desired acceleration
    ddf = (2*np.exp((t/.5)+t_0) * (np.exp((t/.5)) - np.exp(t_0))) / \
        (np.exp((t/.5)) + np.exp(t_0))**3

    return f, df, ddf


def PID_init_controller(pos_error, int_error, vel_error):
    '''
    pos_error.shape == int_error.shape == vel_error.shape == (4,)
    '''

    # compute PID-control output
    # TODO: use 4 dimensional coeffs?
    K = 50.0
    I = 25.0
    D = 15.0

    return K*pos_error + D*vel_error + I*int_error


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


'''
New controller used to bring arm into initial configuration
'''


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
    pos_epsilon = np.deg2rad(0.1)
    vel_epsilon = np.deg2rad(0.01)

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
        pid_tau = PID_init_controller(pos_error, int_error, vel_error)
        ctc_tau = sim.data.qfrc_bias[act_indexes]

        pid_tau = np.minimum(np.maximum(pid_tau, -60.0), 60.0)

        # set pid controller torque
        sim.data.ctrl[act_indexes] = (pid_tau)/60  # - ctc_tau

        for _ in range(n_steps):
            # HUMAN_INIT_CONFIG = np.array([vec_get_joint_qpos("gh_z"), vec_get_joint_qpos("gh_x"), vec_get_joint_qpos("J3"), vec_get_joint_qpos("el_x")]) #["gh_z", "gh_x", "gh_y", "el_x"]
            # _ = vec_set_joint_qpos(hum_joint_names, HUMAN_INIT_CONFIG)
            sim.step()
            iter += 1
            if viewer is not None:
                viewer.render()
        pos_close = np.abs(des_start_pos - sim_joint_pos[-1, :]) < pos_epsilon
        vel_close = np.abs(des_start_vel - sim_joint_vel[-1, :]) < vel_epsilon

        if iter > 5000:
            cancel = True

        if (all(pos_close) and all(vel_close)) or cancel:
            start_reached = True
            np.disp(all(pos_close) and all(vel_close))

    return sim, sim_joint_pos, sim_joint_vel, int_error, hum_joint_pos


def pid_controlled_experiment(sim, sim_nominal, ref_joint_pos, ref_joint_vel, ref_joint_acc, exo_joint_names, hum_joint_names, act_indexes, int_error):
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

    # list of contact bodies
    la_prefix = "la"
    # upper arm support geoms
    la_sup_ids = np.vectorize(model.geom_name2id)(
        get_arm_support_geoms(model.geom_names, la_prefix))

    # upper arm strap geoms
    la_strap_ids = np.vectorize(model.geom_name2id)(
        get_strap_geoms_names(model.geom_names, "larm_strap_"))

    # upper arm soft colliding body geoms
    la_soft_ids = np.vectorize(model.geom_name2id)(
        get_micro_bodies(model.geom_names, la_prefix, "G"))

    larm_body_id = model.body_name2id("ulna_r")

    # simulated joint pos, vel and integrated error are saved
    exo_joint_pos = np.zeros((0, 4))
    exo_joint_vel = np.zeros((0, 4))
    # int_error = np.array([int_error[0], int_error[1], int_error[2], 0.0]) #np.zeros(4)

    # simulated human joint positions and velocities
    hum_joint_pos = np.zeros((0, 4))
    hum_joint_vel = np.zeros((0, 4))

    # debug/plot saved control outputs
    pid_applied_ctrl = np.zeros((0, 4))
    ctc_applied_ctrl = np.zeros((0, 4))
    ff_applied_ctrl = np.zeros((0, 1))

    # store torque generated by human cocontracting
    cc_torque_hum = np.zeros((0, 4))
    training_torque = np.zeros((0, 4))

    # store lower arm contact forces arrows
    la_con_torques = np.zeros((0, 3))

    # flag for unintended high velocities, torques, ...
    flag_velocity = False

    for des_joint_pos, des_joint_vel, des_joint_acc in tqdm(zip(ref_joint_pos, ref_joint_vel, ref_joint_acc)):
        # collect pos and vel
        exo_joint_pos = np.vstack(
            (exo_joint_pos, vec_get_joint_qpos(exo_joint_names)))
        exo_joint_vel = np.vstack(
            (exo_joint_vel, vec_get_joint_qvel(exo_joint_names)))

        hum_joint_pos = np.vstack(
            (hum_joint_pos, vec_get_joint_qpos(hum_joint_names)))
        hum_joint_vel = np.vstack(
            (hum_joint_vel, vec_get_joint_qvel(hum_joint_names)))

        # update integrated error
        # TODO: check what is the value of qvel at step 0
        pos_error = des_joint_pos - exo_joint_pos[-1, :]
        int_error += pos_error*dt
        vel_error = des_joint_vel - exo_joint_vel[-1, :]

        # for joint in minimal_joint_list:
        #     sim_nominal.data.set_joint_qpos(joint, sim.data.get_joint_qpos(joint))
        #     sim_nominal.data.set_joint_qvel(joint, sim.data.get_joint_qvel(joint))

        # compute torque given pid controller
        pid_tau = PID_controller(pos_error, int_error, vel_error)
        pid_tau = np.minimum(np.maximum(pid_tau, -60.0), 60.0)
        ctc_tau = sim_nominal.data.qfrc_bias[act_indexes]

        # set nominal model to desired state for feedforward control
        nom_exo_el_id = nominal_model.joint_name2id("J4")
        sim_nominal.data.qpos[:] = vec_get_joint_qpos(
            exo_joint_names)  # des_joint_pos
        sim_nominal.data.qvel[:] = vec_get_joint_qvel(
            exo_joint_names)  # des_joint_vel
        # sim_nominal.data.qacc[:] = des_joint_acc

        # # Checkout difference between qfrc_bias und qfrc_inverse
        # functions.mj_inverse(sim_nominal.model, sim_nominal.data)
        # inverse_ctrl = sim_nominal.data.qfrc_inverse[nom_exo_el_id]
        # inverse_ctrl = np.minimum(np.maximum(inverse_ctrl, -60.0), 60.0)

        # set pid controller torque     np.array([pid_tau[0], pid_tau[1], pid_tau[2], inverse_ctrl]) / 60 #
        sim.data.ctrl[act_indexes] = (pid_tau) / 60
        # sim_nominal.data.ctrl[act_indexes] = (inverse_ctrl) / 60
        pid_applied_ctrl = np.vstack((pid_applied_ctrl, ((pid_tau))))
        ctc_applied_ctrl = np.vstack((ctc_applied_ctrl, ((ctc_tau))))
        # ff_applied_ctrl = np.vstack((ff_applied_ctrl, ((inverse_ctrl)) ))
        label = pid_tau - ctc_tau
        training_torque = np.vstack((training_torque, ((label))))
        cc_torque_hum = np.vstack((cc_torque_hum, sim.data.qfrc_actuator[np.vectorize(
            model.joint_name2id)(hum_joint_names)]))

        # extract contact forces
        la_step_con_torques = np.zeros((3, 0))
        el_joint_pos = sim.data.body_xpos[larm_body_id]

        for con_idx in range(sim.data.ncon):
            contact = sim.data.contact[con_idx]

            # con.geom1 < con.geom2
            # and la_soft_ids < la_strap_ids, la_sup_ids
            # implies for arm - soft body and arm
            if contact.geom1 not in la_soft_ids:
                continue

            # we want the force applied by the support to geom1 -> we have a minus
            if contact.geom2 in la_strap_ids or contact.geom2 in la_sup_ids:
                # get force
                la_con_force = -get_contact_force(
                    model, sim.data, con_idx)
                # compute torque (in global coordinates)
                geom_pos = sim.data.geom_xpos[contact.geom2]
                la_step_con_torques = np.hstack(
                    (la_step_con_torques, (np.cross(geom_pos - el_joint_pos, la_con_force)).reshape(3, 1)))

        # convert to local coordinates
        # rotation matrix must be transposed (mjc specifications)
        el_joint_rot_mat = sim.data.body_xmat[larm_body_id].reshape(3, 3).T
        la_con_torques = np.vstack(
            (la_con_torques, np.sum(el_joint_rot_mat@la_step_con_torques, axis=1)))

        if vec_get_joint_qvel(exo_joint_names)[3] > 2.0:
            flag_velocity = True
            break

        for _ in range(n_steps):
            sim.step()
            sim_nominal.step()

            if viewer is not None:
                viewer.render()

    return exo_joint_pos, exo_joint_vel, hum_joint_pos, hum_joint_vel, pid_applied_ctrl, ctc_applied_ctrl, cc_torque_hum, training_torque, flag_velocity, la_con_torques


# [ 0.87110449, -3.7113069 , -0.44155919,  5.91104743]) --> only fwd
# def autonmous_assessment(model_path, nominal_model_path):
if __name__ == "__main__":
    TOGGLE_VIEW = False

    '''
    Perturb the human joint and concurrently assess neuromechanics parameters
    '''
    minimal_joint_list = ["J1", "J2", "J3",
                          "J4"]  # ["gh_z", "gh_y", "gh_x", "el_x"]

    # parse xacro model
    # xacro_path = os.path.join(models_folder_path(
    # ), "exo_with_patient", "nesm_with_simple_patient.xacro")
    # assert os.path.isfile(xacro_path)

    # xacro_to_xml(xacro_path)

    # load mujoco model
    model_path = os.path.join(
        gen_models_folder_path(), "exo_with_patient", "nesm_with_simple_patient.xml")
    assert os.path.isfile(model_path)

    # load mujoco model
    nominal_model_path = os.path.join(
        models_folder_path(), "exo_with_patient", "fully_connected_simple.xml")
    assert os.path.isfile(nominal_model_path)

    # instantiate model
    model = load_model_from_path(model_path)
    nominal_model = load_model_from_path(nominal_model_path)

    # deactivate gravity in models
    model.opt.gravity[:] = np.zeros(3)
    nominal_model.opt.gravity[:] = np.zeros(3)

    sim = MjSim(model)
    sim_nominal = MjSim(nominal_model)

    # measurements time step
    dt = 0.01  # torque update every hundredth of second

    # count number of simulation steps per torque update
    n_steps = dt / model.opt.timestep
    assert (n_steps).is_integer()
    n_steps = int(n_steps)

    # hardcode joint and actuator names
    exo_act_names = ["sAA", "sFE", "sIE", "eFE"]
    exo_joint_names = ["J1", "J2", "J3", "J4"]
    hum_joint_names = ["gh_z", "gh_x", "gh_y", "el_x"]

    # TODO: what exactly is done here?
    assert set(exo_act_names) <= set(model.actuator_names)
    assert set(exo_joint_names+hum_joint_names) <= set(model.joint_names)
    act_indexes = np.vectorize(model.actuator_name2id)(exo_act_names)

    # initialisation of the contact straps
    viewer = MjViewer(sim) if TOGGLE_VIEW else None
    for _ in range(500):
        sim.step()
        if viewer is not None:
            viewer.render()

    # get index and names of flexors/extensors/joint
    elbow_flexors_names = ["bic_b_avg", "bic_l",
                           "brachialis_avg", "brachiorad_avg"]
    elbow_flexors_index = [model.actuator_name2id(
        muscle_name) for muscle_name in elbow_flexors_names]

    elbow_extensors_names = ["tric_long_avg",
                             "tric_med_avg", "tric_lat_avg", "anconeus_avg"]
    elbow_extensors_index = [model.actuator_name2id(
        muscle_name) for muscle_name in elbow_extensors_names]

    # set extensors and flexors to actuation for cocontraction
    print("Start cocontraction of muscles")
    for flexor_index, extensor_index in zip(elbow_flexors_index, elbow_extensors_index):
        sim.data.ctrl[flexor_index] = 0.4
        sim.data.ctrl[extensor_index] = 0.4

    sim.step()

    # for _ in range(500):
    #     cc_torques = sim.data.qfrc_actuator[np.vectorize(model.joint_name2id)(hum_joint_names)]
    #     sim.data.qfrc_applied[np.vectorize(model.joint_name2id)(hum_joint_names)] = -cc_torques
    #     sim.step()
    #     if viewer is not None:
    #         viewer.render()

    # TODO: Is this the way to compute cocontraction produced torques (sim.step necessaity questionable)
    # functions.mj_fwdActuation(sim.model, sim.data)

    # Ensure static position for all DoF not involved in elbow assessment
    # drive system into the desired starting position and velocity
    start_pos = np.array(
        [np.deg2rad(0.0), np.deg2rad(-90.0), np.deg2rad(0.0), np.deg2rad(110.0)])
    start_vel = np.array([0.0, 0.0, 0.0, 0.0])
    sim, sim_init_pos, sim_init_vel, int_error, hum_hard_init = init_experiment_controller(
        sim, start_pos, start_vel, exo_joint_names, hum_joint_names, act_indexes)

    print("Elbow position initialized...")

    # save joint position after initialization as startig point of trajectory
    init_joint_pos = sim.data.get_joint_qpos(exo_joint_names[3])

    # sync nominal and simulation model
    for joint in minimal_joint_list:
        sim_nominal.data.set_joint_qpos(joint, sim.data.get_joint_qpos(joint))
        sim_nominal.data.set_joint_qvel(joint, sim.data.get_joint_qvel(joint))
    sim_nominal.forward()

    # Generate complete reference position and velocity vector
    # Retrieve desired trajectory (position, velocity and acceleration)
    delay_time = sim.data.time
    traj_it = 0
    traj_time = 0.0  # delay_time
    ref_joint_pos = np.zeros((0, 4))
    ref_joint_vel = np.zeros((0, 4))
    while(traj_it*model.opt.timestep < 6.0):
        traj_time = traj_time + model.opt.timestep
        des_pos, des_vel, des_acc = trajectory(traj_time, init_joint_pos)
        # collect pos and vel
        ref_joint_pos = np.vstack(
            (ref_joint_pos, np.array([np.deg2rad(0.0), np.deg2rad(-90.0), np.deg2rad(0.0), des_pos])))
        ref_joint_vel = np.vstack(
            (ref_joint_vel, np.array([0.0, 0.0, 0.0, des_vel])))
        ref_joint_acc = np.vstack(
            (ref_joint_vel, np.array([0.0, 0.0, 0.0, des_acc])))
        traj_it += 1

    # carry out pid controlled experiment, based on ref data #TODO: return hum_vel
    print("Start autonmous assessment...")
    exo_pos_values, exo_vel_values, hum_pos_values, hum_vel_values, pid_control_torque, ctc_control_torque, hum_muscle_torque, training_data_torque, velocity_flag, la_con_torques = pid_controlled_experiment(
        sim, sim_nominal, ref_joint_pos, ref_joint_vel, ref_joint_acc, exo_joint_names, hum_joint_names, act_indexes, int_error)

    print("Vecoity Flag: " + str(velocity_flag))

    # Compute/Learn paramters from training data
    reg_data = LinearRegression().fit(
        hum_pos_values[:, 3].reshape(-1, 1), -training_data_torque[:, 3])
    score = reg_data.score(
        hum_pos_values[:, 3].reshape(-1, 1), -training_data_torque[:, 3])
    print(f"R2 score of regression: {score}")
    stiffness = reg_data.coef_
    print(f"Estimated joint stiffness: {stiffness}")
    regression_predictions = reg_data.predict(
        hum_pos_values[:, 3].reshape(-1, 1))

    # Compute/Learn paramters from training data
    reg_gt = LinearRegression().fit(
        hum_pos_values[:, 3].reshape(-1, 1), hum_muscle_torque[:, 3])
    score_gt = reg_gt.score(
        hum_pos_values[:, 3].reshape(-1, 1), hum_muscle_torque[:, 3])
    print(f"R2 score of ground truth: {score_gt}")
    stiffness_gt = reg_gt.coef_
    print(f"Estimated ground truth: {stiffness_gt}")
    regression_ground_truth = reg_gt.predict(
        hum_pos_values[:, 3].reshape(-1, 1))

    # compute time array
    length_measuement = exo_pos_values[:, 3].shape[0]
    time = model.opt.timestep * np.arange(0, length_measuement)

    _, axes_3 = plt.subplots()
    axes_3.plot(time, la_con_torques[:, 1],
                label="contact torque")
    axes_3.plot(
        time, hum_muscle_torque[:, -1] - ctc_control_torque[:, 3], label="human torque without contact")
    # axes_3.plot(
    #     time, ctc_control_torque[:, 3]/estimated_lever_arm, label="gravity compensation")
    axes_3.plot(
        time, -(pid_control_torque[:, 3] + ctc_control_torque[:, 3]), label="exo torque without contact")
    axes_3.set_xlabel("time (s)")
    axes_3.set_ylabel("force (N)")
    axes_3.set_title("Lower arm contact forces norm")

    axes_3.legend(loc="upper right")

    plt.show()
