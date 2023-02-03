from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
import matplotlib.pyplot as plt
import os

from shared_utils.mujoco import n_step_forward
from shared_utils.general import gen_models_folder_path
from elbow_muscles_eval.utils import add_musc_force, add_joint_pos, add_joint_vel

import matplotlib
matplotlib.use('TkAgg')


def trajectory(t, x_0):
    '''
    Return the value of the trajectory at provided time step
    '''
    # Sigmoid function for desired trajectory
    f = x_0 - (x_0 / (1 + np.exp(-((t/2) - 5))))

    # Analytical differentiation for desired velocity
    df = -(x_0*np.exp(5-(t/2))) / (np.exp(5-(t/2)) + 1)**2
    return f, df


def PID_controller(des_x, des_dx, exo_x, exo_dx, i_error):
    '''
    Perform robotic assessment by perturbing the human joint along a predefined trajectory - for now 1 DoF
    '''
    # return desired position and velocity
    # des_x, des_dx = trajectory(time_step)

    # compute PID-control output
    K = 50.0
    I = 15.0
    D = 5.0
    e = des_x - exo_x
    de = des_dx - exo_dx
    tau = K*e + D*de + I*i_error

    return tau


def hold_arm_static(shoulderAA_pos, shoulderAA_vel, shoulderFE_pos, shoulderFE_vel, shoulderIE_pos, shoulderIE_vel):
    '''
    Use PID controllers to hold remaining DoF of exo arm in static position
    '''
    des_shoulderAA_pos = 0.0
    des_shoulderAA_vel = 0.0

    des_shoulderFE_pos = 0.0  # -np.pi/2
    des_shoulderFE_vel = 0.0

    des_shoulderIE_pos = 0.4
    des_shoulderIE_vel = 0.0

    # compute PID-control output
    K = 50.0
    D = 5.0

    e1 = des_shoulderAA_pos - shoulderAA_pos
    de1 = des_shoulderAA_vel - shoulderAA_vel
    tauAA = K*e1 + D*de1

    e2 = des_shoulderFE_pos - shoulderFE_pos
    de2 = des_shoulderFE_vel - shoulderFE_vel
    tauFE = K*e2 + D*de2

    e3 = des_shoulderIE_pos - shoulderIE_pos
    de3 = des_shoulderIE_vel - shoulderIE_vel
    tauIE = K*e3 + D*de3

    return (tauAA/60), (tauFE/60), (tauIE/30)


def autonmous_assessment(model_path):
    '''
    Perturb the human joint and concurrently assess neuromechanics parameters
    '''
    # elbow joint assessment
    hum_joint_name = "el_x"
    exo_joint_name = "J4"

    shoulderAA = "J1"
    exo_sAA = "sAA"
    shoulder_FE = "J2"
    exo_sFE = "sFE"
    shoulder_IE = "J3"
    exo_sIE = "sIE"

    # instantiate model
    model = load_model_from_path(model_path)
    sim = MjSim(model)
    viewer = MjViewer(sim)

    hum_joint_index = model.joint_name2id(hum_joint_name)
    exo_joint_index = model.joint_name2id(exo_joint_name)

    # initialisation of the contact straps
    print("Wait for contact ropes to set into place...")
    init_pos_steps = 250
    n_step_forward(init_pos_steps, sim)
    #init_torque = 10
    #sim.data.qfrc_applied[joint_index] = init_torque

    # Ensure static position for all DoF not involved in elbow assessment
    for i in range(250):
        shoulderAA_pos = sim.data.get_joint_qpos(shoulderAA)
        shoulderAA_vel = sim.data.get_joint_qvel(shoulderAA)
        shoulderFE_pos = sim.data.get_joint_qpos(shoulder_FE)
        shoulderFE_vel = sim.data.get_joint_qvel(shoulder_FE)
        shoulderIE_pos = sim.data.get_joint_qpos(shoulder_IE)
        shoulderIE_vel = sim.data.get_joint_qvel(shoulder_IE)
        controlAA, controlFE, controlIE = hold_arm_static(
            shoulderAA_pos, shoulderAA_vel, shoulderFE_pos, shoulderFE_vel, shoulderIE_pos, shoulderIE_vel)
        sim.data.ctrl[model.actuator_name2id(exo_sAA)] = controlAA
        sim.data.ctrl[model.actuator_name2id(exo_sFE)] = controlFE
        sim.data.ctrl[model.actuator_name2id(exo_sIE)] = controlIE

    print("Initializing elbow position...")
    init_pos = np.pi*2/3
    #init_pos_steps = 250
    # n_step_forward(init_pos_steps, sim, viewer)
    integrated_error = 0

    while(np.abs(sim.data.get_joint_qpos(exo_joint_name) - init_pos) > 0.035):  # error smaller than 2 deg
        error = init_pos - sim.data.get_joint_qpos(exo_joint_name)
        integrated_error += error
        sim.data.qfrc_applied[exo_joint_index] = 25*error + 5*integrated_error
        sim.step()
        viewer.render()

    # save joint position after initialization as startig point of trajectory
    init_joint_pos = sim.data.get_joint_qpos(exo_joint_name)

    sim.data.qfrc_applied[exo_joint_index] = 0

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
    for flexor_index, extensor_index in zip(elbow_flexors_index, elbow_extensors_index):
        sim.data.ctrl[flexor_index] = 0.66
        sim.data.ctrl[extensor_index] = 0.66

    # set muscles actuation to start cocontraction
    # for i in range(len(sim.data.ctrl)):
    #    sim.data.ctrl[i] = 1.

    # cocontraction_steps = 200
    # print("Start cocontraction...")
    # n_step_forward(cocontraction_steps, sim, viewer)

    # # activate external extension torque
    # extension_torque = -50
    # sim.data.qfrc_applied[joint_index] = extension_torque

    # store relevant simulation values
    musc_torque = []
    hum_pos_values = []
    hum_vel_values = []
    exo_pos_values = []
    exo_vel_values = []
    reference_pos = []
    reference_vel = []

    # low pass filter for control inputs
    mov_avg_eFE = [0]*10

    exo_eFE = "eFE"
    i_err = 0.0
    delay_time = sim.data.time
    traj_it = 0
    traj_time = delay_time
    print("Start autonmous assessment...")
    while (traj_it*model.opt.timestep < 20.0):  # sim.data.time
        # Ensure static position for all DoF not involved in elbow assessment
        shoulderAA_pos = sim.data.get_joint_qpos(shoulderAA)
        shoulderAA_vel = sim.data.get_joint_qvel(shoulderAA)
        shoulderFE_pos = sim.data.get_joint_qpos(shoulder_FE)
        shoulderFE_vel = sim.data.get_joint_qvel(shoulder_FE)
        shoulderIE_pos = sim.data.get_joint_qpos(shoulder_IE)
        shoulderIE_vel = sim.data.get_joint_qvel(shoulder_IE)
        controlAA, controlFE, controlIE = hold_arm_static(
            shoulderAA_pos, shoulderAA_vel, shoulderFE_pos, shoulderFE_vel, shoulderIE_pos, shoulderIE_vel)
        sim.data.ctrl[model.actuator_name2id(exo_sAA)] = controlAA
        sim.data.ctrl[model.actuator_name2id(exo_sFE)] = controlFE
        sim.data.ctrl[model.actuator_name2id(exo_sIE)] = controlIE

        hum_elbow_pos = sim.data.get_joint_qpos(hum_joint_name)
        hum_elbow_vel = sim.data.get_joint_qvel(hum_joint_name)
        exo_elbow_pos = sim.data.get_joint_qpos(exo_joint_name)
        exo_elbow_vel = sim.data.get_joint_qvel(exo_joint_name)

        # Move exo along desired trajectory using PID controller
        traj_time = sim.data.time - delay_time
        des_pos, des_vel = trajectory(traj_time, init_joint_pos)
        i_err += (des_pos - exo_elbow_pos)
        control_input = PID_controller(
            des_pos, des_vel, exo_elbow_pos, exo_elbow_vel, i_err)
        mov_avg_eFE.append((control_input/30))
        mov_avg_eFE.pop(0)
        sim.data.ctrl[model.actuator_name2id(exo_eFE)] = (
            control_input / 30)  # np.mean(mov_avg_eFE)

        sim.step()
        # viewer.render()
        traj_it += 1

        # sim.data.qfrc_actuator[exo_joint_index]  (np.mean(mov_avg_eFE)) #add_musc_force(sim.data, musc_torque, hum_joint_index)
        musc_torque.append(sim.data.qfrc_actuator[exo_joint_index])
        reference_pos.append(des_pos)
        reference_vel.append(des_vel)
        add_joint_pos(sim.data, hum_pos_values, hum_joint_name)
        add_joint_pos(sim.data, exo_pos_values, exo_joint_name)
        add_joint_vel(sim.data, hum_vel_values, hum_joint_name)
        add_joint_vel(sim.data, exo_vel_values, exo_joint_name)

    # compute time array
    time = model.opt.timestep * np.arange(0, traj_it)

    print("Plot results of assessment...")
    # plot time x muscular_torque
    _, axes = plt.subplots(3, 1)
    axes[0].plot(time, musc_torque)
    axes[0].set_xlabel("Time (s)")
    #axes[0].set_ylabel("Muscular torque (Nm)")
    axes[0].set_title("eFE applied control torque (Nm)")

    axes[1].plot(time, hum_pos_values)
    axes[1].plot(time, exo_pos_values)
    axes[1].plot(time, reference_pos)
    #axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("positions (rad)")
    #axes[1].set_title("Extension of cocontracted muscle (extension torque)")

    axes[2].plot(time, hum_vel_values)
    axes[2].plot(time, exo_vel_values)
    axes[2].plot(time, reference_vel)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("velocitites (rad/s)")
    #axes[2].set_title("Extension of cocontracted muscle (extension torque)")
    plt.show()


if __name__ == "__main__":
    # play
    PATH_TO_MODEL = os.path.join(
        gen_models_folder_path(), "exo_with_patient", "nesm_with_patient.xml")
    assert os.path.isfile(PATH_TO_MODEL)

    print(PATH_TO_MODEL)
    autonmous_assessment(PATH_TO_MODEL)
