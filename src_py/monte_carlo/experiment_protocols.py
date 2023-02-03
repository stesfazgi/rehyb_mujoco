import numpy as np

from sklearn.linear_model import LinearRegression
from mujoco_py import MjSim
from typing import Final
from enum import IntEnum

from contact_forces.utils import get_contact_force
from contact_forces.utils import get_arm_support_geoms, get_strap_geoms_names
from soft_tissues_eval.utils import get_micro_bodies

''' Model static names'''
# Hardcoded names - steming from model
EXO_JOINTS: Final[list] = ["J1", "J2", "J3", "J4"]
HUM_JOINTS: Final[list] = ["gh_z", "gh_x", "gh_y", "el_x"]
EXO_ACTS: Final[list] = ["sAA", "sFE", "sIE", "eFE"]
MUSCLES: Final[list] = ["bic_b_avg", "bic_l",
                        "brachialis_avg", "brachiorad_avg", "tric_long_avg",
                        "tric_med_avg", "tric_lat_avg", "anconeus_avg"]

'''Experiment init'''
# desired starting position and velocity of the exoskeleton
START_CONF: Final[np.ndarray] = np.array(
    # joint pos
    [np.deg2rad([0.0, -90., 0, 110.]),
     # joint vel
     np.zeros(4)]
).T
# tolerance w.r.t. initialization pos reached
INIT_TOL: Final[np.ndarray] = np.deg2rad([
    # pos epsilon; .1 in legacy grid mode
    1.,
    # vel epsilon; .01 in legacy grid mode
    .25,
])
# length of the passive init allowing strap attachment
# TODO-HARD: find optimal value
STRAP_INIT: Final[int] = 500

'''Experiment specifications'''
# time interval between two torque measurements
MEAS_DT: Final[float] = .01
EXP_TIME_BOUNDS: Final[tuple] = (0., 6.+MEAS_DT)

# coeff of the pid controller
PID_COEFF: Final[np.ndarray] = np.array([50., 15., 25.])
# abs max torque
MAX_TORQUE: Final[float] = 60.
# cocontraction muscular actuation level
CC_LEVEL: Final[float] = .4
# nominal max con force on la during exp
MAX_CON_FORCE: Final[float] = 283.04270030814166

FAILED_INIT_OUT: Final[tuple] = ((0., 0., 0.), (False, True))
FAILED_EXP_OUT: Final[tuple] = ((0., 0., 0.), (True, False))


class StaticData():
    '''
    Wrapper of static data depending exclusively on the model
    that don't need to be recomputed
    '''

    def __init__(self, model) -> None:
        # exo joints ids
        self.ej_ids = np.vectorize(
            model.joint_name2id)(EXO_JOINTS)
        # hum joints ids
        self.hj_ids = np.vectorize(
            model.joint_name2id)(HUM_JOINTS)
        # exo actuators ids
        self.ea_ids = np.vectorize(model.actuator_name2id)(
            EXO_ACTS)
        # elbow muscles ids
        self.em_ids = np.vectorize(model.actuator_name2id)(MUSCLES)
        # lower arm soft collider geom ids
        self.la_soft_ids = np.vectorize(model.geom_name2id)(
            get_micro_bodies(model.geom_names, "la", "G"))
        # lower arm exo geoms ids
        self.la_exo_ids = np.vectorize(model.geom_name2id)(
            get_arm_support_geoms(model.geom_names, "la")
            + get_strap_geoms_names(model.geom_names, "larm_strap_"))
        # number of model timesteps per measurement
        self.n_steps = MEAS_DT / model.opt.timestep
        assert (self.n_steps).is_integer()
        self.n_steps = int(self.n_steps)


class DataIdx(IntEnum):
    # simulated exoskeleton joint pos, vel
    EXO_POS = 0
    EXO_VEL = 1
    # simulated human joint pos, vel
    HUM_POS = 2
    HUM_VEL = 3
    # control outputs
    PID = 4
    CTC = 5
    # cocontraction torque
    CCT = 6


def trajectory(t: np.ndarray, x_0: float, t_0: float = 5.) -> np.ndarray:
    '''
    Return the value of the trajectory at provided time step

    t can be an array (typically an array) of dim (n,)
    t_0 is a time offset
    '''

    # (len_trajectory, joint_id, pos+vel,)
    exo_trajectory = np.zeros((len(t), 4, 2,))

    # set sFE to -90 deg
    exo_trajectory[:, 1, 0] = np.full_like(t, np.deg2rad(-90))

    # compute trajectory of the elbow (sigmoid)
    exp_array = np.exp(t_0 - 2*t)

    exo_trajectory[:, -1, 0] = x_0*(1 - 1 / (1.1*(1+exp_array)))
    exo_trajectory[:, -1, 1] = -2*x_0 * exp_array / (1.1*(1+exp_array)**2)

    # ignore acceleration
    # ddf = 4*x_0*exp_array*(1-exp_array) / (1.1*(1+exp_array)**3)

    return exo_trajectory


def PID_controller(error_array: np.ndarray) -> float:
    '''
    error_array.shape = (4,3) = (n_joints, pos-vel-int error)
    '''
    return np.clip(error_array@PID_COEFF, -MAX_TORQUE, MAX_TORQUE)


def get_eh_contact_force(model, sim, static_data: StaticData) -> float:
    '''
    Returns the norm of the total contact force applied by the exoskeleton on the patient

    static_data contains exo and patient geom ids
    '''
    # force applied by patient on exoskeleton
    force_buffer = np.zeros(3)
    for con_idx in range(sim.data.ncon):
        contact = sim.data.contact[con_idx]

        # con.geom1 < con.geom2
        # and la_soft_ids < la_strap_ids, la_sup_ids
        if contact.geom1 not in static_data.la_soft_ids \
                and contact.geom2 in static_data.la_exo_ids:
            # add force
            force_buffer += get_contact_force(model, sim.data, con_idx)

    return np.linalg.norm(force_buffer)


def init_experiment_controller(sim, static_data: StaticData, vec_getters: dict, max_iter: int = 2500):
    '''
    New controller used to bring arm into initial configuration

    max_iter = 5000 in legacy grid mode
    '''
    # pre alloc exo pos, vel and hum pos
    data_container = np.zeros((max_iter, 4, 3))
    error_array = np.zeros((4, 3))

    # run control until close to desired starting position/velocity
    cancel_flag = True

    for idx in range(max_iter):
        # collect exo pos, vel and hum pos
        data_container[idx, :,
                       DataIdx.EXO_POS] = vec_getters["qpos"](EXO_JOINTS)
        data_container[idx, :,
                       DataIdx.EXO_VEL] = vec_getters["qvel"](EXO_JOINTS)
        data_container[idx, :,
                       DataIdx.HUM_POS] = vec_getters["qpos"](HUM_JOINTS)

        # update integrated error
        error_array[:, :2] = START_CONF - data_container[idx, :, :2]
        error_array[:, 2] += error_array[:, 0]*MEAS_DT

        # have we reached the init config?
        if np.all(np.abs(error_array[:, :2]) < INIT_TOL):
            cancel_flag = False
            break

        # compute and set torque
        pid_tau = PID_controller(error_array)
        sim.data.ctrl[static_data.ea_ids] = pid_tau / MAX_TORQUE

        # step sim
        for _ in range(static_data.n_steps):
            sim.step()

    return data_container, error_array[:, 2], cancel_flag


def pid_controlled_experiment(sim, sim_nominal, ref_traj: np.ndarray, static_data: StaticData, int_error: np.ndarray, vec_getters: dict):
    '''
    ref_trap: tuple of size 3 embedding pos, vel and acc array
    '''
    # preallocate container; see DataIdx for content
    data_container = np.zeros((len(ref_traj), 4, len(DataIdx)))
    error_array = np.zeros((4, 3))
    error_array[:, 2] = int_error

    # flag for unintended high velocities, torques, ...
    flag_velocity = False

    # maximal contact force
    max_con_force = 0.

    for idx, state_config in enumerate(ref_traj):
        # collect pos and vel
        data_container[idx, :, DataIdx.EXO_POS] = vec_getters["qpos"](
            EXO_JOINTS)
        data_container[idx, :, DataIdx.EXO_VEL] = vec_getters["qvel"](
            EXO_JOINTS)

        data_container[idx, :, DataIdx.HUM_POS] = vec_getters["qpos"](
            HUM_JOINTS)
        data_container[idx, :, DataIdx.HUM_VEL] = vec_getters["qvel"](
            HUM_JOINTS)

        # update error, apply and save pid torque
        error_array[:, :2] = state_config - data_container[idx, :, :2]
        error_array[:, 2] += error_array[:, 0]*MEAS_DT

        pid_tau = PID_controller(error_array)
        sim.data.ctrl[static_data.ea_ids] = pid_tau / MAX_TORQUE

        data_container[idx,
                       :, DataIdx.PID] = pid_tau
        data_container[idx,
                       :, DataIdx.CCT] = sim.data.qfrc_actuator[static_data.hj_ids]

        # get contact force
        max_con_force = np.max(
            (max_con_force, get_eh_contact_force(sim.model, sim, static_data)))

        # update nominal model and get feedforward control
        sim_nominal.data.qpos[:] = data_container[idx, :, DataIdx.EXO_POS]
        sim_nominal.data.qvel[:] = data_container[idx, :, DataIdx.EXO_VEL]
        sim_nominal.forward()

        data_container[idx,
                       :, DataIdx.CTC] = sim_nominal.data.qfrc_bias[static_data.ea_ids]

        if data_container[idx, 3, DataIdx.EXO_VEL] > 2.0:
            # elbow too fast => arm slipped out
            # ASK: should we log failing configurations?
            flag_velocity = True
            break

        for _ in range(static_data.n_steps):
            sim.step()

    return data_container, max_con_force, flag_velocity


def spasticity_assessment(sampled_model, sim_nominal, static_data: StaticData):
    '''
    Function to perform the automatic spasticity assessment
    Input: Model (sampled from monte carlo simulation)
    Outout: - Estimated Stiffness (<-- Damping doesn't really make sense since muscle torques are velocity independent)
            - Estimated Stiffness using ground truth torques instead of dynamics computation
            - Flag during assessment (if one of the experiments did not perform well)
            - Flag during initialization (if driving to start position produced issues)
            -
    '''

    # instantiate nominal model
    sim = MjSim(sampled_model)

    vec_getters = {
        "qpos": np.vectorize(sim.data.get_joint_qpos),
        "qvel": np.vectorize(sim.data.get_joint_qvel),
    }

    # initialisation of the contact straps
    for _ in range(STRAP_INIT):
        sim.step()

    # set extensors and flexors to actuation for cocontraction
    sim.data.ctrl[static_data.em_ids] = CC_LEVEL

    # Ensure static position for all DoF not involved in elbow assessment
    # drive system into the desired starting position and velocity
    _, int_error, init_flag = init_experiment_controller(
        sim, static_data, vec_getters)

    if init_flag:
        return FAILED_INIT_OUT

    # Compute desired trajectory (position, velocity and acceleration)
    time_array = np.arange(*EXP_TIME_BOUNDS, MEAS_DT)
    init_elb_pos = sim.data.get_joint_qpos(EXO_JOINTS[3])
    ref_traj = trajectory(time_array, init_elb_pos)

    # carry out pid controlled experiment, based on ref data
    data_container, max_con_force, assessment_flag = pid_controlled_experiment(
        sim, sim_nominal, ref_traj, static_data, int_error, vec_getters)

    if assessment_flag:
        return FAILED_EXP_OUT

    # offset con force
    max_con_force -= MAX_CON_FORCE

    # Compute measured stiffness
    reg_data_coef = LinearRegression().fit(
        data_container[:, 3, DataIdx.HUM_POS].reshape(-1, 1),
        # label = ctc - pid
        data_container[:, 3, DataIdx.CTC] - data_container[:, 3, DataIdx.PID]
    ).coef_.item()

    # Compute gt stiffness
    reg_gt_coef = LinearRegression().fit(
        data_container[:, 3, DataIdx.HUM_POS].reshape(-1, 1),
        data_container[:, 3, DataIdx.CCT]
    ).coef_.item()

    # wrap outputs and flags
    return (reg_data_coef, reg_gt_coef, max_con_force), \
        (assessment_flag, init_flag)


if __name__ == "__main__":
    ...
