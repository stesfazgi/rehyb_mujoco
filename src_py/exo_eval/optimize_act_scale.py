import os
import numpy as np

from shared_utils.general import models_folder_path
from exo_eval.read_test_data_from_csv import read_test_data_from_csv
from mujoco_py import load_model_from_path, MjSim
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error


EXP_TORQUE_DATA = {}
EXP_ANGLE_DATA = {}
EXP_TIME_DATA = {}
MODELS = {}

N_STEPS = None


def elbow_act_score(elbow_act_scale):
    score = 0.

    for test_number in range(1, 7):
        sim = MjSim(MODELS[test_number])
        sim.forward()

        # initialize joint position using ground truth (in rad)
        _ = np.vectorize(sim.data.set_joint_qpos)(
            MODELS[test_number].joint_names, EXP_ANGLE_DATA[test_number][:, 0])

        # initialize pos save
        simulated_joint_pos = np.zeros(
            (0, EXP_ANGLE_DATA[test_number].shape[0]))

        ctrl_scales = np.hstack((np.ones(3), elbow_act_scale))
        assert ctrl_scales.shape == (4,)

        for j in range(len(EXP_TIME_DATA[test_number])):
            # set ctrl (scaled by gear)
            sim.data.ctrl[:] = EXP_TORQUE_DATA[test_number][:, j] / \
                MODELS[test_number].actuator_gear[:, 0] * ctrl_scales
            # save joint pos
            simulated_joint_pos = np.vstack((simulated_joint_pos,
                                            np.vectorize(sim.data.get_joint_qpos)(MODELS[test_number].joint_names)))

            for _ in range(N_STEPS):
                sim.step()

        score += mean_squared_error(simulated_joint_pos[:, -1],
                                    EXP_ANGLE_DATA[test_number][-1, :])

    return score


def elbow_friction_score(elbow_friction_loss):
    score = 0.

    for test_number in range(1, 7):
        MODELS[test_number].dof_frictionloss[-1] = elbow_friction_loss

        sim = MjSim(MODELS[test_number])
        sim.forward()

        # initialize joint position using ground truth (in rad)
        _ = np.vectorize(sim.data.set_joint_qpos)(
            MODELS[test_number].joint_names, EXP_ANGLE_DATA[test_number][:, 0])

        # initialize pos save
        simulated_joint_pos = np.zeros(
            (0, EXP_ANGLE_DATA[test_number].shape[0]))

        for j in range(len(EXP_TIME_DATA[test_number])):
            # set ctrl (scaled by gear)
            sim.data.ctrl[:] = EXP_TORQUE_DATA[test_number][:, j] / \
                MODELS[test_number].actuator_gear[:, 0]
            # save joint pos
            simulated_joint_pos = np.vstack((simulated_joint_pos,
                                            np.vectorize(sim.data.get_joint_qpos)(MODELS[test_number].joint_names)))

            for _ in range(N_STEPS):
                sim.step()

        score += mean_squared_error(simulated_joint_pos[:, -1],
                                    EXP_ANGLE_DATA[test_number][-1, :])

    return score


def crossed_score(params):
    '''
    params[0]: actuator scale
    params[1]: friction loss
    '''
    score = 0.

    for test_number in range(1, 7):
        MODELS[test_number].dof_frictionloss[-1] = params[1]

        sim = MjSim(MODELS[test_number])
        sim.forward()

        # initialize joint position using ground truth (in rad)
        _ = np.vectorize(sim.data.set_joint_qpos)(
            MODELS[test_number].joint_names, EXP_ANGLE_DATA[test_number][:, 0])

        # initialize pos save
        simulated_joint_pos = np.zeros(
            (0, EXP_ANGLE_DATA[test_number].shape[0]))

        ctrl_scales = np.hstack((np.ones(3), params[0]))
        assert ctrl_scales.shape == (4,)

        for j in range(len(EXP_TIME_DATA[test_number])):
            # set ctrl (scaled by gear)
            sim.data.ctrl[:] = EXP_TORQUE_DATA[test_number][:, j] / \
                MODELS[test_number].actuator_gear[:, 0] * ctrl_scales
            # save joint pos
            simulated_joint_pos = np.vstack((simulated_joint_pos,
                                            np.vectorize(sim.data.get_joint_qpos)(MODELS[test_number].joint_names)))

            for _ in range(N_STEPS):
                sim.step()

        score += mean_squared_error(simulated_joint_pos[:, -1],
                                    EXP_ANGLE_DATA[test_number][-1, :])

    return score


N_min_steps = 1


def minimize_callback(xk):
    global N_min_steps

    printed_string = f"{N_min_steps:<10}"

    for x in xk:
        printed_string += f"{x:<20}"

    print(printed_string)
    N_min_steps += 1


if __name__ == "__main__":
    # choose experience and model
    test_name = "20210208_EFE_StaticTorque"

    model_path = os.path.join(
        models_folder_path(), "exoskeleton", "exoskeleton_with_weight.xml")
    assert os.path.isfile(model_path)

    # choose the correct tolerance
    d_theta = 1e-6

    print("Caching models and experimental data...")
    # experimental data and mujoco models are cached
    for test_number in range(1, 7):
        EXP_TIME_DATA[test_number], EXP_ANGLE_DATA[test_number], _, EXP_TORQUE_DATA[test_number], _, _, _ = read_test_data_from_csv(
            test_name, test_number)
        # adapt convention for shoulder flexion extension
        # EXP_ANGLE_DATA[test_number][1, :] = - \
        #     EXP_ANGLE_DATA[test_number][1, :] - 90.
        # EXP_TORQUE_DATA[test_number][1, :] = - \
        #     EXP_TORQUE_DATA[test_number][1, :]
        # convert deg to rad
        EXP_ANGLE_DATA[test_number] = np.deg2rad(EXP_ANGLE_DATA[test_number])

        MODELS[test_number] = load_model_from_path(model_path)

        if N_STEPS is None:
            N_STEPS = 0.01 / MODELS[test_number].opt.timestep
            assert (N_STEPS).is_integer()
            N_STEPS = int(N_STEPS)

        if test_number >= 5:
            mask = [True, False, True, False]
            lock_angle = np.mean(np.deg2rad(
                EXP_ANGLE_DATA[test_number][mask, :]), axis=1)
            assert lock_angle.shape == (2,)

            MODELS[test_number].jnt_range[mask] = np.vstack(
                (lock_angle-d_theta, lock_angle+d_theta)).T

        else:
            lock_angle = np.mean(np.deg2rad(
                EXP_ANGLE_DATA[test_number][:-1, :]), axis=1)
            assert lock_angle.shape == (3,)

            MODELS[test_number].jnt_range[:-
                                          1] = np.vstack((lock_angle-d_theta, lock_angle+d_theta)).T

    print("Optimizing elbow scale...")
    elbow_scale_0 = np.ones(1)

    # optimized parameter tolerance
    tol = 1e-6

    OPT_CROSS = True
    OPT_FRICTION = True

    if(OPT_CROSS):
        opt_func = crossed_score
        bounds = [(0, np.inf), (0, np.inf)]
        init_val = np.array([1., 2.])
        print(f"{'Step':<10}{'Scale':<20}{'Friction':<20}")

    elif(OPT_FRICTION):
        opt_func = elbow_friction_score
        bounds = [(0, np.inf)]
        init_val = np.ones(1)

        print(f"{'Step':<10}{'Elbow friction':<20}")
    else:
        opt_func = elbow_act_score
        bounds = [None]
        init_val = np.ones(1)

        print(f"{'Step':<10}{'Scale':<20}")

    minimize(opt_func, init_val,
             method="Nelder-Mead",
             callback=minimize_callback,
             bounds=bounds, tol=tol)
