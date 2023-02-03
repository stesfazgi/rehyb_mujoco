'''
Evaluates simulation using data from 20210208 experiments

So far, only static tests are implemented.
Warning: not same angular convention in the experimental data
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tikzplotlib

from shared_utils import get_project_root
from shared_utils.general import models_folder_path
from sklearn.metrics import mean_squared_error
from exo_eval.read_test_data_from_csv import read_test_data_from_csv
from mujoco_py import load_model_from_path, MjSim, MjViewer

matplotlib.use('TkAgg')


if __name__ == "__main__":
    TOGGLE_VIEW = False
    SAVE_PGF = True

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    ''' Load experimental data '''
    # choose experience
    for test_number in range(1, 7):
        test_name = "20210208_EFE_StaticTorque"

        assert (test_name == "20210208_EFE_StaticTorque" and 1 <= test_number <= 6) \
            or (test_name == "20210208_EFE_PositionRamp" and 1 <= test_number <= 2)

        time, joint_theta, _, motor_torques, _, _, _ = read_test_data_from_csv(
            test_name, test_number)

        # convert sFE to model convention
        # joint_theta[1, :] = -joint_theta[1, :] - 90.
        # motor_torques[1, :] = -motor_torques[1, :]

        ''' Load Mujoco model '''
        model_path = os.path.join(
            models_folder_path(), "exoskeleton", "exoskeleton_with_weight.xml")
        assert os.path.isfile(model_path)
        model = load_model_from_path(model_path)

        # lock joints; the exact setting depends on the simulation
        d_theta = 1e-6
        if((test_name == "20210208_EFE_StaticTorque" and test_number <= 4)
                or (test_name == "20210208_EFE_PositionRamp" and test_number == 1)):
            # sAA, sFE, sIE are locked
            lock_angle = np.mean(np.deg2rad(joint_theta[:-1, :]), axis=1)
            assert lock_angle.shape == (3,)

            model.jnt_range[:-
                            1] = np.vstack((lock_angle-d_theta, lock_angle+d_theta)).T

        else:
            # sAA, sIE are locked
            mask = [True, False, True, False]
            lock_angle = np.mean(np.deg2rad(joint_theta[mask, :]), axis=1)
            assert lock_angle.shape == (2,)

            model.jnt_range[mask] = np.vstack(
                (lock_angle-d_theta, lock_angle+d_theta)).T

        # set friction
        # with stiffness
        # model.dof_frictionloss[-1] = 3.0785319421283606
        # without stiffness
        model.dof_frictionloss[-1] = 2.7868500893593593

        # initialize sim
        sim = MjSim(model)
        viewer = MjViewer(sim) if TOGGLE_VIEW else None
        sim.forward()

        # count number of mujoco steps per torque update
        dt = 0.01  # torque update every hundredth of second
        n_steps = dt / sim.model.opt.timestep
        assert (n_steps).is_integer()
        n_steps = int(n_steps)

        # initialize joint position using ground truth (in rad)
        _ = np.vectorize(sim.data.set_joint_qpos)(
            model.joint_names, np.deg2rad(joint_theta)[:, 0])
        sim.forward()

        # initialize pos save
        simulated_joint_pos = np.zeros((0, joint_theta.shape[0]))

        # the scale of the elbow actuator has been optimized
        # ctrl_scale = np.ones(4)
        # with stiffness
        # ctrl_scale = np.array([1., 1., 1., 0.8551712011495531])
        # without stiffness
        ctrl_scale = np.array([1., 1., 1., 0.7234048749654893])

        for j in range(len(time)):
            # set ctrl (scaled by gear)
            sim.data.ctrl[:] = motor_torques[:, j] / model.actuator_gear[:, 0] \
                * ctrl_scale
            # save joint pos
            simulated_joint_pos = np.vstack((simulated_joint_pos,
                                            np.vectorize(sim.data.get_joint_qpos)(model.joint_names)))

            for i in range(n_steps):
                sim.step()
                if viewer is not None:
                    viewer.render()

        # convert simulation data to deg
        simulated_joint_pos = np.rad2deg(simulated_joint_pos)

        print(mean_squared_error(simulated_joint_pos[:, -1],
                                 joint_theta[-1, :]))

        # plot results
        if(SAVE_PGF):

            generated_pgfs_dir = os.path.join(
                get_project_root(), "bin", "pgf_plots")
            assert os.path.isdir(generated_pgfs_dir)

            # fig, axes = plt.subplots(2, 2, figsize=(15, 15))
            plt.plot(time, joint_theta[-1],
                     label="Ground Truth", color=colors[0])
            plt.plot(
                time, simulated_joint_pos[:, -1], label="Simulation", color=colors[1])

            plt.ylim([-5, 130])

            plt.xlabel("Time (s)")
            plt.ylabel("Joint position (deg)")
            plt.title(f"eFE static test {test_number}")
            plt.legend()

            # set same scale to all plots, not necessarily in the same limits
            tikzplotlib.save(os.path.join(
                generated_pgfs_dir, f"eFE_StaticTorque{test_number}.pgf"))
        else:
            # just plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 15))

            # size of dots
            s = np.full_like(simulated_joint_pos[:, 0], 5)

            for i, ax in enumerate(axes.reshape(-1)):
                # plot reference
                ax.scatter(time, joint_theta[i], label="Ground Truth", s=s)

                # plot simulation
                ax.scatter(
                    time, simulated_joint_pos[:, i], label="Simulation", s=s)

                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Joint position (deg)")
                ax.set_title(model.actuator_names[i])
                ax.legend()

            # set same scale to all plots, not necessarily in the same limits
            ax_with_lim = [(ax, ax.get_ylim()) for ax in axes.reshape(-1)]
            max_y_delta = np.max(
                [lmax-lmin for _, (lmin, lmax) in ax_with_lim])

            for ax, (lmin, lmax) in ax_with_lim:
                ax_delta = (max_y_delta - (lmax-lmin))/2
                ax.set_ylim(lmin-ax_delta, lmax+ax_delta)

            plt.subplots_adjust(wspace=.2, hspace=.3)
            plt.show()
