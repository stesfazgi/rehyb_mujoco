'''
WIP script evaluating the compressive properties of soft materials
The evaluated XML model is located at: models/soft_tissues/generated_compression_test.xml
Currently, one applies increasing force on soft material and measures the caused displacement
The impact of density is shown
'''

import matplotlib.pyplot as plt
from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
import os
import pickle
from itertools import product

from shared_utils.mujoco import n_step_forward
from xml_generation.utils import wrap_save_xml_element
from xml_generation.arm_collider.utils import create_test_collider
from shared_utils.general import get_project_root, models_folder_path, gen_models_folder_path

import matplotlib
matplotlib.use('TkAgg')

# boolean toggle
RENDER_PRE_COMPRESSION = False
RENDER_COMPRESSION = False
DEBUG = True


def execute_pre_compression(model, sim, compressor_body_index, compressor_joint_index, compressor_act_index,
                            RENDER_PRE_COMPRESSION=False, pre_compression_force=.01,
                            pre_compression_damping=.1, max_delta_position=.00000001):
    ''' the goal of pre compression is to block the soft body
    between the two planes of the compressor '''

    viewer = MjViewer(sim) if RENDER_PRE_COMPRESSION else None

    # set damping and force
    model.dof_damping[compressor_joint_index] = pre_compression_damping
    sim.data.ctrl[compressor_act_index] = pre_compression_force

    # give momentum to the compressor
    n_step_forward(10, sim)

    old_pos = sim.data.body_xpos[compressor_body_index][0]
    while True:
        sim.step()
        if viewer:
            viewer.render()

        pos = sim.data.body_xpos[compressor_body_index][0]
        if(abs(pos - old_pos) < max_delta_position):
            # the compressor is stabilised
            break
        old_pos = pos

    # cancel pre compression damping and force
    sim.data.ctrl[compressor_act_index] = 0
    model.dof_damping[compressor_joint_index] = 0


def execute_compression(sim, compressor_body_index, compressor_act_index,
                        blue_bodies_idx=(np.array([27, 28]),), green_bodies_idx=(np.array([155, 156]),),
                        RENDER_COMPRESSION=False, compression_lower_bound=.05, force_step=10.,
                        force_init=1., stab_phase_length=100):
    '''Applies a progressive compression to the soft body
    Returns a tuple of two arrays (displacement, force)
    Displacement = extend of compression
    force = force applied by the compressor'''

    # the force applied by the compressor follows a pre determined scheme
    viewer = MjViewer(sim) if RENDER_COMPRESSION else None

    softbody_width_array = []

    sim.data.ctrl[compressor_act_index] = force_init

    while sim.data.body_xpos[compressor_body_index][0] > compression_lower_bound:
        # stabilise compression
        n_step_forward(stab_phase_length, sim, viewer)

        # measure equilibrium position and save it
        # to measure the width of the soft body, we will use reference soft bodies
        # that are colored in blue and green in the xml model
        green_mean_x = np.mean(sim.data.body_xpos[green_bodies_idx, 0])
        blue_mean_x = np.mean(sim.data.body_xpos[blue_bodies_idx, 0])

        softbody_width_array.append(np.abs(green_mean_x - blue_mean_x))

        # increase force
        sim.data.ctrl[compressor_act_index] += force_step

    # compute force array
    n = len(softbody_width_array)
    force_array = force_init + force_step*np.arange(0, n)

    # compute soft body displacement array
    softbody_width_array = np.array(softbody_width_array)

    return softbody_width_array, force_array


if __name__ == "__main__":
    # generate composite element
    EXTENDER_GAP = .4
    GEN_COMPRESSION_DIR = os.path.join(
        gen_models_folder_path(), "soft_tissues")

    # body params
    root_name = "test_collider"
    root_pos = [EXTENDER_GAP/2, 0., 0.]

    # composite params
    cmp_type = "ellipsoid"
    count = [5, 5, 10]
    PREFIX = "cmp_"

    # geom params
    geom_type = "sphere"
    size = [.0085]
    rgba = [.8, .2, .1, .2]

    # linear search over mass
    mass = .01
    spacing = .02

    solimp_smooth_def = np.array([.9, .95, .001, .5, 2])
    solref_smooth_def = np.array([-2777, -111])
    solref_smooth_min = np.array([-20, -111])

    solimp_smooth = solimp_smooth_def

    n_samples = 5

    stiffness_range = -500*np.arange(0, n_samples)
    damping_range = 0*np.arange(0, n_samples)
    solref_smooth_samples = solref_smooth_min + \
        np.stack((stiffness_range, damping_range)).T

    stiffness_range = -1500 - 500*np.arange(0, 2)
    damping_range = -10 - 50*np.arange(0, 10)
    solref_smooth_samples = np.array(
        [array for array in product(stiffness_range, damping_range)])
    # solref_smooth_base = np.array([.1, 1])
    # dampratio_range = 0.*np.arange(0, n_samples)
    # time_const_range = -.03*np.arange(0, n_samples)
    # solref_smooth_samples = solref_smooth_base + \
    #     np.stack((time_const_range, dampratio_range)).T

    # arrays saving displacement and force profiles of the experiments
    displacement_arrays = []
    force_arrays = []

    # TO DO: remove create collider from the loop and directly modify
    # the properties of the python - XML element

    for i, solref_smooth in enumerate(solref_smooth_samples):
        if(DEBUG):
            print(f"Generating soft body {i} ...")
        tested_collider = create_test_collider(
            root_name, root_pos, cmp_type, count, spacing, mass, prefix=PREFIX, geom_type=geom_type, size=size, rgba=rgba, solrefsmooth=solref_smooth, solimpsmooth=solimp_smooth)

        # save composite collider alone
        wrap_save_xml_element(tested_collider, "tested_collider.xml",
                              directory_path=GEN_COMPRESSION_DIR)

        if(DEBUG):
            print(f"Loading soft body {i} ...")
        # load the compression test model
        compression_abs_path = os.path.join(
            models_folder_path(), "soft_tissues", "generated_compression_test.xml")
        assert os.path.isfile(compression_abs_path)

        model = load_model_from_path(compression_abs_path)

        sim = MjSim(model)
        try:
            sim.forward()  # ensure that everything is initialized in the sim object
        except:
            print(f"Pre compression failed for params: {solref_smooth}")
            displacement_arrays.append([])
            force_arrays.append([])
            continue

        # extract useful element indexes
        compressor_body_name = "compressor"
        compressor_body_index = model.body_names.index(compressor_body_name)
        compressor_joint_name = "compressor_slide"
        compressor_joint_index = model.joint_name2id(compressor_joint_name)
        compressor_act_index = 0
        soft_body_name = "test_collider"
        soft_body_index = model.body_names.index(soft_body_name)

        if(DEBUG):
            print(f"Starting pre compression of soft body {i} ...")

        # execute pre compression
        try:
            execute_pre_compression(model, sim, compressor_body_index, compressor_joint_index,
                                    compressor_act_index, RENDER_PRE_COMPRESSION)
        except:
            print(f"Pre compression failed for params: {solref_smooth}")
            displacement_arrays.append([])
            force_arrays.append([])
            continue

        if(DEBUG):
            print(f"Starting compression of soft body {i} ...")

        # execute compression
        try:
            soft_body_displacement, force_array = execute_compression(sim, compressor_body_index, compressor_act_index,
                                                                      RENDER_COMPRESSION, force_step=10.)
        except:
            print(f"Compression failed for params: {solref_smooth}")
            displacement_arrays.append([])
            force_arrays.append([])
            continue

        displacement_arrays.append(soft_body_displacement)
        force_arrays.append(force_array)
        if(DEBUG):
            print(f"Soft body {i} Done \n")

    path_to_save_dir = os.path.join(get_project_root(
    ), "bin", "outputs", "compression_test")
    assert os.path.isdir(path_to_save_dir)

    # save force arrays
    with open(os.path.join(path_to_save_dir,
                           "force_array.array"), 'wb') as f:
        pickle.dump(force_arrays, f)

    with open(os.path.join(path_to_save_dir,
                           "disp_array.array"), 'wb') as f:
        pickle.dump(displacement_arrays, f)

    # save solref parameters
    np.save(os.path.join(path_to_save_dir, "solref_params.npy"),
            solref_smooth_samples)

    # plot displacement/force relationship for each experiment
    # the plots are placed according to a nx2 matrix

    # n_plots = len(force_arrays)
    # n_cols = 2
    # n_rows = (n_plots+1) // 2

    # if(n_plots == 1):
    #     fig, axes = plt.subplots(1, 1)
    # else:
    #     fig, axes = plt.subplots(n_rows, n_cols)

    # for i, (disp_array, force_array) in enumerate(zip(displacement_arrays, force_arrays)):
    #     if(n_plots == 1):
    #         axe = axes
    #     elif(n_plots <= n_cols):
    #         axe = axes[i]
    #     else:
    #         axe = axes[i//n_cols, i % n_cols]

    #     axe.scatter(disp_array*1000, force_array)
    #     axe.set_xlabel("Displacement (mm)")
    #     axe.set_ylabel("Load (N)")
    #     axe.set_title(
    #         f"k = {-solref_smooth_samples[i][0]}, b = {-solref_smooth_samples[i][1]}")

    # plt.show()
