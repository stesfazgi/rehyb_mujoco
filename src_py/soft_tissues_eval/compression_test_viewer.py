import numpy as np
import os

from mujoco_py import load_model_from_path, MjSim

from shared_utils.general import models_folder_path
from soft_tissues_eval.utils import edit_MjModel_cmp_smoothness
from soft_tissues_eval.soft_compression_eval import execute_pre_compression, execute_compression


if __name__ == "__main__":
    TOGGLE_VIEW = True

    model_path = os.path.join(models_folder_path(
    ), "soft_tissues", "compression_test_with_solref.xml")
    assert os.path.isfile(model_path)

    model = load_model_from_path(model_path)

    # prefix of the evaluated composite body
    cmp_prefix = "cmp_"

    # get ref micro element for composite body width
    # they are colored in green and blue
    GREEN_RGBA = np.array([0.2, 0.8, 0.1, 1], dtype=np.float32)
    BLUE_RGBA = np.array([0.1, 0.2, 0.8, 1.], dtype=np.float32)

    green_bodies_idx = np.nonzero((model.geom_rgba == GREEN_RGBA).all(axis=1))
    blue_bodies_idx = np.nonzero((model.geom_rgba == BLUE_RGBA).all(axis=1))

    assert len(green_bodies_idx[0]) == len(blue_bodies_idx[0]) == 2

    # draw uniform samples of solref stiffness
    # solref_stiffs = np.arange(-100, -2150, -250)
    solref_stiffs = np.array([-100])

    solref_damps = np.full_like(solref_stiffs, -111.)
    solrefs = np.column_stack((solref_stiffs, solref_damps))

    print(solrefs)

    for i, solref in enumerate(solrefs):
        # set solref
        edit_MjModel_cmp_smoothness(model, cmp_prefix, solref)

        # extract useful element indexes
        compressor_body_name = "compressor"
        compressor_body_index = model.body_names.index(compressor_body_name)
        compressor_joint_name = "compressor_slide"
        compressor_joint_index = model.joint_name2id(compressor_joint_name)
        compressor_act_index = 0
        soft_body_name = "test_collider"
        soft_body_index = model.body_names.index(soft_body_name)

        # init simulation
        sim = MjSim(model)
        sim.forward()  # ensure that everything is initialized in the sim object

        print(f"Starting pre compression of soft body {i} ...")

        # execute pre compression
        execute_pre_compression(model, sim, compressor_body_index, compressor_joint_index,
                                compressor_act_index)

        print(f"Starting compression of soft body {i} ...")

        # execute compression
        soft_body_width, force_array = execute_compression(sim, compressor_body_index,
                                                           compressor_act_index, force_step=10.,
                                                           green_bodies_idx=green_bodies_idx,
                                                           blue_bodies_idx=blue_bodies_idx,
                                                           RENDER_COMPRESSION=TOGGLE_VIEW)
