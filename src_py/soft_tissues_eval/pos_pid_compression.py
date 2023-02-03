from time import time
import matplotlib.pyplot as plt
from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np

from shared_utils.general import models_folder_path

import os

import matplotlib
matplotlib.use('TkAgg')

TOGGLE_VIEW = True

# load the compression test model
compression_abs_path = os.path.join(
    models_folder_path(), "soft_tissues", "generated_compression_test.xml")
assert os.path.isfile(compression_abs_path)

model = load_model_from_path(compression_abs_path)
sim = MjSim(model)
viewer = MjViewer(sim) if TOGGLE_VIEW else None

sim.step()  # ensure that everything is initialized in the sim object

compressor_act_index = 0

# define the reference trajectory
compressor_name = "compressor"
compressor_index = model.body_names.index(compressor_name)
compressor_init_x = sim.data.body_xpos[compressor_index][0]

desired_velocity = .01  # m.s-1
timestep = model.opt.timestep
trajectory_step = desired_velocity*timestep

wished_range = .4
n_steps = int(wished_range/trajectory_step)

# along the x axis
# check the bounds of the arange
ref_x_trajectory = compressor_init_x - np.arange(0, n_steps) * trajectory_step

# define pid coefficients
Kp, Ki, Kd = 1, 1, 1
old_error = 0
I = 0

# DEBUG the initialisation

for ref_x_pos in ref_x_trajectory:
    compressor_x_pos = sim.data.body_xpos[compressor_index][0]
    error = ref_x_pos - compressor_x_pos

    I += error*timestep
    D = (error - old_error)/timestep

    # check the sign
    sim.data.ctrl[compressor_act_index] = -(Kp*error + Ki*I + Kd*D)

    sim.step()
    viewer.render()

    old_error = error
