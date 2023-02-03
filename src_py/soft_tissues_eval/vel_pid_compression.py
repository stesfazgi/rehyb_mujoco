from mujoco_py import load_model_from_path, MjSim, MjViewer
from shared_utils.mujoco import n_step_forward
from shared_utils.general import models_folder_path
import matplotlib.pyplot as plt

import os

import matplotlib
matplotlib.use('TkAgg')

# boolean toggle
RENDER_PRE_COMPRESSION = True
RENDER_COMPRESSION = True
DEBUG = True

# load the compression test model
compression_abs_path = os.path.join(
    models_folder_path(), "soft_tissues", "generated_compression_test.xml")
assert os.path.isfile(compression_abs_path)

model = load_model_from_path(compression_abs_path)

sim = MjSim(model)
sim.step()  # ensure that everything is initialized in the sim object

viewer = MjViewer(sim) if (
    RENDER_PRE_COMPRESSION or RENDER_COMPRESSION) else None

# extract useful element indexes
compressor_body_name = "compressor"
compressor_body_index = model.body_names.index(compressor_body_name)
compressor_joint_name = "compressor_slide"
compressor_joint_index = model.joint_name2id(compressor_joint_name)
compressor_act_index = 0

# extract other useful model data
timestep = model.opt.timestep

'''PRE COMPRESSION'''
# the goal of pre compression is to block the soft body
# between the two planes of the compressor

# pre compression parameters
pre_compression_force = .01
pre_compression_damping = .1
max_delta_position = .00000001

# set damping and force
model.dof_damping[compressor_joint_index] = pre_compression_damping
sim.data.ctrl[compressor_act_index] = pre_compression_force

# increase the speed of the compressor
n_step_forward(10, sim)

old_pos = sim.data.body_xpos[compressor_body_index][0]
while True:
    sim.step()
    if RENDER_PRE_COMPRESSION:
        viewer.render()

    pos = sim.data.body_xpos[compressor_body_index][0]
    if(abs(pos - old_pos) < max_delta_position):
        break
    old_pos = pos

# cancel pre compression force and damping
sim.data.ctrl[compressor_act_index] = 0
model.dof_damping[model.joint_name2id("compressor_slide")] = 0

if(DEBUG):
    print("End of pre compression")

'''PID CONTROLLED COMPRESSION'''
# define pid coefficients
Kp, Ki, Kd = 10, 10, .001
old_error = 0
I = 0
compression_upper_bound = .05

vel_history = []
desired_velocity = -.001  # m.s-1 and < 0 because compression

# DEBUG the initialisation
while sim.data.body_xpos[compressor_body_index][0] > compression_upper_bound:
    # compute the error
    compressor_x_vel = sim.data.body_xvelp[compressor_body_index][0]
    error = desired_velocity - compressor_x_vel

    # deduce the derivative / integral
    I += error*timestep
    D = (error - old_error)/timestep

    # apply force (- because opposite frame)
    sim.data.ctrl[compressor_act_index] = -(Kp*error + Ki*I + Kd*D)

    # one step further
    sim.step()

    if RENDER_COMPRESSION:
        viewer.render()

    # update old_error
    old_error = error
    vel_history.append(compressor_x_vel)


# plt.scatter(time, vel_history)

# plt.xlabel("Time (s)")
# plt.ylabel("Velocity (m.s-1)")

# plt.show()
