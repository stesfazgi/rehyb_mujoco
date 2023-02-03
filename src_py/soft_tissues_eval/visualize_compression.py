import os
import matplotlib.pyplot as plt
import tikzplotlib
import pickle
import numpy as np

from mujoco_py import MjSim, load_model_from_path

from shared_utils.general import get_project_root, models_folder_path
from soft_tissues_eval.utils import edit_MjModel_cmp_smoothness
from soft_tissues_eval.soft_compression_eval import execute_pre_compression, execute_compression

import matplotlib
matplotlib.use('TkAgg')


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

#  measure initial width of the composite body
sim = MjSim(model)
sim.forward()
green_mean_x = np.mean(sim.data.body_xpos[green_bodies_idx, 0])
blue_mean_x = np.mean(sim.data.body_xpos[blue_bodies_idx, 0])
cmp_init_width = np.abs(green_mean_x - blue_mean_x)

print(f"Initial width of the composite body: {cmp_init_width}")

# measure length of the composite body
top_micro_elememt = "cmp_B2_2_9"
bottom_micro_element = "cmp_B2_2_0"

top_idx = model.body_name2id(top_micro_elememt)
bottom_idx = model.body_name2id(bottom_micro_element)

top_pos = sim.data.body_xpos[top_idx]
bottom_pos = sim.data.body_xpos[bottom_idx]

cmp_length = np.linalg.norm(top_pos - bottom_pos)

print(f"Length of composite body: {cmp_length}")

# deduce approximative compression surface
S_approx = cmp_length*cmp_init_width

print(f"Compression surface approximation: {S_approx}")

# draw uniform samples of solref stiffness
# solref_stiffs = np.arange(-100, -2150, -250)
solref_stiffs = np.array([-100, -975])

solref_damps = np.full_like(solref_stiffs, -111.)
solrefs = np.column_stack((solref_stiffs, solref_damps))

print(solrefs)

# cache the compression results to dir
path_to_cache = os.path.join(
    get_project_root(), "bin", "outputs", "compression_test")
assert os.path.isdir(path_to_cache)
cache_file_path = os.path.join(path_to_cache, "compression_results.pkl")

# object containing data
compression_data = {}

# if (False):
if (os.path.isfile(cache_file_path)):
    with open(cache_file_path, "rb") as f:
        compression_data = pickle.load(f)
else:
    for i, solref in enumerate(solrefs):
        compression_data[str(solref)] = {}
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
                                                           compressor_act_index, force_step=10., green_bodies_idx=green_bodies_idx, blue_bodies_idx=blue_bodies_idx)

        # save results of the compression
        compression_data[str(solref)]["width"] = soft_body_width
        compression_data[str(solref)]["force"] = force_array

    # cache results
    with open(cache_file_path, "wb") as f:
        pickle.dump(compression_data, f, pickle.HIGHEST_PROTOCOL)


generated_pgfs_dir = os.path.join(get_project_root(), "bin", "pgf_plots")
assert os.path.isdir(generated_pgfs_dir)

# convert width to strain and force to compressive stress
young_mod = {}

for solref in solrefs:
    compression_data[str(solref)]["width"] = (
        cmp_init_width - compression_data[str(solref)]["width"]) / cmp_init_width

    compression_data[str(solref)]["force"] = compression_data[str(
        solref)]["force"] / S_approx

    young_mod[str(solref)] = compression_data[str(solref)
                                              ]["force"][-1]/compression_data[str(solref)]["width"][-1]

    print(f"Young mod {young_mod[str(solref)]}")


# plot properties
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
linestyles = ['-', '--', '-.', ':']

# plot curves
plt.figure(figsize=(14, 8))


low_ym_width = np.linspace(
    0, compression_data[str(solrefs[0])]["width"][-1], 10)
low_ym_force = young_mod[str(
    solrefs[0])]*low_ym_width \
    #  + compression_data[str(solrefs[0])]["force"][0]


plt.plot(compression_data[str(solrefs[0])]["width"],
         compression_data[str(solrefs[0])]["force"]/1000, label="Material B", c=colors[0], linestyle=linestyles[1])
plt.plot(low_ym_width, low_ym_force/1000, label="Young modulus B",
         c=colors[0], linestyle=linestyles[0])


high_ym_width = np.linspace(
    0, compression_data[str(solrefs[1])]["width"][-1], 10)
high_ym_force = young_mod[str(
    solrefs[1])]*high_ym_width \
    #  + compression_data[str(solrefs[1])]["force"][0]

plt.plot(compression_data[str(solrefs[1])]["width"],
         compression_data[str(solrefs[1])]["force"]/1000, label="Material A", c=colors[1], linestyle=linestyles[1])
plt.plot(high_ym_width, high_ym_force/1000, label="Young modulus A",
         c=colors[1], linestyle=linestyles[0])

theo_strain = np.linspace(0, (low_ym_width[-1]+high_ym_width[-1]) / 2, 10)
theo_stress_relaxed = 30 * theo_strain
theo_stress_activated = 66 * theo_strain[:-1]
plt.plot(theo_strain, theo_stress_relaxed, label="Relaxed muscle",
         c=colors[2], linestyle=linestyles[2])
plt.plot(theo_strain[:-1], theo_stress_activated, label="Activated muscle",
         c=colors[2], linestyle=linestyles[3])

plt.plot()

plt.legend()


plt.xlabel("Strain (-)")
plt.ylabel("Compressive stress (kPa)")

# plt.show()


tikzplotlib.save(os.path.join(generated_pgfs_dir, "compression_test.pgf"))
