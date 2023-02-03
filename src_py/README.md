## Python modules of this project

**WARNING: The package hierarchy is outdated**

The content of the `src_py` directory is presented below:

    ├── biceps_spring_model
    ├── elbow_muscles_eval
    ├── play_back
    ├── shared_utils
    ├── soft_tissues_eval
    └── xml_generation
        ├── arm_collider
        ├── arm_inertia
        ├── arm_support
        └── model_cleaning

Let's describe each of those items concisely:

- `biceps_spring_model`: Directory containing Python scripts evaluating a basic spring model of the biceps.
- `elbow_muscles_eval`: Directory containing Python scripts evaluating the muscular model implemented in `nesm_with_patient.xml`.
- `play_back`: Directory containing basic Python scripts allowing real-time visualization of simulated experiments.
- `shared_utils`: Python module containing general purpose utils.
- `soft_tissues_eval`: Directory containing Python scripts evaluating the properties of composite elements.
- `xml_generation`: Python module dedicated to the generation of xml files. It contains four submodules: `arm_collider` (in charge of generating composite elements modelling soft tissues), `arm_inertia` (in charge of generating inertial elements of the human arm), `arm_support` (in charge of generating geom elements of NESM's arm supports) and `model_cleaning` (gathering several tools easing the edition of xml files).

## Few tips to get by with mujoco-py

### A few demos to begin with

- `setup_guide/demo_grasping/demo_script.py` relies on common mujoco-py concepts. It may be a good starting point to try to understand it.
- Other examples can be found on [mujoco-py GitHub](https://github.com/openai/mujoco-py/tree/master/examples). [tosser.py](https://github.com/openai/mujoco-py/blob/master/examples/tosser.py) is the most basic one, but way more advanced scripts are also provided!

### How to use mujoco-py documentation?

Here is the link towards [mujoco-py documentation](https://openai.github.io/mujoco-py/build/html/index.html). You may notice that it is relatively shallow. Therefore, one has to combine it with [MuJoCo's default API documentation](https://mujoco.readthedocs.io/en/latest/APIreference.html).

Let's imagine that you want to find how to set the control of an actuator:

1. Use `Ctrl+F` on the [API documentation](https://mujoco.readthedocs.io/en/latest/APIreference.html) and look for the keyword `control`.
2. After a few mismatches, you should find an interesting field of the `_mjData` class called `ctrl`.
3. You can know look for `ctrl` into [mujoco-py documentation](https://openai.github.io/mujoco-py/build/html/index.html).

Most of the time, keywords used in mujoco-py are the same as those used in the original MuJoCo API.
