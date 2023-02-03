'''
A demo of a human arm equipped with the exoskeleton grasping an iron bar
The reference XML model is located at: 'setup_guide/demo_grasping/nesm_with_patient_demo.py'
The actuation levels of muscles and robotic actuators are hardcoded
Start by reading the code of the function 'play_experiment', which is the main function of this script
'''
from mujoco_py import load_model_from_path, MjSim, MjViewer

from shared_utils.mujoco import n_step_forward

# hardcoded actuation sequences
stages_length = [50, 250, 100, 350, 250, 50, 600, 400]
stages_actuators = [["e_pronation", "s_intra", "s_extra", "f_extension", "f_extension2", "w_flexion"],
                    ["biceps"],
                    ["sF/E", "sA/A"],
                    ["biceps", "triceps", "sF/E"],
                    ["f_extension", "f_extension2",
                        "f_flexion", "f_flexion2", "s_extra"],
                    ["biceps"],
                    ["sF/E"],
                    ["w_adduction", "s_intra"]]
stages_act_levels = [[1., .26, .31, .5, .5, .8],
                     [.5],
                     [.12, .05],
                     [.1, 0., .1],
                     [0., 0., .5, .5, .5],
                     [1.],
                     [.7],
                     [.1, .4]]


def set_actuation_levels(model, sim, actuator_names, act_levels):
    '''
    model: PyMjModel object associated to the reference XML model
    sim: MjSim object built on model
    actuators: list of actuator names belonging to model
    actuator_levels: list of actuator levels
    '''
    assert len(actuator_names) == len(
        act_levels), "There should be as many actuator names as actuation levels"

    for act_name, level in zip(actuator_names, act_levels):
        assert act_name in model.actuator_names, "the actuator name should belong to the model"

        # sim.data.ctrl is the array of actuation levels
        # to retrieve the index of each actuator in that array, its name
        #    and the model method actuator_name2id are used
        sim.data.ctrl[model.actuator_name2id(act_name)] = level


def cancel_all_actuations(sim):
    '''
    sim: MjSim object built on the reference model
    '''
    for i in range(len(sim.data.ctrl)):
        sim.data.ctrl[i] = 0.


def play_experiment(path_to_demo="nesm_with_patient_demo.xml", toggle_view=True):
    '''
    path_to_demo: relative path to the XML model defining the exoskeleton
    toggle_view: boolean toggling the visual interface
    '''

    # load the model and create simulation object from the model
    model = load_model_from_path(path_to_demo)
    sim = MjSim(model)

    # initialize the simulation:
    # at step 0, the arm is horizontal and the bar floats in the air
    # one waits 1000 steps to ensure that the arm is vertical and that the bar lies on its support
    # for more clarity, in the project root folder, execute:
    #   $ ./simulate.sh setup_guide/demo_grasping/nesm_with_patient_demo.xml
    init_steps = 1000
    n_step_forward(init_steps, sim)

    # create the visual interface if toggle_view == true
    viewer = MjViewer(sim) if toggle_view else None

    # save the current state of the simulation
    initialised_sim = sim.get_state()
    while True:
        # during the next 100 steps, simply show the scene motionless
        show_steps = 100
        n_step_forward(show_steps, sim, viewer)

        # launch the sequence of actuation stages
        STAGE_NUMBER_ERROR = "There should be one length, one list of actuators and one list of actuation levels per stage"
        assert len(stages_length) == len(stages_actuators) and len(
            stages_length) == len(stages_act_levels), STAGE_NUMBER_ERROR

        for n_steps, actuators, act_levels in zip(stages_length, stages_actuators, stages_act_levels):
            # n_steps is the length (ie the number of steps) of the current stage
            # actuators is the list of actuators whom actuation levels will be modified at the beginning
            #   of the current stage
            # act_levels is the list of actuation levels that will be applied to the previously mentionned
            #   actuators

            # actuation levels are set
            set_actuation_levels(model, sim, actuators, act_levels)

            # simulate n_steps further
            n_step_forward(n_steps, sim, viewer)

        # all stages have been carried out
        # one now sets the simulation back to its initialised state
        sim.set_state(initialised_sim)

        # for some reason, set_state doesn't affect the actuation array
        # therefore, one cancels everything by hand
        cancel_all_actuations(sim)


if __name__ == "__main__":
    play_experiment()
