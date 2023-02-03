from mujoco_py import load_model_from_path, MjSim, MjViewer
from numpy import log10
from shared_utils.mujoco import n_step_forward

from shared_utils.general import round_to_n, models_folder_path

import os
import re

DEBUG = True


def get_glitch_model():
    glitch_abs_path = os.path.join(
        models_folder_path(), "soft_tissues", "generated_glitch_test.xml")
    assert os.path.isfile(glitch_abs_path)
    return load_model_from_path(glitch_abs_path)


def tunneling_happened(collider_zcoords, support_zcoord):
    return collider_zcoords[1] < support_zcoord


def small_glitch_happened(collider_zcoords):
    return abs(
        collider_zcoords[2]-collider_zcoords[1]) > abs(collider_zcoords[3]-collider_zcoords[2])


def is_support_geom(geom_name, support_prefix):
    pattern = "^"+support_prefix+"_[0-9]+$"
    return re.match(pattern, geom_name) is not None


def filter_support_geoms(geoms, support_prefix):
    return [geom for geom in geoms if is_support_geom(geom, support_prefix)]


def get_support_center_index(support_prefix, model, n_box=None):
    # retrieving support pos
    # uses an heuristic only valid if n_box > 5

    if n_box is None:
        n_box = len(filter_support_geoms(model.geom_names, support_prefix))

    central_geom_name = support_prefix+f"_{n_box//2}"
    return model.geom_names.index(central_geom_name)


def glitch_limit_dichotomy(collider_name, support_prefix, lower_bound=.001, upper_bound=10., tol=.001, n_box=None):
    # load model
    model = get_glitch_model()

    # initialize simulation
    sim = MjSim(model)
    sim.step()

    # retrieving support pos
    # uses an heuristic only valid if n_box > 5
    support_center_index = get_support_center_index(
        support_prefix, model, n_box)
    support_zcoord = sim.data.geom_xpos[support_center_index][2]

    # preparing retrieval of collider pos
    collider_index = model.body_names.index(collider_name)

    a, b = lower_bound, upper_bound

    while(abs(a-b) > tol):
        # initialize forces
        impact_force = (a+b)/2
        pullup_force = -impact_force/100
        collider_zcoords = []

        if(DEBUG):
            print(round_to_n(a, n=int(abs(log10(tol))+1)),
                  round_to_n(b, n=int(abs(log10(tol))+1)),
                  round_to_n(impact_force))

        # reset model to initial state
        sim.reset()
        sim.step()

        # first stage: still
        n_step_forward(50, sim)
        collider_zcoords.append(sim.data.body_xpos[collider_index][2])

        # second stage: impact
        sim.data.ctrl[0] = impact_force
        n_step_forward(1000, sim)
        collider_zcoords.append(sim.data.body_xpos[collider_index][2])

        if tunneling_happened(collider_zcoords, support_zcoord):
            if DEBUG:
                print(f"Tunneling\n")
            b = impact_force
            continue

        # third stage: release pressure
        sim.data.ctrl[0] = 0
        n_step_forward(500, sim)
        collider_zcoords.append(sim.data.body_xpos[collider_index][2])

        # fourth stage: pull up
        sim.data.ctrl[0] = pullup_force
        n_step_forward(500, sim)
        collider_zcoords.append(sim.data.body_xpos[collider_index][2])

        if small_glitch_happened(collider_zcoords):
            if DEBUG:
                print(f"Small glitch\n")
            b = impact_force
        else:
            if DEBUG:
                print(f"Everything fine\n")
            a = impact_force

    return (a+b)/2


def play_glitch_experiment(impact_force):
    # load model
    model = get_glitch_model()

    # initialize simulation
    sim = MjSim(model)
    sim.step()

    # initialize viewer
    viewer = MjViewer(sim)

    pullup_force = -impact_force/100

    # first stage: still
    n_step_forward(50, sim, viewer)

    # second stage: impact
    sim.data.ctrl[0] = impact_force
    n_step_forward(1000, sim, viewer)

    # third stage: release pressure
    sim.data.ctrl[0] = 0
    n_step_forward(500, sim, viewer)

    # fourth stage: pull up
    sim.data.ctrl[0] = pullup_force
    n_step_forward(500, sim, viewer)


if __name__ == "__main__":
    limit_force = glitch_limit_dichotomy("test_collider", "ua", tol=.001)
    print(f"Limit force: {limit_force}")

    play_glitch_experiment(limit_force)
    play_glitch_experiment(limit_force*1.02)
