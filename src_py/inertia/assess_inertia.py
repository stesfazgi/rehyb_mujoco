import os
import numpy as np

from mujoco_py import load_model_from_path
from shared_utils import gen_models_folder_path
from soft_tissues_eval.utils import get_micro_bodies


def get_bodies_total_mass(body_names, model):
    '''
    body_names is a list of body names of the model

    two assumptions are made:
        - all composite body' names end with "soft_collider"
        - the corresponding coposite bodies have no other body child 
        than their micro elements

    the total mass is returned
    '''
    total_mass = 0

    body_indexes = [model.body_name2id(name) for name in body_names]

    for i, name in enumerate(body_names):
        is_soft_collider = name.endswith('soft_collider')

        if is_soft_collider:
            # the soft collider has no other child than its micro elements
            total_mass += model.body_subtreemass[body_indexes[i]]
        else:
            # other bodies are not composite
            total_mass += model.body_mass[body_indexes[i]]

    return total_mass


def huygens_wo_mass(pos_array):
    '''
    pos_array: numpy array of shape (*, 3)

    returns an array "out" of shape (*, 3, 3) such that:

    for all i, if pos_array[i] = [x, y, z]:
        out[i] =  [[y2+z2,   -xy,   -xz]
                   [  -xy, x2+z2,   -yz]
                   [  -xz,   -yz, x2+y2]]
    '''
    # [x, y, z]
    assert isinstance(pos_array, np.ndarray) and pos_array.ndim == 2 \
        and pos_array.shape[1] == 3

    # [x2, y2, z2]
    square = pos_array*pos_array

    # [y2 + z2, x2 + z2, y2 + x2]
    diag_vect = np.roll(square, -1, axis=1) + np.roll(square, 1, axis=1)

    # [xy, yz, zx]
    mixed_product = pos_array*np.roll(pos_array, -1, axis=1)

    # [xy, xz, yz]
    mixed_product[:, [1, 2]] = mixed_product[:, [2, 1]]

    # [[0, -xy, -xz]
    #  [0,   0, -yz]
    #  [0,   0,   0]]
    upper_triangles = np.zeros((pos_array.shape[0], 3, 3))
    rows, cols = np.triu_indices(3, 1)
    upper_triangles[:, rows, cols] = -mixed_product

    # [[  0, -xy, -xz]
    #  [-xy,   0, -yz]
    #  [-xz, -yz,   0]]
    upper_triangles += np.transpose(upper_triangles, axes=(0, 2, 1))

    # [[y2+z2,   -xy,   -xz]
    #  [  -xy, x2+z2,   -yz]
    #  [  -xz,   -yz, x2+y2]]
    rows = cols = np.arange(3)
    upper_triangles[:, rows, cols] += diag_vect

    return upper_triangles


def huygens_matrix(mass_array, pos_array):
    assert isinstance(mass_array, np.ndarray) and isinstance(
        pos_array, np.ndarray)
    assert mass_array.ndim == 1 and pos_array.ndim == 2
    assert pos_array.shape == (len(mass_array), 3)

    wo_mass_matrix = huygens_wo_mass(pos_array)

    return np.multiply(mass_array.reshape(-1, 1, 1), wo_mass_matrix)


def get_composite_body_inertia(model, center_name, prefix):
    '''
    Warning: an approximation is made here: the rotation of each micro element
    is not taken into account
    '''
    # get id of all micro bodies
    mb_names = get_micro_bodies(model.body_names, prefix)
    mb_ids = np.vectorize(model.body_name2id)(mb_names)

    # get huygens matrix of mbs in central element frame
    mb_pos = model.body_pos[mb_ids]
    mb_mass = model.body_mass[mb_ids]
    mb_huygens_matrix = huygens_matrix(mb_mass, mb_pos)

    # add inertia arrays to huygens matrix
    # NB: all inertias should be the same and diagonal for mbs
    assert np.unique(model.body_inertia[mb_ids], axis=0).shape == (1, 3)
    mb_inertia = model.body_inertia[mb_ids][0].reshape(1, -1)
    mb_huygens_matrix[:, np.arange(3), np.arange(
        3)] += mb_inertia.reshape(1, 3)

    # sum all matrix of inertia
    mb_total_inertia = np.sum(mb_huygens_matrix, axis=0)
    assert mb_total_inertia.shape == (3, 3)

    # finally add inertia of the central element
    center_id = model.body_name2id(center_name)
    central_inertia = model.body_inertia[center_id]
    assert central_inertia.shape == (3,)
    mb_total_inertia += np.diag(central_inertia)

    return mb_total_inertia


def upper_arm_inertia(model):
    '''
    model should be 'nesm_with_patient'
    '''

    # import all necessary names / ids
    humerus_name = "humerus_r"
    humerus_id = model.body_name2id(humerus_name)

    collider_name = "ua_soft_collider"
    collider_id = model.body_name2id(collider_name)
    prefix = "ua"

    # inertia in collider frame
    collider_inertia = get_composite_body_inertia(model, collider_name, prefix)

    # translate to main frame
    collider_mass = model.body_subtreemass[collider_id]
    collider_pos = model.body_pos[collider_id] - model.body_ipos[humerus_id]

    collider_huygens = huygens_matrix(
        np.array([collider_mass]), collider_pos.reshape(1, 3)).squeeze()

    collider_inertia += collider_huygens

    if (model.body_inertia[humerus_id].shape == (3,)):
        collider_inertia += np.diag(model.body_inertia[humerus_id])
    elif (model.body_inertia[humerus_id].shape == (5,)):
        raise NotImplementedError
    else:
        raise ValueError

    return collider_inertia


def lower_arm_inertia(model):
    '''
    model should be 'nesm_with_patient'
    '''

    # import all necessary names / ids
    ulna_name = "ulna_r"
    ulna_id = model.body_name2id(ulna_name)

    radius_name = "radius_r"
    radius_id = model.body_name2id(radius_name)

    collider_name = "la_soft_collider"
    collider_id = model.body_name2id(collider_name)
    prefix = "la"

    # collider inertia (in collider frame)
    collider_inertia = get_composite_body_inertia(model, collider_name, prefix)

    # translate to ulna frame
    collider_mass = model.body_subtreemass[collider_id]
    collider_pos = model.body_pos[collider_id] - model.body_ipos[ulna_id]

    collider_huygens = huygens_matrix(
        np.array([collider_mass]), collider_pos.reshape(1, 3)).squeeze()

    collider_inertia += collider_huygens

    # radius inertia
    radius_inertia = model.body_inertia[radius_id]
    assert radius_inertia.shape == (3,)

    radius_inertia = np.diag(radius_inertia)

    # translate radius inertia to main frame
    radius_pos = model.body_pos[radius_id] + \
        model.body_ipos[radius_id] - model.body_ipos[ulna_id]
    radius_mass = model.body_mass[radius_id]

    radius_huygens = huygens_matrix(
        np.array([radius_mass]), radius_pos.reshape(1, 3)).squeeze()

    radius_inertia += radius_huygens

    # sum all inertias
    ulna_inertia = model.body_inertia[radius_id]
    assert ulna_inertia.shape == (3,)
    ulna_inertia = np.diag(ulna_inertia)

    ulna_inertia += collider_inertia
    ulna_inertia += radius_inertia

    return ulna_inertia


if __name__ == "__main__":
    ''' Test huygens_wo_mass'''
    pos_array = np.arange(1, 7).reshape(2, 3)
    m1 = [[13, -2, -3], [-2, 10, -6], [-3, -6, 5]]
    m2 = [[61, -20, -24], [-20, 52, -30], [-24, -30, 41]]

    res = np.array([m1, m2])
    assert np.array_equal(res, huygens_wo_mass(pos_array))
    print("'huygens_wo_mass' OK")

    ''' Test huygens_matrix '''
    mass1, mass2 = 2, 3
    res[0] *= mass1
    res[1] *= mass2

    masses = np.array([mass1, mass2])

    assert np.array_equal(res, huygens_matrix(masses, pos_array))
    print("'huygens_matrix' OK")

    ''' Measure upper arm and lower arm mass '''
    model_path = os.path.join(gen_models_folder_path(),
                              'exo_with_patient',
                              'nesm_with_patient.xml')

    assert os.path.isfile(model_path)

    model = load_model_from_path(model_path)

    # upper arm mass
    ua_body_names = ["humerus_r", "ua_soft_collider"]
    ua_mass = get_bodies_total_mass(ua_body_names, model)
    print(f"Upper arm mass: {ua_mass}")

    # lower arm mass
    la_body_names = ["ulna_r", "la_soft_collider", "radius_r"]
    la_mass = get_bodies_total_mass(la_body_names, model)
    print(f"Lower arm mass: {la_mass}")
