'''
Several functions used to compute the inertia matrix and the mass of composite elements
'''

import numpy as np
from utils import get_micro_bodies


def compute_point_inertia(m, pos):
    '''
    m: mass
    pos: position in a given frame R
    returns the matrix of inertia of the mass point (m, pos) in frame R
    '''

    assert m > 0, "m must be stricly positive"
    assert isinstance(pos, np.ndarray) and pos.shape == (
        3,), "pos must be a numpy array of shape (3,)"

    # compute the diagonal of the inertia matrix
    # [x2, y2, z2]
    square = pos*pos
    # [y2 + z2, x2 + z2, y2 + x2]
    diag_vect = np.roll(square, -1) + np.roll(square, 1)

    # compute the upper triangle of the inertia matrix
    mixed_product = pos*np.roll(pos, -1)  # [xy, yz, zx]
    mixed_product[[1, 2]] = mixed_product[[2, 1]]  # [xy, xz, yz]

    upper_triangle = np.zeros((3, 3))
    upper_triangle[np.triu_indices(3, 1)] = -mixed_product

    # lower_triangle = upper_triangle.T (symmetry of inertia matrix)
    return m*(np.diag(diag_vect) + upper_triangle + upper_triangle.T)


def compute_points_inertia(masses, positions):
    '''
    returns the matrix of inertia of the set of mass points defined by
    masses, positions
    '''

    assert len(masses) == len(
        positions), "Masses and positions must have same length"

    I = np.zeros((3, 3))
    for m, pos in zip(masses, positions):
        I += compute_point_inertia(m, pos)

    return I


def compute_composite_inertia(model, sim, prefix, central_element_name):
    '''
    model, sim: given MjModel and MjSim
    prefix: prefix of the targeted composite element (TCE)
    central_element_name: name of the central element of the TCE

    returns the matrix of inertia of the TCE
    '''
    # get list of micro bodies names and ids
    micro_bodies = get_micro_bodies(model.body_names, prefix)
    micro_bodies_ids = [model.body_name2id(
        body_name) for body_name in micro_bodies]

    # extract absolute pos and mass
    micro_bodies_masses = model.body_mass[micro_bodies_ids]
    micro_bodies_pos = np.array(
        [sim.data.get_body_xpos(name) for name in micro_bodies])

    # center pos on the central element
    micro_bodies_pos -= sim.data.get_body_xpos(central_element_name)

    return compute_points_inertia(micro_bodies_masses, micro_bodies_pos)


def compute_composite_mass(model, sim, prefix, central_element_name):
    '''
    model, sim: given MjModel and MjSim
    prefix: prefix of the targeted composite element (TCE)
    central_element_name: name of the central element of the TCE

    returns the mass of the TCE
    '''
    # get list of micro bodies names and ids
    micro_bodies = get_micro_bodies(model.body_names, prefix)
    micro_bodies_ids = [model.body_name2id(
        body_name) for body_name in micro_bodies]

    # get central element id
    central_element_id = model.body_name2id(central_element_name)

    # extract mass of micro elements
    micro_bodies_masses = model.body_mass[micro_bodies_ids]

    return np.sum(micro_bodies_masses) + model.body_mass[central_element_id]


if __name__ == "__main__":
    # BEGIN: test compute_point_inertia
    # test 1:
    r1 = np.array([1., 0., 0.])
    m1 = 1.
    sol1 = np.array([[0., 0., 0.],
                     [0., 1., 0.],
                     [0., 0., 1.]])

    assert np.array_equal(compute_point_inertia(
        m1, r1), sol1), "compute_point_inertia: Test 1 failed"

    # test 2:
    r2 = np.array([1., 0., 0.])
    m2 = 2.
    sol2 = np.array([[0., 0., 0.],
                     [0., 2., 0.],
                     [0., 0., 2.]])

    assert np.array_equal(compute_point_inertia(
        m2, r2), sol2), "compute_point_inertia: Test 2 failed"

    # test 3
    r3 = np.array([1., 2., 3.])
    m3 = 1.
    sol3 = np.array([[13., -2., -3.],
                     [-2., 10., -6.],
                     [-3., -6., 5.]])

    assert np.array_equal(compute_point_inertia(
        m3, r3), sol3), "compute_point_inertia: Test 3 failed"

    print("compute_point_inertia OK")
    # END: test compute_point_inertia

    # BEGIN: test compute_points_inertia
    positions = np.identity(3)
    masses = np.ones(3)
    sol = np.array([[4., 0., 0.],
                    [0., 4., 0.],
                    [0., 0., 4.]])
    assert np.array_equal(compute_point_inertia(
        m3, r3), sol3), "compute_points_inertia: Test 1 failed"
    print("compute_points_inertia OK")
    # END: test compute_points_inertia
