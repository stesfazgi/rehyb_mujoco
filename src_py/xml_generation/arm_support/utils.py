import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from xml.etree import ElementTree as ET

from shared_utils.general import round_to_3
from shared_utils.plot import plotFrame

'''
For further details on geometric notations, see "arm_support_schematic.pdf"
'''


def computeC(O, A):
    '''
    O, A: np.array, shape = (2,)

    Computes and returns C position using O and A position in a 2D frame centered on C, 
    with X_axis pointing toward the right, Y_axis toward the top
    '''
    # put O to the origin via translation
    At = A-O

    # I is the middle of OA
    It = At/2

    # med_vect = (xv, yx) is a unitary orientation vector of OA's perpendicular bissector
    xv = 1
    yv = -xv*At[0]/At[1]
    med_vect = np.array([xv, yv])
    med_vect = med_vect / np.linalg.norm(med_vect)

    # one can then deduce the position of C, center of the inscribed circle of OAB
    lbda = -It[0]/med_vect[0]
    Ct = np.array([0, It[1] + lbda*med_vect[1]])

    return Ct + O


def compute_OABC(l, L):
    '''
    l, L: float

    Computes and returns O, A, B, C positions in a 2D frame centered on C, 
    with X_axis pointing toward the right, Y_axis toward the top 
    '''
    # carry out all calcutions in xyz cannonical frame
    O = np.zeros(2)

    xA, yA = L/2, l*np.cos(np.arcsin(L/(2*l)))
    A = np.array([xA, yA])

    xB, yB = -xA, yA
    B = np.array([xB, yB])

    C = computeC(O, A)

    # vertical translation (C becomes the origin)
    O -= C
    A -= C
    B -= C
    C -= C

    return O, A, B, C


def compute_interpolation_polygon(n, O, A, C):
    '''
    n: int, >= 2
    O, A, C: np.array, shape = (2,)

    Computes and returns the vertexes of the n-gon
    interpolating the arc (BOA)
    '''
    # theta_m = (CO, CA), r = norm(OC)
    r = np.linalg.norm(C-O)
    theta_m = np.arccos(np.dot(-C+O, -C+A)/r**2)  # rad

    thetas = -(np.pi/2+theta_m) + np.arange(n)/(n-1)*2*theta_m
    polygon_x = r*np.cos(thetas)
    polygon_y = r*np.sin(thetas)

    polygon = np.stack((polygon_x, polygon_y)).T + C

    return polygon


def compute_boxes_features(polygon, thickness, depth, xyz_slack, xyz_sign):
    '''
    polygon: np.array, shape = (nbox+1,)
    thickness, depth: float
    xyz_slack: array, shape = (3,)
    xyz_sign: array, shape = (3,), abs(xyz_sign) = [1, 1, 1]

    Return geometric characteristics (pos, angle, and size)
    of boxes forming polygon's edges

    The final frame is the 3D frame defined by xyz_slack, xyz_sign and color_seq
    '''
    # number of boxes = number of edges of the polygon
    n_box = len(polygon)-1

    # width of boxes
    width = np.linalg.norm(polygon[1]-polygon[0])

    # half-sizes of boxes and remove useless precision
    h_size = round_to_3(np.array([width/2, thickness/2, depth/2]))

    # angle around z
    u = np.array([0, -1])
    polygon_edges = polygon[1:] - polygon[:-1]
    thetas_Z = np.array(
        [np.arccos(np.dot(u, polygon_edges[i])/width) - np.pi/2 for i in range(n_box)])
    thetas_Z *= xyz_sign[2]
    thetas_3D = np.hstack(
        (np.zeros((n_box, 2)), np.rad2deg(thetas_Z).reshape((n_box, 1))))
    thetas_3D = round_to_3(thetas_3D)

    # pos of the centers of the boxes
    pos_2D = (polygon[:n_box]+polygon[1:])/2  # Bi = (Ci+1 + Ci)/2

    thickness_slack = np.array(
        [np.cos(thetas_Z-np.pi/2), np.sin(thetas_Z-np.pi/2)]).T*thickness/2
    pos_2D += thickness_slack

    pos_3D = np.hstack((pos_2D, np.ones(n_box).reshape((n_box, 1)) * depth/2))
    pos_3D += np.tile(xyz_slack, (n_box, 1))  # apply slack
    pos_3D *= xyz_sign  # mult axis signs
    pos_3D = round_to_3(pos_3D)  # remove useless precision

    return pos_3D, h_size, thetas_3D


def boxes_to_XML(positions, size, thetas, color_seq, def_class=None, prefix=None, root_name="generated"):
    '''
    positions, thetas: np.array, shape = (nbox, 3)
    size: np.array, shape = (3,)
    color_seq: array, shape = (3,), sort(color_seq) = ['b', 'g', 'r']
    prefix, root_name: string

    Returns an element tree object containing nbox boxes
    with pos, thetas and size features, in color_seq order

    root_name is the name of the root element wrapping geoms; it is removed by MuJoCo's parser
    if a prefix is provided, then boxes will be named: prefix+"_"+index
    '''

    root = ET.Element(root_name)

    for index, (pos, theta) in enumerate(zip(positions, thetas)):
        geom = ET.SubElement(root, "geom")

        geom.set("type", "box")
        if prefix is not None:
            geom.set("name", f"{prefix}_{index}")
        geom.set(
            "pos", f"{pos[color_seq.index('r')]} {pos[color_seq.index('g')]} {pos[color_seq.index('b')]}")
        geom.set(
            "size", f"{size[color_seq.index('r')]} {size[color_seq.index('g')]} {size[color_seq.index('b')]}")
        geom.set(
            "euler", f"{theta[color_seq.index('r')]} {theta[color_seq.index('g')]} {theta[color_seq.index('b')]}")
        if def_class is not None:
            geom.set("class", def_class)

    return root


def draw_OABC_schematics(O, A, B, C, polygon, xyz_sign, color_seq):
    '''
    O, A, B, C: np.array, shape = (2,)
    polygon: np.array, shape = (nbox+1,)
    xyz_sign: array, shape = (3,), abs(xyz_sign) = [1, 1, 1]
    color_seq: array, shape = (3,), sort(color_seq) = ['b', 'g', 'r']

    Plots a schematics of O, A, B, C points, the interpolating polygon, the
    interpolated circle and the frame
    '''
    polygon_xy = np.array(polygon).T

    # draw a figure to check
    _, ax = plt.subplots()

    # compute limits of schematics
    min_x = min(polygon_xy[0])
    max_x = max(polygon_xy[0])
    eps_x = .05*(max_x-min_x)
    ax.set_xlim([min_x-eps_x, max_x+eps_x])

    min_y = min(min(polygon_xy[1]), C[1], O[1])
    max_y = max(max(polygon_xy[1]), C[1], O[1])
    eps_y = .05*(max_y-min_y)
    ax.set_ylim([min_y-eps_y, max_y+eps_y])

    plt.gca().set_aspect('equal', adjustable='box')

    # scatter O, A, B, C and add label
    OABC = np.array([O, A, B, C])
    OABC_labels = ['O', 'A', 'B', 'C']
    label_offset = np.array([eps_x, eps_y])/3

    plt.scatter(*OABC.T, color='black')
    for point, label in zip(OABC, OABC_labels):
        ax.annotate(label, tuple(point), xytext=tuple(point+label_offset))

    # draw the interpolated circle
    r = np.linalg.norm(C-O)
    theta_m = np.arccos(np.dot(-C+O, -C+A)/r**2)/np.pi*180  # degree

    ax.add_patch(pat.Arc(xy=tuple(C), width=2*r, height=2*r,
                 angle=-90, theta1=-theta_m, theta2=theta_m, color='orange'))

    # draw interpolation polygon
    plt.scatter(polygon_xy[0], polygon_xy[1], color='b')
    for i in range(len(polygon_xy[0])-1):
        plt.plot(polygon_xy[0][i:i+2], polygon_xy[1][i:i+2], color='b')

    # plot the frame after application of xyz_sign and color_seq
    plotFrame(ax, origin=[min_x+4*eps_x, max_y-6*eps_y],
              xyz_sign=xyz_sign, color_seq=color_seq)

    plt.show()
