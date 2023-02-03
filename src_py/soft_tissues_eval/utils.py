import re
import numpy as np


def is_micro_body(body_name, prefix, element_type="B"):
    '''
    Use prefix = ".*" to match any micro body name
    '''
    # "B" -> look for body ; "G" -> geom
    assert element_type in ["B", "G"]

    pattern = f"^{prefix}{element_type}[0-9]+_[0-9]+_[0-9]+$"
    return re.match(pattern, body_name) is not None


def get_coord_micro_body(micro_body_name, element_type="B"):
    ''' 
    One must ensure that: is_micro_body(micro_body_name, ".*", element_type) is True
    '''
    # "B" -> look for body ; "G" -> geom
    assert element_type in ["B", "G"]

    coords_str = micro_body_name.split("_")[-3:]
    coords_str[0] = coords_str[0].split(element_type)[-1]

    return list(map(int, coords_str))


def get_micro_bodies(bodies, prefix, element_type="B"):
    # "B" -> look for body ; "G" -> geom
    assert element_type in ["B", "G"]

    return [body for body in bodies if is_micro_body(body, prefix, element_type)]


def get_extreme_micro_bodies(micro_bodies, axis):
    '''
    axis == 0 (ie x) or 1 (ie y) or 2 (ie z)
    '''

    def comp_micro_bodies(micro_body):
        '''
        Return coord on the chosen axis
        '''
        return get_coord_micro_body(micro_body)[axis]

    max_coord = comp_micro_bodies(max(micro_bodies, key=comp_micro_bodies))
    min_coord = comp_micro_bodies(min(micro_bodies, key=comp_micro_bodies))

    argmax_micro_bodies = [
        geom for geom in micro_bodies if comp_micro_bodies(geom) == max_coord]
    argmin_micro_bodies = [
        geom for geom in micro_bodies if comp_micro_bodies(geom) == min_coord]

    return argmin_micro_bodies, argmax_micro_bodies


def compute_composite_body_mass(model, prefix, body_name):
    ''' 
    model: PyMjModel (mujoco-py)
    prefix, body_name: string

    Computes mass of the composite body called 'body_name' with geom 'prefix'
    in the 'model'
    '''
    # get number of micro geoms
    nber_micro_bodies = len(get_micro_bodies(model.body_names, prefix))

    # get mass of one micro geom
    first_micro_body_index = next(i for (i, body) in enumerate(
        model.body_names) if is_micro_body(body, prefix))
    micro_body_mass = model.body_mass[first_micro_body_index]

    # get mass of root body
    root_body_index = next(i for (i, body) in enumerate(
        model.body_names) if body == body_name)
    root_body_mass = model.body_mass[root_body_index]

    return root_body_mass + nber_micro_bodies*micro_body_mass


def edit_cmp_smoothness(root, cmp_prefix,
                        solrefs=[None, None, None],
                        solimps=[None, None, None]):
    '''
    solref: string (format: "float float")
    cmp_prefix: string (prefix of the targeted composite object)
    root: Element (root of the ElementTree of the target MJCF file)
    '''
    assert len(solrefs) == len(solimps) == 3

    CENTER_CONSTRAINT_INDEX = 0
    NEIGH_CONSTRAINT_INDEX = 1
    VOL_CONSTRAINT_INDEX = 2

    SOLREF_DEFAULT = "0.02 1"
    SOLIMP_DEFAULT = "0.9 0.95 0.001 0.5 2"
    for i, (solref, solimp) in enumerate(zip(solrefs, solimps)):
        if solref is None:
            solrefs[i] = SOLREF_DEFAULT
        if solimp is None:
            solimps[i] = SOLIMP_DEFAULT

    # equality constraints will be edited
    equality_elements = root.findall("./equality")
    assert len(equality_elements) == 1

    equality_element = equality_elements[0]

    # all composite joint names match the following regex
    # we assume no other joint match this regex
    cmp_J_regex = f"^{cmp_prefix}J[0-9]+_[0-9]+_[0-9]+$"
    cmp_J_m = re.compile(cmp_J_regex)
    assert cmp_J_m.match(f"{cmp_prefix}J4_4_9") is not None \
        and cmp_J_m.match(f"{cmp_prefix}J_4_9") is None

    cmp_T_regex = f"^{cmp_prefix}T$"
    cmp_T_m = re.compile(cmp_T_regex)
    assert cmp_T_m.match(f"{cmp_prefix}T") is not None \
        and cmp_T_m.match(f"{cmp_prefix}F") is None

    for constraint in equality_element.iter('joint'):
        if 'joint1' in constraint.attrib and cmp_J_m.match(constraint.attrib['joint1']):

            if 'joint2' in constraint.attrib and cmp_J_m.match(constraint.attrib['joint2']):
                # this is a neighbour constraint
                constraint.attrib['solref'] = solrefs[NEIGH_CONSTRAINT_INDEX]
                constraint.attrib['solimp'] = solimps[NEIGH_CONSTRAINT_INDEX]

            else:
                # this is a central constraint
                constraint.attrib['solref'] = solrefs[CENTER_CONSTRAINT_INDEX]
                constraint.attrib['solimp'] = solimps[CENTER_CONSTRAINT_INDEX]

    for constraint in equality_element.iter('tendon'):
        if 'tendon1' in constraint.attrib and cmp_T_m.match(constraint.attrib['tendon1']):
            # this is a volume contraint
            constraint.attrib['solref'] = solrefs[VOL_CONSTRAINT_INDEX]
            constraint.attrib['solimp'] = solimps[VOL_CONSTRAINT_INDEX]


def edit_MjModel_cmp_smoothness(model, cmp_prefix: str, solref: np.ndarray = np.array([-2777, -111])) -> None:
    '''
    model: target MjModel object
    cmp_prefix: string; prefix of the target composite body
    solref: solref values that must be assigned to all constraints of the cmp body

    Modifies the solref parameter of all constraints structuring a given composite body
    '''

    ''' First: Set composite joint constraints'''
    # regex for composite body joints
    cmp_J_regex = f"^{cmp_prefix}J[0-9]+_[0-9]+_[0-9]+$"
    cmp_J_m = re.compile(cmp_J_regex)

    # get matching joint names
    cmp_J_names = [
        name for name in model.joint_names if cmp_J_m.match(name) is not None]

    # convert names to ids; also include total tendon name
    cmp_J_ids = np.array([model.joint_name2id(name) for name in cmp_J_names]
                         + [model.tendon_name2id(f"{cmp_prefix}T")])

    # deduce constraint mask (True for all constraints involving the joints)
    constraint_mask = np.isin(model.eq_obj1id, cmp_J_ids)

    # set constraints solrefs
    model.eq_solref[constraint_mask] = np.tile(
        solref, (np.count_nonzero(constraint_mask), 1))
