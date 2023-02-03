from xml.etree import ElementTree as ET

from shared_utils.general import list_to_string, round_to_3
from soft_tissues_eval.utils import get_micro_bodies, get_extreme_micro_bodies, get_coord_micro_body


def add_micro_geom(composite_parent, mass, geom_type="sphere", size=None, rgba=None, solref=None, solimp=None, margin=None, gap=None):
    geom = ET.SubElement(composite_parent, "geom")

    geom.set("type", f"{geom_type}")
    geom.set("mass", f"{mass}")

    if size is not None:
        geom.set("size", list_to_string(size))
    if rgba is not None:
        geom.set("rgba", list_to_string(rgba))
    if solref is not None:
        geom.set("solref", list_to_string(solref))
    if solimp is not None:
        geom.set("solimp", list_to_string(solimp))
    if margin is not None:
        geom.set("margin", f"{margin}")
    if gap is not None:
        geom.set("gap", f"{gap}")

    return geom


COMPOSITE_3D_TYPES = ["box", "cylinder", "ellipsoid"]


def add_composite_element(body_parent, cmp_type, count, spacing, offset=None, prefix=None, solrefsmooth=None, solimpsmooth=None):
    if cmp_type not in COMPOSITE_3D_TYPES:
        print(f"type must be in {COMPOSITE_3D_TYPES}")
        exit(1)

    composite = ET.SubElement(body_parent, "composite")

    composite.set("type", cmp_type)
    composite.set("count", list_to_string(count))
    composite.set("spacing", f"{spacing}")

    if offset is not None:
        composite.set("offset", list_to_string(offset))
    if prefix is not None:
        composite.set("prefix", prefix)
    if solrefsmooth is not None:
        composite.set("solrefsmooth", list_to_string(solrefsmooth))
    if solimpsmooth is not None:
        composite.set("solimpsmooth", list_to_string(solimpsmooth))

    return composite


def create_body_element(name=None, pos=None):
    body = ET.Element("body")

    if name is not None:
        body.set("name", name)

    if pos is not None:
        body.set("pos", list_to_string(pos))

    return body


def add_collider_extension_slide(parent_body):
    slide = ET.SubElement(parent_body, "joint")
    slide.set("type", "slide")
    slide.set("axis", "1 0 0")

    return slide


def create_test_collider(name, pos, cmp_type, count, spacing, mass, offset=None, prefix=None, solrefsmooth=None, solimpsmooth=None, geom_type="sphere", size=None, rgba=None, solref=None, solimp=None, margin=None, gap=None):
    collider = create_body_element(name, pos)

    add_collider_extension_slide(collider)

    composite = add_composite_element(
        collider, cmp_type, count, spacing, offset, prefix, solrefsmooth, solimpsmooth)

    add_micro_geom(composite, mass, geom_type, size,
                   rgba, solref, solimp, margin, gap)

    return collider


def add_glitching_collider_slide(parent_body, name):
    slide = ET.SubElement(parent_body, "joint")
    slide.set("name", name)
    slide.set("type", "slide")
    slide.set("axis", "0 0 -1")

    return slide


def create_glitching_collider(name, pos, cmp_type, count, spacing, mass, slide_name, offset=None, prefix=None, solrefsmooth=None, solimpsmooth=None, geom_type="sphere", size=None, rgba=None, solref=None, solimp=None, margin=None, gap=None):
    collider = create_body_element(name, pos)

    add_glitching_collider_slide(collider, slide_name)

    composite = add_composite_element(
        collider, cmp_type, count, spacing, offset, prefix, solrefsmooth, solimpsmooth)

    add_micro_geom(composite, mass, geom_type, size,
                   rgba, solref, solimp, margin, gap)

    return collider


def create_extender(fixed_name, mobile_name, slide_name, gap, h_thickness=.005, h_length=.05, slide_range=None):
    fixed_part = ET.Element("body")
    fixed_part.set("name", fixed_name)

    fixed_box = ET.SubElement(fixed_part, "geom")
    fixed_box.set("type", "box")
    fixed_box.set("size", f"{h_thickness} {h_length} {h_length}")

    mobile_part = ET.SubElement(fixed_part, "body")
    mobile_part.set("name", mobile_name)
    mobile_part.set("pos", f"{gap} 0 0")

    mobile_box = ET.SubElement(mobile_part, "geom")
    mobile_box.set("type", "box")
    mobile_box.set("size", f"{h_thickness} {h_length} {h_length}")

    slide = ET.SubElement(mobile_part, "joint")
    slide.set("name", slide_name)
    slide.set("type", "slide")
    slide.set("axis", "1 0 0")
    slide.set("limited", "true")

    if slide_range is None:
        slide_range = round_to_3([-.25*gap, 10*gap])
    slide.set("range", list_to_string(slide_range))

    return fixed_part


def create_basic_actuator(extender_slide, name=None, gear=1):
    actuator = ET.Element("actuator")

    motor = ET.SubElement(actuator, "motor")
    if name is not None:
        motor.set("name", name)
    motor.set("joint", extender_slide)
    motor.set("gear", f"{gear}")

    return actuator


def add_weld_constraint(equality_element, body1, body2):
    weld = ET.SubElement(equality_element, "weld")
    weld.set("body1", body1)
    weld.set("body2", body2)


def is_centered_micro_geom(micro_geom, count, axis):
    nber_geoms = count[axis]
    centered_coord = int(get_coord_micro_body(
        micro_geom)[axis]) - nber_geoms // 2

    return (centered_coord == 0) or (nber_geoms % 2 == 0 and centered_coord == -1)


def create_extender_equality_constraints(model_geoms, fixed_body, mobile_body, composite_prefix, count=None, policy=None):
    equality = ET.Element("equality")
    micro_geoms = get_micro_bodies(model_geoms, composite_prefix)
    left_micro_geoms, right_micro_geoms = get_extreme_micro_bodies(
        micro_geoms, 0)

    if policy == "CENTERED" and count is not None:
        left_micro_geoms = [geom for geom in left_micro_geoms if (
            is_centered_micro_geom(geom, count, 1) and is_centered_micro_geom(geom, count, 2))]
        right_micro_geoms = [geom for geom in right_micro_geoms if (
            is_centered_micro_geom(geom, count, 1) and is_centered_micro_geom(geom, count, 2))]

    for geom in left_micro_geoms:
        add_weld_constraint(equality, fixed_body, geom)

    for geom in right_micro_geoms:
        add_weld_constraint(equality, mobile_body, geom)

    return equality
