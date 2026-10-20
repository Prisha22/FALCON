import gmsh
from scipy.interpolate import splprep, splev
import numpy as np


# One first-cell formula for both meshers: the Re-based one in meshing.py (the SU2 config sets the viscosity from
# Re with REYNOLDS_LENGTH = 1). The dimensional sea-level-air formula this file used before gave a wall cell about
# 4.6x too thin at Re 5e5 (design y+ 0.22 for a target of 1).
from Scripts.Meshing.meshing import calculate_boundary_layer_thickness  # noqa: E402,F401

BL_RATIO = 1.04            # growth ratio of the boundary-layer layers
BL_MIN_THICKNESS = 0.001   # boundary-layer thickness used when it already holds enough layers
BL_MIN_LAYERS = 15         # at low Re the wall cell is large, so the layer stack is thickened to keep this many


def boundary_layer_thickness(first_cell, ratio=BL_RATIO, min_layers=BL_MIN_LAYERS, min_thickness=BL_MIN_THICKNESS):
    """Total thickness of the structured layer stack: 1e-3 chord, or more when that would hold fewer than
    `min_layers` layers of the given first cell and growth ratio."""
    return max(min_thickness, first_cell * (ratio ** min_layers - 1.0) / (ratio - 1.0))


def generate_hybrid(upper_surface, lower_surface, Re, M, y_plus=1.0, show_graphics: bool = True,
                    output_format: str = '.su2',
                    hide_output: bool = True):
    length_le = 0.1
    inlet_radius = 15
    downstream_distance = 25
    r_inlet = inlet_radius
    center = (length_le, 0, 0)
    name = "airfoil"

    first_cell_thickness = calculate_boundary_layer_thickness(Re, M, y_plus=y_plus)
    print(f"Target y+: {y_plus} -> Calculated First Cell Thickness: {first_cell_thickness:.4e} m")

    af_upper_pointdata = []
    af_lower_pointdata = []
    for point in upper_surface:
        x = point[0]
        y = point[1]
        af_upper_pointdata.append([x, y, 0])
    for point in lower_surface:
        x = point[0]
        y = point[1]
        af_lower_pointdata.append([x, y, 0])
    gmsh.initialize()
    gmsh.model.add(name)
    if hide_output:
        gmsh.option.setNumber("General.Terminal", 0)

    af_lower_pointdata.pop()
    af_lower_pointdata.pop(0)
    model = gmsh.model.geo
    af_upper_points = [gmsh.model.geo.addPoint(x, y, z) for x, y, z in af_upper_pointdata]
    af_lower_points = [gmsh.model.geo.addPoint(x, y, z) for x, y, z in af_lower_pointdata]

    pc = gmsh.model.geo.addPoint(0.5, 0, 0)
    pc1 = gmsh.model.geo.addPoint(-0.7, 0, 0)
    pc2 = gmsh.model.geo.addPoint(0.5, 1.2, 0)
    pc3 = gmsh.model.geo.addPoint(1.7, 0, 0)
    pc4 = gmsh.model.geo.addPoint(0.5, -1.2, 0)
    c8 = gmsh.model.geo.addCircleArc(pc1, pc, pc2)
    c9 = gmsh.model.geo.addCircleArc(pc2, pc, pc3)
    c5 = gmsh.model.geo.addCircleArc(pc3, pc, pc4)
    c6 = gmsh.model.geo.addCircleArc(pc4, pc, pc1)
    circle_loop = gmsh.model.geo.addCurveLoop([c8, c9, c5, c6])

    # Share the upper surface's LE and TE points by adding them to the ends. Overwriting af_lower_points[0] and [-1]
    # instead dropped two real lower-surface points and left a corner at the nose (1.2e-3 c off the wall on E387).
    af_lower_points.insert(0, af_upper_points[-1])
    af_lower_points.append(af_upper_points[0])

    af_upper = gmsh.model.geo.addSpline(af_upper_points)
    af_lower = gmsh.model.geo.addSpline(af_lower_points)
    airfoil_loop = gmsh.model.geo.addCurveLoop([af_upper, af_lower])

    center_point = gmsh.model.geo.addPoint(center[0], center[1], 0)
    inlet_top_point = gmsh.model.geo.addPoint(center[0], r_inlet, 0)
    inlet_bottom_point = gmsh.model.geo.addPoint(center[0], -r_inlet, 0)
    front_line = gmsh.model.geo.addCircleArc(inlet_top_point, center_point, inlet_bottom_point)

    top_wake_point = gmsh.model.geo.addPoint(downstream_distance, r_inlet, 0)
    bottom_wake_point = gmsh.model.geo.addPoint(downstream_distance, -r_inlet, 0)
    top_line = gmsh.model.geo.addLine(inlet_top_point, top_wake_point)
    outlet_line = gmsh.model.geo.addLine(top_wake_point, bottom_wake_point)
    bottom_line = gmsh.model.geo.addLine(bottom_wake_point, inlet_bottom_point)

    back_loop = gmsh.model.geo.addCurveLoop([top_line, outlet_line, bottom_line, -front_line])
    back_section = gmsh.model.geo.addPlaneSurface([back_loop, circle_loop])
    airfoil_section = gmsh.model.geo.addPlaneSurface([circle_loop, airfoil_loop])

    gmsh.model.geo.synchronize()

    boundary_layer_field_id = gmsh.model.mesh.field.add("BoundaryLayer")
    gmsh.model.mesh.field.setNumbers(boundary_layer_field_id, "CurvesList", [af_upper, af_lower])
    gmsh.model.mesh.field.setNumber(boundary_layer_field_id, "Size", first_cell_thickness)
    gmsh.model.mesh.field.setNumber(boundary_layer_field_id, "Thickness", boundary_layer_thickness(first_cell_thickness))
    gmsh.model.mesh.field.setNumber(boundary_layer_field_id, "SizeFar", 0.0009)
    gmsh.model.mesh.field.setNumber(boundary_layer_field_id, "Ratio", BL_RATIO)
    gmsh.model.mesh.field.setNumber(boundary_layer_field_id, "Quads", 1)
    gmsh.option.setNumber('Mesh.BoundaryLayerFanElements', 10)
    gmsh.model.mesh.field.setNumbers(boundary_layer_field_id, 'FanPointsList', [1])
    gmsh.model.mesh.field.setAsBoundaryLayer(boundary_layer_field_id)

    box_field_id = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(box_field_id, "VIn", 0.8)
    gmsh.model.mesh.field.setNumber(box_field_id, "VOut", 1.5)
    gmsh.model.mesh.field.setNumber(box_field_id, "XMin", -1.5)
    gmsh.model.mesh.field.setNumber(box_field_id, "XMax", 2)
    gmsh.model.mesh.field.setNumber(box_field_id, "YMin", -1.7)
    gmsh.model.mesh.field.setNumber(box_field_id, "YMax", 1.7)
    gmsh.model.mesh.field.setNumber(box_field_id, "Thickness", 7)

    box_field_id1 = gmsh.model.mesh.field.add("Box")
    # Cell count ~ 57k + 12.5 / VIn**2, nearly airfoil independent: 0.0053 -> ~500k, 0.003 -> ~1.44M
    gmsh.model.mesh.field.setNumber(box_field_id1, "VIn", 0.0053)
    gmsh.model.mesh.field.setNumber(box_field_id1, "VOut", 0.8)
    gmsh.model.mesh.field.setNumber(box_field_id1, "XMin", -0.7)
    gmsh.model.mesh.field.setNumber(box_field_id1, "XMax", 1.7)
    gmsh.model.mesh.field.setNumber(box_field_id1, "YMin", -1)
    gmsh.model.mesh.field.setNumber(box_field_id1, "YMax", 1)
    gmsh.model.mesh.field.setNumber(box_field_id1, "Thickness", 10)

    min_field_id = gmsh.model.mesh.field.add("Min")
    # The BoundaryLayer field is applied through setAsBoundaryLayer only. Listing it here as well lets its wall
    # sizing drive the whole inner disk, which stretches triangles into radial slivers (min angle 0.085 deg).
    gmsh.model.mesh.field.setNumbers(min_field_id, "FieldsList", [box_field_id, box_field_id1])


    gmsh.model.addPhysicalGroup(1, [af_upper, af_lower], name='Airfoil')
    gmsh.model.addPhysicalGroup(1, [front_line], name='Inlet')
    gmsh.model.addPhysicalGroup(1, [top_line, outlet_line, bottom_line], name='Outlet')
    gmsh.model.addPhysicalGroup(2, [back_section, airfoil_section], name='FlowDomain')
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field_id)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    file_path = name + '.su2'
    gmsh.write(file_path)
    gmsh.option.setNumber("Mesh.SurfaceFaces", 1)

    if show_graphics:
        gmsh.fltk.run()
    gmsh.finalize()

    lines = []
    with open(file_path, 'r') as source_file:
        lines = source_file.readlines()
    with open(file_path, 'w') as modified_file:
        for line in lines:
            if line.startswith('NMARK'):
                modified_file.write('NMARK= 3\n')
            else:
                modified_file.write(line)