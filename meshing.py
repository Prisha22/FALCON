#meshing.py
import gmsh
from scipy.interpolate import splprep, splev
import numpy as np
import math


def calculate_boundary_layer_thickness(Re, M, y_plus=1.0):
    """
    Calculate first cell thickness based on Reynolds number, Mach number, and target y+.
    """
    cf = (2 * np.log10(float(Re)) - .65) ** -2.3  # Skin friction coefficient
    tau_w = cf * .5 * 1.225 * (float(M) * 341.348) ** 2  # Wall shear stress
    u_star = np.sqrt(tau_w / 1.225)
    y = y_plus * (1.813e-5) / (1.225 * u_star)  # First cell thickness
    return y


def generate_mesh(xcoords, ycoords, Re, M, y_plus=1.0, show_graphics: bool = True, output_format: str = '.su2',
                  hide_output: bool = True):
    """
    Creates C-block structured mesh with dynamically calculated boundary layer layers.
    """
    length_le = 0.1
    trailing_edge_thickness = 1e-3
    inlet_radius = 15
    downstream_distance = 25
    boundary_growth_rate = 1.2

    first_cell_thickness = calculate_boundary_layer_thickness(Re, M, y_plus=y_plus)
    print(f"Target y+: {y_plus} -> Calculated First Cell Thickness: {first_cell_thickness:.4e} m")

    domain_thickness = inlet_radius
    # Ensure the argument to log is positive
    log_arg = (domain_thickness * (boundary_growth_rate - 1) / first_cell_thickness) + 1
    if log_arg > 0:
        n_volume = int(math.log(log_arg) / math.log(boundary_growth_rate))
    else:
        n_volume = 120  # Fallback for extreme values
    print(f"Using Growth Rate: {boundary_growth_rate} -> Calculated Layers (n_volume): {n_volume}")
    # --- End of fix ---

    n_airfoil = 401
    n_wake = 301
    n_leading_edge = 180
    leading_edge_length = .1

    center = (length_le, 0, 0)
    name = "airfoil"

    af_upper_pointdata = []
    af_lower_pointdata = []
    af_le_pointdata = []
    upper_aft_started = True
    leading_edge_started = False
    lower_aft_started = False
    leading_edge_point = False

    for i in range(len(xcoords)):
        x = xcoords[i]
        y = ycoords[i]
        if x <= length_le and upper_aft_started:
            af_upper_pointdata.append([x, y, 0])
            leading_edge_started = True
            upper_aft_started = False
        if x >= length_le and leading_edge_started:
            af_le_pointdata.append([x, y, 0])
            leading_edge_started = False
            lower_aft_started = True
        if upper_aft_started:
            af_upper_pointdata.append([x, y, 0])
        elif leading_edge_started:
            if x == 0.0 and not leading_edge_point:
                af_le_pointdata.append([x, y, 0])
                leading_edge_point = True
            else:
                af_le_pointdata.append([x, y, 0])
        elif lower_aft_started:
            af_lower_pointdata.append([x, y, 0])

    gmsh.initialize()
    gmsh.model.add(name)

    if hide_output:
        gmsh.option.setNumber("General.Terminal", 0)

    af_upper_pointdata.pop()
    af_lower_pointdata.pop()
    af_lower_pointdata.pop(0)
    model = gmsh.model.geo
    af_upper_points = [gmsh.model.geo.addPoint(x, y, z) for x, y, z in af_upper_pointdata]
    af_lower_points = [gmsh.model.geo.addPoint(x, y, z) for x, y, z in af_lower_pointdata]
    af_le_points = [gmsh.model.geo.addPoint(x, y, z) for x, y, z in af_le_pointdata]
    af_top_point = af_le_points[0]
    af_bottom_point = af_le_points[-1]
    af_te_point = af_upper_points[0]
    af_upper_points.append(af_top_point)
    af_lower_points.insert(0, af_bottom_point)
    af_lower_points.append(af_te_point)
    af_upper = model.addBSpline(af_upper_points)
    af_lower = model.addBSpline(af_lower_points)
    af_le = model.addBSpline(af_le_points)
    center_point = model.addPoint(center[0], center[1], 0)
    inlet_top_point = model.addPoint(center[0], inlet_radius, 0)
    inlet_bottom_point = model.addPoint(center[0], -inlet_radius, 0)
    front_line = model.addCircleArc(inlet_top_point, center_point, inlet_bottom_point)
    afTop_inletTop = model.addLine(af_top_point, inlet_top_point)
    inletBottom_afBottom = model.addLine(inlet_bottom_point, af_bottom_point)
    inlet_loop = model.addCurveLoop([front_line, inletBottom_afBottom, -af_le, afTop_inletTop])
    inlet_section = model.addPlaneSurface([inlet_loop])
    top_te_point = model.addPoint(1, inlet_radius, 0)
    top_line = model.addLine(top_te_point, inlet_top_point)
    topTe_afTe = model.addLine(top_te_point, af_te_point)
    top_loop = model.addCurveLoop([-af_upper, -afTop_inletTop, top_line, -topTe_afTe])
    top_section = model.addPlaneSurface([top_loop])
    bottom_te_point = model.addPoint(1, -inlet_radius, 0)
    bottom_line = model.addLine(inlet_bottom_point, bottom_te_point)
    afTe_bottomTe = model.addLine(af_te_point, bottom_te_point)
    bottom_loop = model.addCurveLoop([-inletBottom_afBottom, -af_lower, -afTe_bottomTe, bottom_line])
    bottom_section = model.addPlaneSurface([bottom_loop])
    top_wake_point = model.addPoint(downstream_distance, inlet_radius, 0)
    center_wake_point = model.addPoint(downstream_distance, 0, 0)
    top_wake_line = model.addLine(top_wake_point, top_te_point)
    center_wake_line = model.addLine(af_te_point, center_wake_point)
    outlet_top = model.addLine(center_wake_point, top_wake_point)
    top_wake_loop = model.addCurveLoop([topTe_afTe, center_wake_line, outlet_top, top_wake_line])
    top_wake_section = model.addPlaneSurface([top_wake_loop])
    bottom_wake_point = model.addPoint(downstream_distance, -inlet_radius, 0)
    outlet_bottom = model.addLine(bottom_wake_point, center_wake_point)
    bottom_wake_line = model.addLine(bottom_te_point, bottom_wake_point)
    bottom_wake_loop = model.addCurveLoop([-center_wake_line, outlet_bottom, bottom_wake_line, afTe_bottomTe])
    bottom_wake_section = model.addPlaneSurface([bottom_wake_loop])

    model.mesh.setTransfiniteCurve(front_line, n_leading_edge, "Bump", coef=-.1)
    model.mesh.setTransfiniteCurve(af_le, n_leading_edge)
    model.mesh.setTransfiniteCurve(afTop_inletTop, n_volume, "Progression", boundary_growth_rate)
    model.mesh.setTransfiniteCurve(inletBottom_afBottom, n_volume, "Progression", -boundary_growth_rate)
    model.mesh.setTransfiniteSurface(inlet_section)
    model.mesh.setRecombine(2, inlet_section)
    model.mesh.setTransfiniteCurve(topTe_afTe, n_volume, "Progression", -boundary_growth_rate)
    te_growth_upper = 1.015
    model.mesh.setTransfiniteCurve(af_upper, n_airfoil, "Progression", -te_growth_upper)
    model.mesh.setTransfiniteCurve(top_line, n_airfoil, "Progression", -te_growth_upper)
    model.mesh.setTransfiniteSurface(top_section)
    model.mesh.setRecombine(2, top_section)
    model.mesh.setTransfiniteCurve(afTe_bottomTe, n_volume, "Progression", boundary_growth_rate)
    model.mesh.setTransfiniteCurve(af_lower, n_airfoil, "Progression", te_growth_upper)
    model.mesh.setTransfiniteCurve(bottom_line, n_airfoil, "Progression", -te_growth_upper)
    model.mesh.setTransfiniteSurface(bottom_section)
    model.mesh.setRecombine(2, bottom_section)
    initial_wake_spacing = trailing_edge_thickness * 2.0
    target_wake_spacing = (downstream_distance - 1.0) / (n_wake * 0.5)
    wake_growth = (target_wake_spacing / initial_wake_spacing) ** (1.0 / (n_wake - 1))
    wake_growth = min(wake_growth, 1.05)
    model.mesh.setTransfiniteCurve(top_wake_line, n_wake, "Progression", -wake_growth)
    model.mesh.setTransfiniteCurve(center_wake_line, n_wake, "Progression", wake_growth)
    model.mesh.setTransfiniteCurve(outlet_top, n_volume, "Progression", boundary_growth_rate)
    model.mesh.setTransfiniteSurface(top_wake_section)
    model.mesh.setRecombine(2, top_wake_section)
    model.mesh.setTransfiniteCurve(bottom_wake_line, n_wake, "Progression", wake_growth)
    model.mesh.setTransfiniteCurve(outlet_bottom, n_volume, "Progression", -boundary_growth_rate)
    model.mesh.setTransfiniteSurface(bottom_wake_section)
    model.mesh.setRecombine(2, bottom_wake_section)

    model.addPhysicalGroup(1, [af_upper, af_le, af_lower], name='Airfoil')
    model.addPhysicalGroup(1, [outlet_top, outlet_bottom, top_line, top_wake_line, bottom_line, bottom_wake_line],
                           name='Outlet')
    model.addPhysicalGroup(1, [front_line], name='Inlet')
    model.addPhysicalGroup(2, [inlet_section, top_section, top_wake_section, bottom_section, bottom_wake_section],
                           name='FlowDomain')
    model.synchronize()
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