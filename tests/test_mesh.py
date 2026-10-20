"""Mesh generation: first-cell formula, wall-spacing measurement, structured C-mesh and hybrid gmsh mesh."""
import os

import numpy as np
import pytest

from tests.conftest import naca00xx

pytest.importorskip("gmsh")
from Scripts.Meshing import hybrid, meshing, wall_spacing
from Scripts.Batch import batch_runner as br  


# reading an SU2 mesh (test helper)
def read_su2(path):
    """(cell node lists, points, marker -> number of faces) of a 2-D SU2 mesh."""
    lines = open(path).read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("NELEM"))
    n_elem = int(lines[i].split("=")[1])
    cells = [[int(v) for v in l.split()[1:-1]] for l in lines[i + 1:i + 1 + n_elem]]
    j = next(k for k, l in enumerate(lines) if l.startswith("NPOIN"))
    n_point = int(lines[j].split("=")[1].split()[0])
    points = np.array([[float(v) for v in l.split()[:2]] for l in lines[j + 1:j + 1 + n_point]])
    markers = {}
    for k, l in enumerate(lines):
        if l.startswith("MARKER_TAG"):
            markers[l.split("=")[1].strip()] = int(lines[k + 1].split("=")[1])
    declared = int(next(l for l in lines if l.startswith("NMARK")).split("=")[1])
    return cells, points, markers, declared


def cell_areas(cells, points):
    """Signed areas (shoelace) of the triangles and quads of a mesh."""
    areas = []
    for n in sorted({len(c) for c in cells}):
        idx = np.array([c for c in cells if len(c) == n])
        x, y = points[idx, 0], points[idx, 1]
        areas.append(0.5 * np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1))
    return np.concatenate(areas)



def test_first_cell_scaling():
    f = meshing.calculate_boundary_layer_thickness
    assert f(5e6, 0.1) < f(5e5, 0.1) < f(5e4, 0.1)                      # thinner with Reynolds number
    assert f(5e5, 0.1, y_plus=5.0) == pytest.approx(5 * f(5e5, 0.1, y_plus=1.0))   # proportional to y+
    assert f(5e5, 0.1) == pytest.approx(f(5e5, 0.8))                    # independent of Mach (nu = U c / Re)


@pytest.mark.parametrize("re, yplus", [(5e4, 1.0), (5e6, 1.0), (5e5, 30.0)])
def test_design_yplus_inverts_the_first_cell_formula(re, yplus):
    y = meshing.calculate_boundary_layer_thickness(re, 0.1, y_plus=yplus)
    assert wall_spacing.design_yplus(y, re) == pytest.approx(yplus)


def tiny_mesh(path, with_wall=True):
    """One quad of height 0.1 on a wall (nodes 0, 1) plus a second quad above it."""
    lines = ["NDIME= 2", "NELEM= 2", "9 0 1 3 2 0", "9 2 3 5 4 1", "NPOIN= 6",
             "0.0 0.0 0", "1.0 0.0 1", "0.0 0.1 2", "1.0 0.1 3", "0.0 0.3 4", "1.0 0.3 5",
             "NMARK= 1", "MARKER_TAG= Airfoil" if with_wall else "MARKER_TAG= Inlet",
             "MARKER_ELEMS= 1", "3 0 1"]
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_measure_wall_spacing_on_a_hand_built_mesh(tmp_path):
    coords, spacing = wall_spacing.measure_wall_spacing(tiny_mesh(tmp_path / "t.su2"))
    assert len(spacing) == 2 and np.allclose(spacing, 0.1)             # first-cell height, not the second layer


def test_measure_wall_spacing_without_airfoil_marker(tmp_path):
    assert wall_spacing.measure_wall_spacing(tiny_mesh(tmp_path / "t.su2", with_wall=False)) == (None, None)


#structured C-mesh
@pytest.fixture(scope="module")
def structured(tmp_path_factory):
    """Structured meshes of a NACA 0012 at two Reynolds numbers (about 4 s each)."""
    x, y = naca00xx()
    out, start = {}, os.getcwd()
    try:
        for re in (5e4, 5e6):
            d = tmp_path_factory.mktemp(f"struct_{re:g}")
            os.chdir(d)  # the mesher writes airfoil.su2 into the working directory
            meshing.generate_mesh(x, y, re, 0.1, y_plus=1.0, show_graphics=False, hide_output=True)
            out[re] = str(d / "airfoil.su2")
    finally:
        os.chdir(start)
    return out


@pytest.mark.parametrize("re", [5e4, 5e6])
def test_structured_mesh_file_is_complete(structured, re):
    cells, points, markers, declared = read_su2(structured[re])
    assert 2e5 < len(cells) < 1e6
    assert declared == 3 and set(markers) == {"Airfoil", "Inlet", "Outlet"}
    assert all(n > 100 for n in markers.values())
    assert np.isfinite(points).all()
    assert np.abs(points).max() == pytest.approx(25.0, abs=0.5)         # far field: 15 chords upstream, 25 downstream


@pytest.mark.parametrize("re", [5e4, 5e6])
def test_structured_mesh_has_no_inverted_or_degenerate_cells(structured, re):
    cells, points, _, _ = read_su2(structured[re])
    areas = cell_areas(cells, points)
    assert np.all(areas > 0)                                            # consistent orientation, none collapsed


@pytest.mark.parametrize("re", [5e4, 5e6])
def test_structured_mesh_delivers_the_requested_wall_spacing(structured, re):
    _, spacing = wall_spacing.measure_wall_spacing(structured[re])
    yp = wall_spacing.design_yplus(spacing[np.isfinite(spacing)], re)
    assert 0.85 < np.median(yp) <= 1.02                                 # measured 0.98 and 0.96 for target 1
    assert yp.max() / yp.min() < 1.1                                    # even spacing along the wall


#hybrid gmsh mesh
def min_triangle_angle(cells, points):
    tri = np.array([c for c in cells if len(c) == 3])
    p = points[tri]
    angles = []
    for i in range(3):
        a, b, c = p[:, i], p[:, (i + 1) % 3], p[:, (i + 2) % 3]
        u, v = b - a, c - a
        cosine = np.sum(u * v, axis=1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    return float(np.min(angles))


@pytest.mark.parametrize("re, mach", [(1e4, 0.1), (5e4, 0.1), (5e5, 0.15), (3e7, 0.8)])
def test_hybrid_uses_the_same_first_cell_formula_as_the_structured_mesher(re, mach):
    assert hybrid.calculate_boundary_layer_thickness(re, mach) == meshing.calculate_boundary_layer_thickness(re, mach)


@pytest.mark.parametrize("re", [1e4, 5e4, 2e5, 5e5, 5e6, 3e7])
def test_hybrid_boundary_layer_stack_always_has_enough_layers(re):
    first = hybrid.calculate_boundary_layer_thickness(re, 0.1)
    thickness = hybrid.boundary_layer_thickness(first)
    layers = np.log(1 + (hybrid.BL_RATIO - 1) * thickness / first) / np.log(hybrid.BL_RATIO)
    assert thickness >= hybrid.BL_MIN_THICKNESS
    assert layers >= hybrid.BL_MIN_LAYERS - 1e-6        


def test_hybrid_boundary_layer_is_unchanged_at_high_reynolds_number():
    first = hybrid.calculate_boundary_layer_thickness(5e6, 0.1)
    assert hybrid.boundary_layer_thickness(first) == hybrid.BL_MIN_THICKNESS


@pytest.fixture(scope="module")
def hybrid_meshes(tmp_path_factory):
    """Hybrid meshes of a NACA 0012 at three Reynolds numbers (about 15 s each)."""
    upper, lower = br.split_surfaces(*naca00xx())
    out, start = {}, os.getcwd()
    try:
        for re in (5e4, 5e5, 5e6):
            d = tmp_path_factory.mktemp(f"hybrid_{re:g}")
            os.chdir(d)
            hybrid.generate_hybrid(upper, lower, re, 0.1, y_plus=1.0, show_graphics=False, hide_output=True)
            out[re] = str(d / "airfoil.su2")
    finally:
        os.chdir(start)
    return out


@pytest.mark.slow
@pytest.mark.parametrize("re", [5e4, 5e5, 5e6])
def test_hybrid_mesh_file_is_complete_and_sound(hybrid_meshes, re):
    cells, points, markers, declared = read_su2(hybrid_meshes[re])
    assert 3e5 < len(cells) < 8e5
    assert declared == 3 and set(markers) == {"Airfoil", "Inlet", "Outlet"}
    assert np.isfinite(points).all()
    areas = cell_areas(cells, points)
    assert np.all(areas != 0) and (np.all(areas > 0) or np.all(areas < 0))   # consistent orientation, none collapsed
    assert min_triangle_angle(cells, points) > 5.0                           # measured about 10 degrees


@pytest.mark.slow
@pytest.mark.parametrize("re", [5e4, 5e5, 5e6])
def test_hybrid_mesh_delivers_the_requested_wall_spacing(hybrid_meshes, re):
    _, spacing = wall_spacing.measure_wall_spacing(hybrid_meshes[re])
    yp = wall_spacing.design_yplus(spacing[np.isfinite(spacing)], re)
    assert 0.95 < np.median(yp) <= 1.02                                      # measured 1.000 (0.22 before the fix)
