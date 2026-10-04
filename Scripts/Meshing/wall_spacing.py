import numpy as np

def measure_wall_spacing(mesh_path):
    """First-cell height at every airfoil wall node of a written SU2 mesh.
    For each node on the Airfoil marker, the distance to the nearest node that shares a cell with it but is not
    itself on the wall. Exact for the wall-normal quads of both meshers.
    """
    from collections import defaultdict

    lines = open(mesh_path).read().splitlines()
    i = 0
    while not lines[i].startswith('NELEM'):
        i += 1
    n_elem = int(lines[i].split('=')[1])
    i += 1
    cells = [[int(v) for v in line.split()[1:-1]] for line in lines[i:i + n_elem]]
    i += n_elem
    while not lines[i].startswith('NPOIN'):
        i += 1
    n_point = int(lines[i].split('=')[1].split()[0])
    i += 1
    points = np.array([[float(v) for v in line.split()[:2]] for line in lines[i:i + n_point]])

    wall_nodes = []
    for j in range(i + n_point, len(lines)):
        if lines[j].startswith('MARKER_TAG') and 'Airfoil' in lines[j]:
            for k in range(int(lines[j + 1].split('=')[1])):
                face = lines[j + 2 + k].split()
                wall_nodes += [int(face[1]), int(face[2])]
            break
    wall_nodes = set(wall_nodes)
    if not wall_nodes:
        return None, None

    touching = defaultdict(list)
    for cell in cells:
        for node in cell:
            if node in wall_nodes:
                touching[node].append(cell)

    ordered = sorted(touching)
    spacing = []
    for node in ordered:
        off_wall = [v for cell in touching[node] for v in cell if v not in wall_nodes]
        if off_wall:
            spacing.append(np.min(np.linalg.norm(points[off_wall] - points[node], axis=1)))
        else:
            spacing.append(np.nan)
    return points[ordered], np.array(spacing)


def design_yplus(first_cell, Re):
    """y+ the mesh is built for, by inverting calculate_boundary_layer_thickness.
    This is the flat-plate correlation read backwards, so it verifies that the mesh actually carries the wall
    spacing it was asked for - it says nothing about the y+ the flow produces, which needs the wall shear from
    a solution and runs several times higher near the leading-edge suction peak.
    """
    cf = (2 * np.log10(float(Re)) - .65) ** -2.3
    return np.asarray(first_cell) * float(Re) * np.sqrt(cf / 2)


def report_wall_spacing(mesh_path, Re, target_yplus):
    """Print the wall spacing a written mesh achieved, against the y+ it was asked for."""
    coords, spacing = measure_wall_spacing(mesh_path)
    if spacing is None or not np.any(np.isfinite(spacing)):
        print("Could not measure wall spacing from the written mesh.")
        return None
    valid = np.isfinite(spacing)
    yplus = design_yplus(spacing[valid], Re)
    print(f"Achieved wall spacing over {valid.sum()} wall nodes: median {np.median(spacing[valid]):.4e} "
          f"(min {spacing[valid].min():.4e}, max {spacing[valid].max():.4e})")
    print(f"  -> design y+ median {np.median(yplus):.3f} against target {target_yplus:g} "
          f"(range {yplus.min():.3f} to {yplus.max():.3f}); the y+ the flow produces needs a solution")
    return {'median': float(np.median(yplus)), 'min': float(yplus.min()), 'max': float(yplus.max()),
            'first_cell_median': float(np.median(spacing[valid])), 'n': int(valid.sum())}
