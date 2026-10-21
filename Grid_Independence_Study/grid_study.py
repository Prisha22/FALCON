"""
Recompute the grid-refinement results from the SU2 histories in this folder.

    python grid_study.py

NACA 0012 (structured C-mesh, L1-L4): coefficients at the last iteration (all levels converged to rms[Rho] 1e-12),
differences relative to L4, and the grid convergence index (Celik et al. 2008) of C_d from L2-L4 with r = sqrt(2).
Double wedge (hybrid mesh, L0-L3): coefficients averaged over the last 500 iterations, differences relative to L3.
"""
import csv
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def read_history(path):
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    header = [c.strip().strip('"') for c in rows[0]]
    data = [[float(x) for x in r] for r in rows[1:]]
    return {name: [r[i] for r in data] for i, name in enumerate(header)}


def coefficients(path, last=1):
    h = read_history(path)
    return sum(h["CL"][-last:]) / last, sum(h["CD"][-last:]) / last


def gci(f1, f2, f3, r, fs=1.25):
    """f1 finest. Returns observed order, Richardson extrapolation, GCI_fine (%) with the observed order and with
    p = 2, or None when the three values do not converge monotonically."""
    e21, e32 = f2 - f1, f3 - f2
    if e21 == 0 or e32 / e21 <= 0:
        return None
    p = math.log(e32 / e21) / math.log(r)
    ext = f1 + (f1 - f2) / (r ** p - 1)
    return p, ext, 100 * fs * abs(e21 / f1) / (r ** p - 1), 100 * fs * abs(e21 / f1) / (r ** 2 - 1)


def table(title, levels):
    print(title)
    print(f"{'Grid':<5}{'Cells':>11}{'C_l':>10}{'C_d':>10}{'dC_l %':>9}{'dC_d %':>9}")
    cl_f, cd_f = levels[-1][2], levels[-1][3]
    for name, cells, cl, cd in levels:
        print(f"{name:<5}{cells:>11,}{cl:>10.5f}{cd:>10.5f}{100 * (cl / cl_f - 1):>+9.2f}{100 * (cd / cd_f - 1):>+9.2f}")


def naca():
    folder = os.path.join(HERE, "NACA0012_structured")
    cells = {1: 114935, 2: 229376, 3: 459378, 4: 917504}
    levels = [(f"L{L}", n, *coefficients(os.path.join(folder, f"history_n0012_L{L}.csv"))) for L, n in cells.items()]
    table("NACA 0012, M = 0.15, Re = 6e6, alpha = 10 deg (structured C-mesh)", levels)
    res = gci(levels[3][3], levels[2][3], levels[1][3], math.sqrt(2))
    if res:
        p, ext, g, g2 = res
        print(f"C_d, L2-L4: observed order {p:.2f}, Richardson extrapolation {ext:.5f}, "
              f"GCI {g:.1f}% ({g2:.1f}% with p = 2)")
    print()


def wedge():
    folder = os.path.join(HERE, "DoubleWedge_hybrid")
    cells = {0: 125435, 1: 261129, 2: 504089, 3: 1013510}
    for case, label in [("rans_a02", "RANS, alpha = 2 deg"), ("rans_a08", "RANS, alpha = 8 deg"),
                        ("rans_a14", "RANS, alpha = 14 deg"), ("euler_a06", "Euler, alpha = 6 deg")]:
        levels = [(f"L{L}", n, *coefficients(os.path.join(folder, case, f"history_wedge_L{L}.csv"), last=500))
                  for L, n in cells.items()]
        table(f"Double wedge, M = 1.86, Re = 0.66e6, {label} (hybrid mesh, mean of last 500 iterations)", levels)
        print()


if __name__ == "__main__":
    naca()
    wedge()
