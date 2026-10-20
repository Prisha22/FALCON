import math
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")  # no display in CI


def _is_repo_root(path):
    return os.path.isfile(os.path.join(path, "Scripts", "Geometry", "read_airfoil.py"))


def _find_repo_root():
    """The folder that contains Scripts/, found from this file upwards (and one level down), then from the cwd."""
    for start in (os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
        folder = start
        while True:
            candidates = [folder]
            try:
                candidates += [os.path.join(folder, n) for n in sorted(os.listdir(folder))
                               if os.path.isdir(os.path.join(folder, n))]
            except OSError:
                pass
            for c in candidates:
                if _is_repo_root(c):
                    return c
            parent = os.path.dirname(folder)
            if parent == folder:
                break
            folder = parent
    return None


REPO_ROOT = _find_repo_root()
if REPO_ROOT is None:
    pytest.exit("Could not find the FALCON repository (a folder containing Scripts/Geometry/read_airfoil.py) "
                "above or next to the tests. Put the tests inside your FALCON folder.", returncode=2)
if sys.path[0] != REPO_ROOT:
    sys.path.insert(0, REPO_ROOT)
_cached = sys.modules.get("Scripts")
if _cached is not None and getattr(_cached, "__file__", None) is None:   # a wrong, namespace-only "Scripts"
    for _name in [n for n in sys.modules if n == "Scripts" or n.startswith("Scripts.")]:
        del sys.modules[_name]


def naca00xx(thickness=0.12, n=100):
    """Selig-ordered NACA 00xx coordinates: trailing edge -> leading edge (upper) -> trailing edge (lower)."""
    beta = np.linspace(0.0, math.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))  # cosine spacing, 0 -> 1
    yt = 5 * thickness * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    xs = np.concatenate([x[::-1], x[1:]])
    ys = np.concatenate([yt[::-1], -yt[1:]])
    return xs, ys


@pytest.fixture
def airfoil_dir(tmp_path):
    """A directory holding naca0012.dat in Selig format."""
    d = tmp_path / "airfoils"
    d.mkdir()
    x, y = naca00xx()
    with open(d / "naca0012.dat", "w") as f:
        f.write("NACA 0012\n")
        for a, b in zip(x, y):
            f.write(f"{a:.6f} {b:.6f}\n")
    return str(d)
