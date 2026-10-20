"""Coordinate reading and normalization, CST/PARSEC reconstruction and spline re-paneling (Scripts/Geometry)."""
import numpy as np
import pytest

from Scripts.Geometry import read_airfoil
from Scripts.Geometry.cst import CST, AirfoilNormalizer
from Scripts.Geometry.interpolate import Interpolate
from Scripts.Geometry.parsec import Parsec
from Scripts.Batch import batch_runner as br

FILE = "naca0012.dat"


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


#reading
def test_coordinates_are_read_in_selig_order(airfoil_dir):
    x, y = read_airfoil.read_airfoil_coordinates(airfoil_dir, FILE)
    le = int(np.argmin(x))
    assert len(x) == len(y) == 199
    assert x[0] == pytest.approx(1.0, abs=1e-6) and x[-1] == pytest.approx(1.0, abs=1e-6)
    assert x[le] == pytest.approx(0.0, abs=1e-6) and le == 99     # leading edge in the middle of the loop
    assert y[:le].min() >= 0 >= y[le + 1:].max()                   # upper surface first, lower second


def test_header_blank_lines_and_extra_spaces_are_ignored(tmp_path):
    (tmp_path / "dirty.dat").write_text("Some name 12%\n\n1.0 0.0\n  0.5   0.06 \n\n0.0 0.0\n0.5 -0.06\n1.0 0.0\n")
    x, y = read_airfoil.read_airfoil_coordinates(str(tmp_path), "dirty.dat")
    assert len(x) == 5 and y[1] == pytest.approx(0.06)


def test_duplicate_x_values_are_removed_keeping_y_pairs(airfoil_dir):
    ip = Interpolate(airfoil_dir, FILE)
    x, y = ip.remove_duplicates([0.0, 0.5, 0.5, 1.0], [0.0, 0.1, 0.1, 0.0])
    assert list(x) == [0.0, 0.5, 1.0] and list(y) == [0.0, 0.1, 0.0]


#normalization
def test_normalizer_puts_le_at_origin_and_chord_at_one():
    t = np.linspace(0, np.pi, 50)
    xs = np.concatenate([0.5 * (1 + np.cos(t)), 0.5 * (1 - np.cos(t))[1:]])
    ys = np.concatenate([0.06 * np.sin(t), -0.06 * np.sin(t)[1:]])
    # the same section scaled to 2.5 chord, rotated 10 degrees and shifted
    th = np.deg2rad(10)
    X = 2.5 * (xs * np.cos(th) - ys * np.sin(th)) + 3.0
    Y = 2.5 * (xs * np.sin(th) + ys * np.cos(th)) - 1.0
    norm = AirfoilNormalizer()
    xn, yn = norm.normalize(X, Y)
    assert xn.min() == pytest.approx(0.0, abs=1e-2) and xn.max() == pytest.approx(1.0, abs=1e-2)
    assert norm.scale_factor == pytest.approx(2.5, rel=1e-2)
    xb, yb = norm.denormalize(xn, yn)                              
    assert np.allclose(xb, X) and np.allclose(yb, Y)



def test_cst_reconstructs_naca0012(airfoil_dir):
    np.random.seed(0)  # CST starts its fit from random guesses
    _, y_in = read_airfoil.read_airfoil_coordinates(airfoil_dir, FILE)
    foil = CST(airfoil_dir, FILE).foil()
    assert foil.shape == y_in.shape
    assert rmse(foil, y_in) < 5e-4                                  # measured about 9e-5


def test_parsec_reconstructs_naca0012(airfoil_dir):
    _, y_in = read_airfoil.read_airfoil_coordinates(airfoil_dir, FILE)
    foil = Parsec(airfoil_dir, FILE).foil()
    assert foil.shape == y_in.shape
    assert rmse(foil, y_in) < 5e-4                                  # measured about 7e-5


def test_fits_keep_a_symmetric_section_symmetric(airfoil_dir):
    np.random.seed(0)
    for fit in (CST(airfoil_dir, FILE).foil(), Parsec(airfoil_dir, FILE).foil()):
        le = len(fit) // 2
        assert fit[:le].max() == pytest.approx(0.06, abs=3e-3)
        assert fit[le:].min() == pytest.approx(-0.06, abs=3e-3)


#re-paneling
def test_spline_repaneling(airfoil_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # airfoil_interpolate changes the working directory itself; restored afterwards
    n = 120
    ip = Interpolate(airfoil_dir, FILE)
    x, y = ip.airfoil_interpolate(n, "CST", None, airfoil_dir, FILE, str(tmp_path))
    upper, lower = ip.get_surface()
    assert len(upper) == len(lower) == n
    assert len(x) == 2 * n                                          # upper + lower[1:] + closing point
    assert (x[0], y[0]) == (x[-1], y[-1])                           # loop is closed
    assert upper[0] == lower[-1] and upper[-1] == lower[0]          # shared trailing and leading edge
    assert min(x) == pytest.approx(0.0, abs=1e-6) and max(x) == pytest.approx(1.0, abs=1e-3)
    assert max(y) == pytest.approx(0.06, abs=3e-3) and min(y) == pytest.approx(-0.06, abs=3e-3)
    assert (tmp_path / "output.dat").exists()


def test_prepare_geometry_without_repaneling(airfoil_dir, tmp_path):
    case = br.Case(FILE, "su2", 5e4, 0.1, 0, 0, 0, repanel=False)
    x, y, upper, lower, info = br.prepare_geometry(case, airfoil_dir, str(tmp_path))
    assert info["method"].startswith("none") and info["n_coordinates"] == 199
    assert upper[-1] == lower[0]                                    # surfaces meet at the leading edge
    assert (tmp_path / "output.dat").exists()
