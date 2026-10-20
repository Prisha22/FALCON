"""XFOIL driver: airfoil coordinate cleaning, argument checks and polar / Cp parsing (with a stand-in xfoil)."""
import os
import sys

import numpy as np
import pytest

from Scripts.Solver_and_Results import xfoil1
from .conftest import naca00xx


def write_dat(path, x, y, header="NACA 0012"):
    with open(path, "w") as f:
        f.write(header + "\n")
        for a, b in zip(x, y):
            f.write(f"{a:.6f} {b:.6f}\n")
    return str(path)


def read_clean(path):
    lines = open(path).read().splitlines()
    return lines[0], np.array([[float(v) for v in l.split()] for l in lines[1:]])


# ---------------------------------------------------------------- coordinate cleaning
def test_clean_geometry_gives_upper_surface_first_from_trailing_edge(tmp_path):
    x, y = naca00xx()
    src = write_dat(tmp_path / "in.dat", x, y)
    assert xfoil1.clean_airfoil_geometry(src, str(tmp_path / "out.dat"))
    header, c = read_clean(tmp_path / "out.dat")
    le = int(np.argmin(c[:, 0]))
    assert header == "Clean_Airfoil" and len(c) == len(x)
    assert c[0, 0] == pytest.approx(1.0, abs=1e-6) and c[-1, 0] == pytest.approx(1.0, abs=1e-6)
    assert c[le, 0] == pytest.approx(0.0, abs=1e-6)
    assert np.all(np.diff(c[:le + 1, 0]) <= 0) and np.all(np.diff(c[le:, 0]) >= 0)   # x monotonic on each side
    assert c[:le, 1].min() >= 0 >= c[le + 1:, 1].max()                                # upper first, lower second


def test_clean_geometry_reorders_a_file_listed_lower_surface_first(tmp_path):
    x, y = naca00xx()
    src = write_dat(tmp_path / "rev.dat", x[::-1], y[::-1])                           # lower -> LE -> upper
    assert xfoil1.clean_airfoil_geometry(src, str(tmp_path / "out.dat"))
    _, c = read_clean(tmp_path / "out.dat")
    le = int(np.argmin(c[:, 0]))
    assert c[:le, 1].mean() > 0 > c[le + 1:, 1].mean()


def test_clean_geometry_skips_text_header_lines_and_bad_rows(tmp_path):
    x, y = naca00xx()
    src = tmp_path / "messy.dat"
    body = "".join(f"{a:.6f} {b:.6f}\n" for a, b in zip(x, y))
    src.write_text("NACA 0012 AIRFOIL\nCoordinates in Selig format\n" + body + "end of file\n")
    assert xfoil1.clean_airfoil_geometry(str(src), str(tmp_path / "out.dat"))
    _, c = read_clean(tmp_path / "out.dat")
    assert len(c) == len(x)


def test_clean_geometry_removes_repeated_points(tmp_path):
    x, y = naca00xx()
    x2, y2 = np.insert(x, 50, x[50]), np.insert(y, 50, y[50])                         # one point listed twice
    src = write_dat(tmp_path / "dup.dat", x2, y2)
    assert xfoil1.clean_airfoil_geometry(src, str(tmp_path / "out.dat"))
    _, c = read_clean(tmp_path / "out.dat")
    assert len(c) == len(x)
    assert np.all(np.linalg.norm(np.diff(c, axis=0), axis=1) > 1e-6)


def test_clean_geometry_rejects_unusable_files(tmp_path):
    tiny = write_dat(tmp_path / "tiny.dat", [1, 0.5, 0], [0, 0.05, 0])
    assert xfoil1.clean_airfoil_geometry(tiny, str(tmp_path / "o.dat")) is False      # fewer than 5 points
    assert xfoil1.clean_airfoil_geometry(str(tmp_path / "missing.dat"), str(tmp_path / "o.dat")) is False


# ---------------------------------------------------------------- run_xfoil_logic
def test_supersonic_mach_is_refused(tmp_path, airfoil_dir):
    polar, cp = xfoil1.run_xfoil_logic("xfoil", str(tmp_path), os.path.join(airfoil_dir, "naca0012.dat"),
                                       1e6, 2, 0, 1, 1.2)
    assert polar == [] and cp == []


def test_missing_executable_gives_empty_results(tmp_path, airfoil_dir):
    polar, cp = xfoil1.run_xfoil_logic(str(tmp_path / "no_such_xfoil"), str(tmp_path),
                                       os.path.join(airfoil_dir, "naca0012.dat"), 1e6, 2, 0, 1, 0.0)
    assert polar == [] and cp == []


def fake_xfoil(tmp_path):
    """Stand-in xfoil: writes a polar file (header lines + 2 data rows) and cp files for the first 2 angles only."""
    script = tmp_path / "fake_xfoil.py"
    script.write_text(
        "polar = ['  XFOIL         Version 6.99', '', '   alpha     CL        CD       CDp       CM     Top_Xtr  Bot_Xtr',\n"
        "         '  ------ -------- --------- --------- -------- ------- -------',\n"
        "         '   0.000  0.0000   0.01000   0.00500  0.0000  0.4500  0.4500',\n"
        "         '   1.000  0.1100   0.01050   0.00550 -0.0010  0.4300  0.4700']\n"
        "open('polar_output.txt', 'w').write('\\n'.join(polar) + '\\n')\n"
        "for i in range(2):\n"
        "    open(f'cp_{i}.dat', 'w').write('x y cp\\n')\n")
    if os.name == "nt":
        exe = tmp_path / "fake_xfoil.bat"
        exe.write_text(f'@"{sys.executable}" "{script}"\n')
    else:
        exe = tmp_path / "fake_xfoil"
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}"\n')
        exe.chmod(0o755)
    return str(exe)


def test_polar_and_cp_files_are_parsed(tmp_path, airfoil_dir):
    run_dir = tmp_path / "run"
    polar, cp = xfoil1.run_xfoil_logic(fake_xfoil(tmp_path), str(run_dir), os.path.join(airfoil_dir, "naca0012.dat"),
                                       1e6, 2, 0, 1, 0.0)
    assert len(polar) == 2                                      # header and dashed lines are skipped
    assert polar[1][:3] == [1.0, 0.11, 0.0105]                  # alpha, CL, CD
    assert [a for a, _ in cp] == [0.0, 1.0]                     # cp_2.dat was never written, so alpha 2 is dropped
    assert (run_dir / "run_airfoil.dat").read_text().startswith("Clean_Airfoil")   # the cleaned geometry was used
    commands = (run_dir / "xfoil_input.txt").read_text().splitlines()
    assert "Visc 1000000.0" in commands or "Visc 1000000" in commands
    assert commands.count("ALFA 0.00") == 1 and "ALFA 2.00" in commands


def test_non_positive_step_falls_back_to_one_degree(tmp_path, airfoil_dir):
    run_dir = tmp_path / "run"
    xfoil1.run_xfoil_logic(fake_xfoil(tmp_path), str(run_dir), os.path.join(airfoil_dir, "naca0012.dat"),
                           1e6, 2, 0, 0, 0.0)
    assert (run_dir / "xfoil_input.txt").read_text().count("ALFA") == 3   # 0, 1, 2
