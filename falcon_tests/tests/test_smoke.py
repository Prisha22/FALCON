"""End-to-end smoke tests. The dry run needs no solver; the real runs are skipped when the solver is missing."""
import json
import os
import shutil

import pytest

from Scripts.Batch import batch_runner as br


def test_dry_run_uses_the_users_settings(tmp_path, airfoil_dir):
    """Mesh + one config per angle, and the GUI's Solver Setup values reach the written SU2 configuration."""
    pytest.importorskip("Scripts.Meshing.meshing")
    case = br.Case("naca0012.dat", "su2", 5e5, 0.1, 0, 2, 2, repanel=False)
    settings = {"regime": "Incompressible", "settings": {"CFL_NUMBER": "3", "ITER": "123", "EXT_ITER": "123"}}
    br.run_batch([case], str(tmp_path / "out"), airfoil_dir, dry_run=True, su2_settings=settings, log=lambda m: None)
    case_dir = tmp_path / "out" / case.case_id
    status = json.load(open(case_dir / "status.json"))
    assert status["state"] == "dry_run", status.get("error")
    assert status["template"] == "GUI Solver Setup tab"
    assert (case_dir / "airfoil.su2").exists()
    for alpha in ("0.00", "2.00"):
        cfg = br._read_cfg(str(case_dir / f"AoA_{alpha}" / f"config_aoa_{alpha}.cfg"))
        assert float(cfg["CFL_NUMBER"]) == 3.0 and int(cfg["ITER"]) == 123   # not the defaults


@pytest.mark.skipif(not os.path.exists(os.path.join(br.TEMPLATE_DIR, "LowReIncomp.cfg")),
                    reason="Configuration_Files templates not present")
def test_dry_run_with_automatic_settings(tmp_path, airfoil_dir):
    pytest.importorskip("Scripts.Meshing.meshing")
    case = br.Case("naca0012.dat", "su2", 5e4, 0.1, 0, 0, 0, repanel=False)
    br.run_batch([case], str(tmp_path / "out"), airfoil_dir, dry_run=True, log=lambda m: None)
    status = json.load(open(tmp_path / "out" / case.case_id / "status.json"))
    assert status["state"] == "dry_run", status.get("error")
    assert status["template"].endswith(".cfg")


def tail(path, n=30):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return f"(could not read {path})"


def su2_diagnostics(case_dir, point):
    """What to look at when a real SU2 run gives no coefficients: the point record and SU2's own last output."""
    status = {}
    try:
        status = json.load(open(os.path.join(case_dir, "status.json")))
    except (OSError, ValueError):
        pass
    run_dir = os.path.join(case_dir, "AoA_2.00")
    return (f"\nSU2 gave no coefficients.\n  point: {point}\n  case state: {status.get('state')}  "
            f"error: {(status.get('error') or '')[-400:]}\n  folder: {case_dir}\n"
            f"--- last lines of log.txt (SU2's screen output) ---\n{tail(os.path.join(case_dir, 'log.txt'))}\n"
            f"--- files in AoA_2.00 ---\n{sorted(os.listdir(run_dir)) if os.path.isdir(run_dir) else 'missing'}")


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("SU2_CFD") is None, reason="SU2_CFD not installed")
def test_su2_short_run_produces_finite_coefficients(tmp_path, airfoil_dir):
    """A real SU2 run for a handful of iterations. It checks that SU2 starts, writes a history and gives
    plausible numbers - not that the flow has converged. Meshing a 400k-cell grid dominates the time."""
    pytest.importorskip("Scripts.Meshing.meshing")
    case = br.Case("naca0012.dat", "su2", 5e5, 0.1, 2, 2, 0, repanel=False, y_plus=30.0)  # coarser wall cell
    settings = {"regime": "Incompressible", "settings": {"ITER": "5", "EXT_ITER": "5"}}
    br.run_batch([case], str(tmp_path / "out"), airfoil_dir, cores=1, su2_settings=settings,
                 su2_timeout=1200.0, log=lambda m: None)          # a hung SU2 stops after 20 minutes
    case_dir = str(tmp_path / "out" / case.case_id)
    point = json.load(open(os.path.join(case_dir, "status.json")))["points"][0]
    if point.get("cl") is None or point.get("cd") is None:
        pytest.fail(su2_diagnostics(case_dir, point), pytrace=False)
    assert 0 < point["cd"] < 1 and abs(point["cl"]) < 5          # physically plausible after a few iterations


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("xfoil") is None, reason="xfoil not installed")
def test_xfoil_naca0012_is_symmetric(tmp_path, airfoil_dir):
    case = br.Case("naca0012.dat", "xfoil", 1e6, 0.0, -2, 2, 2, repanel=False)
    br.run_batch([case], str(tmp_path / "out"), airfoil_dir, xfoil_path=shutil.which("xfoil"), log=lambda m: None)
    pts = {p["alpha"]: p for p in json.load(open(tmp_path / "out" / case.case_id / "status.json"))["points"]}
    assert pts[0.0]["converged"] and abs(pts[0.0]["cl"]) < 0.02          # symmetric section: CL(0) ~ 0
    assert pts[2.0]["cl"] == pytest.approx(-pts[-2.0]["cl"], abs=0.03)   # CL odd in alpha
