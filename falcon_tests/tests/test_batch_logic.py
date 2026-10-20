"""Case list, per-case configuration files and the skip/re-run rule of the batch runner."""
import json
import os

import pytest

from Scripts.Batch import batch_runner as br


def test_alpha_ranges():
    assert br.Case("a.dat", "su2", 5e4, 0.1, 0, 4, 2).alphas() == [0.0, 2.0, 4.0]
    assert br.Case("a.dat", "su2", 5e4, 0.1, 3, 3, 0).alphas() == [3.0]
    assert br.Case("a.dat", "xfoil", 5e4, 0.1, -2, 2, 1).alphas() == [-2.0, -1.0, 0.0, 1.0, 2.0]


def test_case_id_is_filesystem_safe():
    cid = br.Case("My Foil (v2).dat", "su2", 5e4, 0.1, 0, 4, 2, mesh="hybrid", repanel=False).case_id
    assert " " not in cid and "(" not in cid and cid.endswith("_hybrid_raw")


def write_csv(path, rows, header="airfoil,solver,re,mach,alpha_min,alpha_max,alpha_step"):
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return str(path)


def test_load_cases(tmp_path):
    cases = br.load_cases(write_csv(tmp_path / "c.csv", ["naca0012.dat,SU2,5e4,0.1,0,4,2"]))
    assert len(cases) == 1 and cases[0].solver == "su2" and cases[0].reynolds == 5e4


@pytest.mark.parametrize("rows, header, message", [
    (["a.dat,fluent,5e4,0.1,0,4,2"], None, "solver must be"),
    (["a.dat,su2,5e4,0.1,0,4,2", "a.dat,su2,5e4,0.1,0,4,2"], None, "duplicate"),
    (["a.dat,su2,5e4,0.1,0,4"], "airfoil,solver,re,mach,alpha_min,alpha_max", "missing column"),
])
def test_load_cases_rejects_bad_input(tmp_path, rows, header, message):
    path = write_csv(tmp_path / "c.csv", rows, *([header] if header else []))
    with pytest.raises(ValueError, match=message):
        br.load_cases(path)


def test_split_surfaces_shares_leading_edge():
    x = [1.0, 0.5, 0.0, 0.5, 1.0]
    y = [0.0, 0.1, 0.0, -0.1, 0.0]
    upper, lower = br.split_surfaces(x, y)
    assert upper[-1] == lower[0] == [0.0, 0.0] and len(upper) == 3 and len(lower) == 3


def test_config_from_file_keeps_numerics_and_rewrites_case_lines(tmp_path):
    src = tmp_path / "user.cfg"
    src.write_text("SOLVER= RANS\nMACH_NUMBER= 0.3\nAOA= 9\nCFL_NUMBER= 7\nITER= 9000\nOUTPUT_FILES= (RESTART)\n")
    path, settings = br.write_config_from_file(str(src), str(tmp_path), 2.0, 5e4, 0.1, max_iter=1000)
    cfg = br._read_cfg(path)
    assert cfg["CFL_NUMBER"] == "7" and cfg["SOLVER"] == "RANS"            # numerics untouched
    assert float(cfg["AOA"]) == 2.0 and float(cfg["MACH_NUMBER"]) == 0.1   # case lines rewritten
    assert cfg["ITER"] == "1000" and settings["ITER"] == "1000"            # capped
    assert "SURFACE_PARAVIEW" in cfg["OUTPUT_FILES"]


def test_settings_fingerprint_distinguishes_settings():
    a = br.settings_fingerprint(su2_settings={"regime": "Incompressible", "settings": {"CFL_NUMBER": "3"}})
    b = br.settings_fingerprint(su2_settings={"regime": "Incompressible", "settings": {"CFL_NUMBER": "7"}})
    assert a != b != br.settings_fingerprint()


@pytest.fixture
def finished_case(tmp_path):
    case = br.Case("naca0012.dat", "su2", 5e5, 0.1, 0, 4, 2)

    def mark(settings_id):
        d = tmp_path / case.case_id
        d.mkdir(exist_ok=True)
        (d / "status.json").write_text(json.dumps({"state": "converged", "settings_id": settings_id}))
    return case, mark


@pytest.mark.parametrize("recorded, wanted, runs", [
    ("same", "same", False),      # same settings, skipped
    ("old", "new", True),         # changed settings, re-run
    (None, None, False),          # legacy status, automatic run, skipped
    (None, "user", True),         # legacy status, user settings now, re-run
])
def test_rerun_only_when_settings_changed(tmp_path, monkeypatch, finished_case, recorded, wanted, runs):
    case, mark = finished_case
    started = []
    monkeypatch.setattr(br, "_run_pool", lambda cases, *a, **k: started.extend(cases))
    monkeypatch.setattr(br, "write_summary", lambda *a, **k: {})

    def settings(tag):
        return None if tag is None else {"regime": "Incompressible", "settings": {"CFL_NUMBER": tag}}
    mark(None if recorded is None else br.settings_fingerprint(su2_settings=settings(recorded)))
    br.run_batch([case], str(tmp_path), ".", su2_settings=settings(wanted), log=lambda m: None)
    assert bool(started) is runs
