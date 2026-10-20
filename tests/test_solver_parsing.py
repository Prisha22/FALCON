"""SU2 history parsing: coefficients, convergence state, missing or broken files."""
import math
import os
import sys

import pytest

from Scripts.Batch.batch_runner import read_history
from Scripts.Solver_and_Results.su2_analyzer import extract_su2_polar_data

HEADER = '"Inner_Iter","rms[Rho]","CL","CD","CMz"\n'


def write_history(path, rows):
    with open(path, "w") as f:
        f.write(HEADER)
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
    return str(path)


def test_polar_takes_last_row(tmp_path):
    h = write_history(tmp_path / "h.csv", [(0, -3.0, 0.10, 0.050, -0.01), (1, -8.5, 0.55, 0.012, -0.05)])
    aoa, cl, cd, cm = extract_su2_polar_data([(2.0, h, None, None)])
    assert aoa == [2.0] and cl == [0.55] and cd == [0.012] and cm == [-0.05]


def test_polar_missing_file_gives_nan(tmp_path):
    _, cl, cd, cm = extract_su2_polar_data([(0.0, str(tmp_path / "nope.csv"), None, None)])
    assert math.isnan(cl[0]) and math.isnan(cd[0]) and math.isnan(cm[0])


SETTINGS = {"CONV_FIELD": "RMS_DENSITY", "CONV_RESIDUAL_MINVAL": "-8", "ITER": "5000"}


def test_read_history_converged(tmp_path):
    h = write_history(tmp_path / "h.csv", [(0, -3.0, 0.1, 0.05, 0.0), (99, -8.2, 0.5, 0.012, -0.05)])
    out = read_history(h, SETTINGS)
    assert out["state"] == "converged" and out["converged"]
    assert out["cl"] == pytest.approx(0.5) and out["iterations"] == 100


def test_read_history_max_iterations(tmp_path):
    h = write_history(tmp_path / "h.csv", [(4999, -5.0, 0.5, 0.012, -0.05)])
    assert read_history(h, SETTINGS)["state"] == "max_iterations"


def test_read_history_stopped_early(tmp_path):
    h = write_history(tmp_path / "h.csv", [(10, -5.0, 0.5, 0.012, -0.05)])
    assert read_history(h, SETTINGS)["state"] == "stopped"


def test_read_history_diverged(tmp_path):
    h = write_history(tmp_path / "h.csv", [(10, "nan", "nan", "nan", "nan")])
    assert read_history(h, SETTINGS)["state"] == "diverged"


def test_read_history_missing_and_empty(tmp_path):
    assert read_history(str(tmp_path / "none.csv"), SETTINGS)["state"] == "no_history"
    h = write_history(tmp_path / "e.csv", [])
    assert read_history(h, SETTINGS)["state"] == "empty_history"


from Scripts.Solver_and_Results.su2_analyzer import find_su2_output, su2_output_candidates  


def test_candidates_cover_both_spellings(tmp_path):
    names = [os.path.basename(p) for p in su2_output_candidates(str(tmp_path), "history_aoa_2.00", ".csv")]
    assert names == ["history_aoa_2.00.csv", "history_aoa_2.csv"]


@pytest.mark.parametrize("written", ["history_aoa_2.csv", "history_aoa_2.00.csv"])
def test_history_is_found_whichever_spelling_su2_wrote(tmp_path, written):
    (tmp_path / written).write_text("x")
    found = find_su2_output(str(tmp_path), "history_aoa_2.00", ".csv")
    assert found is not None and os.path.basename(found) == written


def test_missing_output_gives_none(tmp_path):
    assert find_su2_output(str(tmp_path), "history_aoa_2.00", ".csv") is None


def fake_su2(tmp_path, written_prefix):
    """A stand-in SU2_CFD that writes its outputs under `written_prefix` (e.g. 'aoa_2' as SU2 8 does)."""
    script = tmp_path / "fake_su2.py"
    script.write_text(
        "import sys\n"
        f"open('history_{written_prefix}.csv', 'w').write('\"Inner_Iter\",\"CL\",\"CD\",\"CMz\"\\n9,0.5,0.012,-0.05\\n')\n"
        f"open('surf_{written_prefix}.csv', 'w').write('x')\n"
        f"open('flow_{written_prefix}.vtu', 'w').write('x')\n")
    if os.name == "nt":
        exe = tmp_path / "fake_su2.bat"
        exe.write_text(f'@"{sys.executable}" "{script}" %*\n')
    else:
        exe = tmp_path / "fake_su2"
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        exe.chmod(0o755)
    return str(exe)


@pytest.mark.parametrize("prefix", ["aoa_2", "aoa_2.00"])   
def test_run_analysis_finds_outputs_whichever_spelling_su2_wrote(tmp_path, prefix):
    from Scripts.Solver_and_Results.su2_analyzer import SU2Runner
    run_dir = tmp_path / "AoA_2.00"
    run_dir.mkdir()
    cfg = run_dir / "config_aoa_2.00.cfg"
    cfg.write_text("")
    runner = SU2Runner(su2_cfd_path=fake_su2(tmp_path, prefix), use_mpi=False, num_procs=1)
    status, history, volume, surface = runner.run_analysis(str(cfg), 2.0, str(run_dir), None, None)
    assert status == "COMPLETED"
    assert history and os.path.exists(history) and surface and os.path.exists(surface)
    assert volume and os.path.exists(volume)
    out = read_history(history, {"CONV_FIELD": "RMS_DENSITY", "ITER": "5000"})
    assert out["cl"] == pytest.approx(0.5) and out["cd"] == pytest.approx(0.012)
