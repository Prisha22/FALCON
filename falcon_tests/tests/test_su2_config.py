"""prepare_su2_config: each regime gets its profile and the user's GUI values win over the defaults."""
import pytest

from Scripts.Solver_and_Results.su2_analyzer import (classify_regime, prepare_su2_config,
                                                     resolve_transition_model)


def build(tmp_path, regime, mach, re, gui=None, alpha=2.0):
    path = prepare_su2_config(str(tmp_path), dict(gui or {}), "airfoil.su2", regime, alpha, re, mach)
    cfg = {}
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("%") and "=" in line:
            k, v = line.split("=", 1)
            cfg[k.strip().upper()] = v.split("%")[0].strip()
    return cfg


def test_regime_classification():
    assert classify_regime("compressible", 0.1, 5e4)["low_re"]
    assert classify_regime("compressible", 0.8, 3e7)["transonic"]
    assert classify_regime("compressible", 1.86, 3e7)["supersonic"]
    assert not classify_regime("incompressible", 0.1, 5e4)["low_re"]
    assert classify_regime("compressible", 0.8, 3e7)["unsteady"]


@pytest.mark.parametrize("re, expected", [(5e4, "LM"), (1e5, "LM"), (1e6, "LM"), (2e6, "NONE")])
def test_lm_follows_reynolds_number(re, expected):
    assert resolve_transition_model({}, re, False, False) == expected


def test_lm_is_forced_inside_the_band():
    assert resolve_transition_model({"KIND_TRANS_MODEL": "NONE"}, 5e5, False, False) == "LM"


def test_low_re_profile(tmp_path):
    cfg = build(tmp_path, "compressible", 0.1, 5e4)
    assert cfg["SOLVER"] == "RANS"
    assert cfg["KIND_TURB_MODEL"] == "SST" and cfg["KIND_TRANS_MODEL"] == "LM"
    assert cfg["CONV_NUM_METHOD_FLOW"] == "LMROE" and cfg["LOW_MACH_PREC"] == "YES"
    assert cfg["TIME_DOMAIN"] == "YES"
    assert float(cfg["CFL_NUMBER"]) == 5.0 and int(cfg["TIME_ITER"]) == 10000


def test_high_re_incompressible_profile(tmp_path):
    cfg = build(tmp_path, "incompressible", 0.15, 5e5)
    assert cfg["SOLVER"] == "INC_RANS"
    assert cfg["KIND_TRANS_MODEL"] == "LM"
    assert float(cfg["CFL_NUMBER"]) == 10.0 and int(cfg["ITER"]) == 5000


def test_no_transition_model_above_1e6(tmp_path):
    cfg = build(tmp_path, "incompressible", 0.15, 2e6)
    assert cfg.get("KIND_TRANS_MODEL", "NONE") == "NONE"


def test_transonic_profile(tmp_path):
    cfg = build(tmp_path, "compressible", 0.8, 3e7)
    assert cfg["TIME_DOMAIN"] == "YES"
    assert float(cfg["CFL_NUMBER"]) == 1.0 and int(cfg["TIME_ITER"]) == 5000
    assert float(cfg["MACH_NUMBER"]) == pytest.approx(0.8)


def test_supersonic_profile(tmp_path):
    cfg = build(tmp_path, "compressible", 1.86, 3.5e7, alpha=1.0)
    assert cfg["CONV_NUM_METHOD_FLOW"] == "ROE"
    assert float(cfg["CFL_NUMBER"]) == 1.0 and int(cfg["ITER"]) == 5000
    assert float(cfg["AOA"]) == pytest.approx(1.0)


@pytest.mark.parametrize("regime, mach, re, iter_key", [
    ("incompressible", 0.15, 5e5, "ITER"), ("compressible", 0.1, 5e4, "TIME_ITER")])
def test_user_settings_override_defaults(tmp_path, regime, mach, re, iter_key):
    cfg = build(tmp_path, regime, mach, re, {"CFL_NUMBER": 3, "ITER": 777})
    assert float(cfg["CFL_NUMBER"]) == 3.0
    assert int(cfg[iter_key]) == 777


def test_case_values_are_written(tmp_path):
    cfg = build(tmp_path, "compressible", 0.1, 5e4, alpha=4.0)
    assert float(cfg["AOA"]) == pytest.approx(4.0)
    assert float(cfg["REYNOLDS_NUMBER"]) == pytest.approx(5e4)
    assert cfg["MESH_FILENAME"] == "airfoil.su2"
