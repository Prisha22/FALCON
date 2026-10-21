import csv
import hashlib
import json
import math
import multiprocessing
import os
import re as regex
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEMPLATE_DIR = os.path.join(REPO_ROOT, "Configuration_Files")

SOLVERS = ("xfoil", "su2")
REQUIRED_COLUMNS = ("airfoil", "solver", "re", "mach", "alpha_min", "alpha_max", "alpha_step")
FINAL_STATES = ("converged", "partial", "failed", "timeout", "crashed")  # "stopped" is not final: a re-run resumes it
RETRY_STATES = ("failed", "timeout", "crashed")

# History column holding the residual named by CONV_FIELD.
RESIDUAL_COLUMNS = {"RMS_PRESSURE": "rms[P]", "RMS_DENSITY": "rms[Rho]"}


@dataclass
class Case:
    airfoil: str
    solver: str
    reynolds: float
    mach: float
    alpha_min: float
    alpha_max: float
    alpha_step: float
    n_points: int = 200
    y_plus: float = 1.0
    mesh: str = "structured"  # SU2 only: "structured" (Scripts/Meshing/meshing.py) or "hybrid" (hybrid.py)
    repanel: bool = True      # False: the airfoil file's own coordinates, no fit or re-paneling

    @property
    def case_id(self):
        stem = os.path.splitext(os.path.basename(self.airfoil))[0]
        name = (f"{self.solver}_{stem}_Re{self.reynolds:g}_M{self.mach:g}"
                f"_a{self.alpha_min:g}_{self.alpha_max:g}_{self.alpha_step:g}")
        # Non-default choices extend the name, so earlier batch folders keep resolving to the same cases.
        if self.solver == "su2" and self.mesh != "structured":
            name += f"_{self.mesh}"
        if not self.repanel:
            name += "_raw"
        return regex.sub(r"[^A-Za-z0-9_.+-]", "_", name)

    def alphas(self):
        """The angles the solver is asked for, generated the way each FALCON runner generates them."""
        if self.solver == "xfoil":
            step = self.alpha_step if self.alpha_step > 0 else 1.0
            return [float(a) for a in np.arange(self.alpha_min, self.alpha_max + 0.0001, step)]
        if self.alpha_step > 0:
            return [float(a) for a in np.arange(self.alpha_min, self.alpha_max + self.alpha_step / 2.0,
                                                self.alpha_step)]
        return [float(self.alpha_min)]


def load_cases(path):
    """Read a case list: one row per case, columns airfoil, solver, re, mach, alpha_min, alpha_max, alpha_step,
    and optionally n_points (re-paneled coordinates per surface, default 200), y_plus (target, default 1), mesh
    (structured or hybrid, default structured) and repanel (yes/no, default yes)."""
    cases = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing column(s) {', '.join(missing)}")
        for row in reader:
            try:
                solver = row["solver"].strip().lower()
                if solver not in SOLVERS:
                    raise ValueError(f"solver must be one of {', '.join(SOLVERS)}, got {row['solver']!r}")
                airfoil = row["airfoil"].strip()
                if not airfoil:
                    raise ValueError("empty airfoil name")
                cases.append(Case(
                    airfoil=airfoil, solver=solver,
                    reynolds=float(row["re"]), mach=float(row["mach"]),
                    alpha_min=float(row["alpha_min"]), alpha_max=float(row["alpha_max"]),
                    alpha_step=float(row["alpha_step"]),
                    n_points=int(float(row.get("n_points") or 200)),
                    y_plus=float(row.get("y_plus") or 1.0),
                    mesh=_mesh_type(row.get("mesh")),
                    repanel=_yes_no(row.get("repanel"), default=True)))
            except (TypeError, ValueError) as e:
                raise ValueError(f"{path}, line {reader.line_num}: {e}") from None
    ids = [c.case_id for c in cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"{path}: duplicate case(s) {', '.join(duplicates)}")
    return cases


MESH_TYPES = ("structured", "hybrid")


def _mesh_type(value):
    mesh = (value or "structured").strip().lower()
    if mesh not in MESH_TYPES:
        raise ValueError(f"mesh must be one of {', '.join(MESH_TYPES)}, got {value!r}")
    return mesh


def _yes_no(value, default):
    if value is None or str(value).strip() == "":
        return default
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "y"):
        return True
    if v in ("0", "false", "no", "n"):
        return False
    raise ValueError(f"expected yes or no, got {value!r}")


def _read_template(name):
    """Key/value pairs of a Configuration_Files template, parsed like FalconApp.parse_su2_cfg."""
    return _read_cfg(os.path.join(TEMPLATE_DIR, name))


# Lines of a user-supplied SU2 configuration that describe the case rather than the numerics; they are rewritten for
# every case and angle, everything else is kept verbatim.
CASE_LINES = ("MACH_NUMBER", "AOA", "REYNOLDS_NUMBER", "MESH_FILENAME", "CONV_FILENAME", "VOLUME_FILENAME",
              "SURFACE_FILENAME", "RESTART_FILENAME", "OUTPUT_FILES")


def write_config_from_file(source, run_dir, alpha, reynolds, mach, max_iter=None):
    """
    Write an SU2 configuration for one angle from a user-supplied configuration: the numerical settings are copied
    verbatim and only the case lines (flow conditions, file names, output files) are replaced. max_iter, if given,
    caps ITER. Returns the written path and the parsed settings that apply to the run.
    """
    lines = open(source).read().splitlines()
    values = {
        "MACH_NUMBER": f"{mach:.6f}", "AOA": f"{alpha:.6f}", "REYNOLDS_NUMBER": f"{reynolds:.0f}",
        "MESH_FILENAME": "airfoil.su2", "CONV_FILENAME": f"history_aoa_{alpha:.2f}",
        "VOLUME_FILENAME": f"flow_aoa_{alpha:.2f}", "SURFACE_FILENAME": f"surf_aoa_{alpha:.2f}",
        "RESTART_FILENAME": "restart_flow.dat",
        # The surface VTU carries the achieved y+ (see prepare_su2_config).
        "OUTPUT_FILES": "(RESTART, TECPLOT_ASCII, PARAVIEW, SURFACE_CSV, SURFACE_PARAVIEW)",
    }
    settings = _read_cfg(source)
    if max_iter and int(float(settings.get("ITER", max_iter))) > max_iter:
        values["ITER"] = str(int(max_iter))
    out, seen = [], set()
    for line in lines:
        m = regex.match(r"^\s*([A-Za-z0-9_]+)\s*=", line)
        key = m.group(1).upper() if m else None
        if key in values:
            out.append(f"{key}= {values[key]}")
            seen.add(key)
        else:
            out.append(line)
    out += [f"{k}= {v}" for k, v in values.items() if k not in seen]
    path = os.path.join(run_dir, f"config_aoa_{alpha:.2f}.cfg")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    settings.update({k: v for k, v in values.items() if k == "ITER"})
    return path, settings


def _read_cfg(path):
    """Key/value pairs of an SU2 configuration file, parsed like FalconApp.parse_su2_cfg."""
    settings = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            m = regex.match(r"^([A-Za-z0-9_]+)\s*=\s*(.*)", line)
            if m:
                settings[m.group(1).upper()] = m.group(2).split("%")[0].strip()
    return settings


def gui_recommended_settings(mach, reynolds):
    """
    The flow regime, template and settings the GUI hands to prepare_su2_config after 'Load Recommended Settings'.

    Mirrors FalconApp.load_recommended_settings, apply_loaded_settings and collect_su2_settings: the template is
    picked from the Mach and Reynolds numbers; each template value that has a widget is applied when the widget
    offers it, the iteration limit and residual go to the convergence fields, and the remaining numerical values
    pass straight through. Settings the template does not name keep their GUI default (the first entry of each
    drop-down, CFL 1.0, residual -8, 5000 iterations). tests/test_batch_runner.py checks this against the GUI.
    """
    from Scripts.Solver_and_Results.su2_analyzer import (SU2_COMPRESSIBLE_SETTINGS, SU2_INCOMPRESSIBLE_SETTINGS,
                                                         template_passthrough)

    if mach >= 1.0:
        template, regime = "supersonic.cfg", "Compressible"
    elif 0.7 <= mach < 1.0:
        template, regime = "Transonic.cfg", "Compressible"
    else:
        regime = "Incompressible"
        template = "HighReIncomp.cfg" if reynolds >= 1e6 else "LowReIncomp.cfg"

    options = SU2_INCOMPRESSIBLE_SETTINGS if regime == "Incompressible" else SU2_COMPRESSIBLE_SETTINGS
    conv_fields = ["RMS_PRESSURE"] if regime == "Incompressible" else ["RMS_DENSITY", "REL_RMS_DENSITY", "RMS_ENERGY"]
    loaded = _read_template(template)

    # Widget defaults.
    settings = {}
    for key, choices in options.items():
        if key in ("ITER", "EXT_ITER", "CONV_RESIDUAL_MINVAL"):
            continue
        if isinstance(choices, list):
            settings[key] = choices[0]
        elif isinstance(choices, dict) and key == "CONV_NUM_METHOD_FLOW":
            settings[key] = next(iter(choices.values()))[0]
    settings["CFL_NUMBER"] = "1.0"
    settings["CONV_FIELD"] = conv_fields[0]
    residual, iterations = "-8.0", "5000"

    # Template values applied to the widgets that offer them.
    used = set()
    for key in list(settings):
        if key not in loaded:
            continue
        used.add(key)
        value = loaded[key]
        if key == "CONV_NUM_METHOD_FLOW":
            allowed = [s for schemes in options["CONV_NUM_METHOD_FLOW"].values() for s in schemes]
        elif key == "CONV_FIELD":
            allowed = conv_fields
        elif key == "CFL_NUMBER":
            allowed = None
        else:
            allowed = options[key]
        if allowed is None or value in allowed:
            settings[key] = value
    iterations = loaded.get("ITER") or loaded.get("TIME_ITER") or iterations
    residual = loaded.get("CONV_RESIDUAL_MINVAL", residual)
    used.update({"ITER", "TIME_ITER", "EXT_ITER", "CONV_RESIDUAL_MINVAL"})

    merged = template_passthrough(loaded, used)
    merged.update(settings)
    merged.update(CONV_RESIDUAL_MINVAL=residual, ITER=iterations, EXT_ITER=iterations)
    return regime, template, merged


def split_surfaces(x, y):
    """Upper (trailing edge -> leading edge) and lower (leading edge -> trailing edge) surfaces of a Selig-ordered
    coordinate loop, sharing the leading-edge point, as lists of [x, y] like Interpolate.get_surface."""
    le = int(np.argmin(x))
    upper = [[float(a), float(b)] for a, b in zip(x[:le + 1], y[:le + 1])]
    lower = [[float(a), float(b)] for a, b in zip(x[le:], y[le:])]
    return upper, lower


def prepare_geometry(case, airfoil_dir, case_dir):
    """
    With case.repanel: CST and PARSEC fits, the lower-RMSE choice and the spline re-paneling to case.n_points per
    surface, as the GUI's Analyze button does. Without it: the airfoil file's own coordinates. Writes output.dat into
    case_dir and returns its coordinates, the upper and lower surfaces (for the hybrid mesher) and a record.
    """
    from Scripts.Geometry import read_airfoil

    name = case.airfoil
    if not case.repanel:
        x, y = read_airfoil.read_airfoil_coordinates(airfoil_dir, name)
        with open(os.path.join(case_dir, "output.dat"), "w") as f:
            f.write(f"{os.path.splitext(name)[0]} (file coordinates, not re-paneled)\n")
            for a, b in zip(x, y):
                f.write(f"{a:.8f} {b:.8f}\n")
        upper, lower = split_surfaces(x, y)
        return x, y, upper, lower, {"method": "none (file coordinates)", "n_coordinates": int(len(x))}

    from main import error  # the CST/PARSEC selection rule lives in the GUI module
    from Scripts.Geometry.cst import CST
    from Scripts.Geometry.interpolate import Interpolate
    from Scripts.Geometry.parsec import Parsec

    _, y_input = read_airfoil.read_airfoil_coordinates(airfoil_dir, name)
    foil_parsec = Parsec(airfoil_dir, name).foil()
    foil_cst = CST(airfoil_dir, name).foil()
    method, rmse_parsec, rmse_cst = error(foil_parsec, foil_cst, y_input)
    interp = Interpolate(airfoil_dir, name)
    interp.airfoil_interpolate(
        case.n_points, method, foil_parsec if method == "PARSEC" else foil_cst, airfoil_dir, name, case_dir)
    upper, lower = interp.get_surface()
    x, y = read_airfoil.read_airfoil_coordinates(case_dir, "output.dat")
    return x, y, upper, lower, {"method": method, "rmse_parsec": float(rmse_parsec), "rmse_cst": float(rmse_cst),
                                "n_coordinates": int(len(x))}


def _read_polar(path):
    rows = []
    if os.path.exists(path):
        for line in open(path, errors="replace"):
            try:
                nums = [float(v) for v in line.split()]
            except ValueError:
                continue
            if len(nums) >= 5:
                rows.append(nums)
    return rows


def run_xfoil_sweep(xfoil_path, work_dir, airfoil_file, reynolds, mach, alphas, n_iter=100, angle_timeout=300.0,
                    setup_timeout=60.0):
    """
    FALCON's XFOIL session (PLOP G, LOAD, PANE, OPER, Visc, Mach, ITER, PACC, then ALFA + CPWR per angle, fed from a
    command file as xfoil1.run_xfoil_logic does) with a time limit per angle. XFOIL reads a piped stdin in blocks and
    buffers its screen output, so progress is followed on disk: a SAVE of the paneled airfoil marks the end of the
    geometry setup, and each angle's CPWR file, written whether or not the angle converged, marks the end of that
    angle. An angle that has not finished within angle_timeout seconds is abandoned: XFOIL is killed and a fresh
    session continues from the next angle. A geometry setup that does not finish within setup_timeout abandons the
    case. Returns (polar rows, timed-out angles, error or None).
    """
    rows, timed_out, error = [], [], None
    remaining = list(alphas)
    session = 0
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    while remaining:
        session += 1
        polar, setup = f"polar_{session}.txt", f"setup_{session}.dat"
        markers = [f"cp_s{session}_{i}.dat" for i in range(len(remaining))]
        for f in [polar, setup] + markers:
            if os.path.exists(os.path.join(work_dir, f)):
                os.remove(os.path.join(work_dir, f))
        commands = ["PLOP", "G", "", f"LOAD {os.path.basename(airfoil_file)}", "PANE", f"SAVE {setup}", "OPER",
                    f"Visc {reynolds}", f"Mach {mach}", f"ITER {int(n_iter)}", "PACC", polar, ""]
        for alpha, marker in zip(remaining, markers):
            commands += [f"ALFA {alpha:.2f}", f"CPWR {marker}"]
        commands += ["PACC", "", "QUIT"]
        input_path = os.path.join(work_dir, f"xfoil_input_{session}.txt")
        with open(input_path, "w") as f:
            f.write("\n".join(commands) + "\n")

        def wait_for(name, limit):
            deadline = time.time() + limit
            while time.time() < deadline:
                if os.path.exists(os.path.join(work_dir, name)):
                    return True
                if proc.poll() is not None:
                    return os.path.exists(os.path.join(work_dir, name))
                time.sleep(0.05)
            return False

        with open(input_path) as input_f:
            proc = subprocess.Popen([xfoil_path], stdin=input_f, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, cwd=work_dir, creationflags=flags)
            try:
                if not wait_for(setup, setup_timeout):
                    error = ("XFOIL exited during geometry load/paneling" if proc.poll() is not None
                             else f"XFOIL geometry load/paneling did not finish within {setup_timeout:g} s")
                    timed_out += remaining if proc.poll() is None else []
                    remaining = []
                else:
                    batch = list(remaining)
                    for alpha, marker in zip(batch, markers):
                        done = wait_for(marker, angle_timeout)
                        remaining.remove(alpha)
                        if done:
                            continue
                        if proc.poll() is None:
                            timed_out.append(alpha)  # hung: kill and carry on from the next angle
                        else:
                            error = f"XFOIL exited at alpha {alpha:g} (exit code {proc.returncode})"
                        break
                    if not remaining:
                        try:
                            proc.wait(30)
                        except subprocess.TimeoutExpired:
                            pass
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(10)
        rows += _read_polar(os.path.join(work_dir, polar))
        for f in [setup] + markers:
            if os.path.exists(os.path.join(work_dir, f)):
                os.remove(os.path.join(work_dir, f))
    return rows, timed_out, error


def run_xfoil_case(case, case_dir, airfoil_dir, xfoil_path, timeout, n_iter=100, angle_timeout=None):
    """Geometry as the GUI prepares it, then the XFOIL sweep. With angle_timeout, angles run one at a time and an
    angle that hangs is abandoned (run_xfoil_sweep); otherwise FALCON's single-run path with one time limit."""
    from Scripts.Solver_and_Results import xfoil1

    _, _, _, _, geometry = prepare_geometry(case, airfoil_dir, case_dir)
    fired = threading.Event()
    timed_out, error = [], None

    if angle_timeout:
        run_file = os.path.join(case_dir, "run_airfoil.dat")
        if not xfoil1.clean_airfoil_geometry(os.path.join(case_dir, "output.dat"), run_file):
            shutil.copy(os.path.join(case_dir, "output.dat"), run_file)
        polar, timed_out, error = run_xfoil_sweep(xfoil_path, case_dir, run_file, case.reynolds, case.mach,
                                                  case.alphas(), n_iter, angle_timeout)
    else:
        def stop():
            fired.set()
            xfoil1.stop_xfoil()

        watchdog = threading.Timer(timeout, stop)
        watchdog.start()
        try:
            polar, _ = xfoil1.run_xfoil_logic(xfoil_path, case_dir, os.path.join(case_dir, "output.dat"),
                                              case.reynolds, case.alpha_max, case.alpha_min, case.alpha_step,
                                              case.mach)
        finally:
            watchdog.cancel()

    # XFOIL writes only converged points to its polar, so an angle missing from it did not converge.
    points = []
    for alpha in case.alphas():
        row = next((r for r in polar if len(r) >= 5 and abs(r[0] - alpha) < 1e-3), None)
        hung = any(abs(a - alpha) < 1e-6 for a in timed_out)
        # XFOIL 6.99 polar columns: alpha, CL, CD, CDp, CM, Top_Xtr, Bot_Xtr
        points.append({
            "alpha": alpha, "converged": row is not None,
            "state": "converged" if row else ("timeout" if hung else "not_converged"),
            "cl": row[1] if row else None, "cd": row[2] if row else None, "cdp": row[3] if row else None,
            "cm": row[4] if row else None,
            "xtr_top": row[5] if row and len(row) > 5 else None,
            "xtr_bottom": row[6] if row and len(row) > 6 else None,
        })
    result = {"geometry": geometry, "points": points, "timed_out": fired.is_set(),
              "xfoil": {"iter": n_iter, "angle_timeout_s": angle_timeout,
                        "angles_timed_out": [float(a) for a in timed_out]}}
    if error:
        result["xfoil"]["error"] = error
    return result


def _count_cells(mesh_path):
    with open(mesh_path) as f:
        for line in f:
            if line.startswith("NELEM"):
                return int(line.split("=")[1])
    return None


def read_history(history_path, settings):
    """Final coefficients and convergence state of one SU2 run, from its history file."""
    out = {"converged": False, "state": "no_history", "cl": None, "cd": None, "cm": None,
           "iterations": None, "final_residual": None}
    if not history_path or not os.path.exists(history_path):
        return out
    df = pd.read_csv(history_path)
    df.columns = [c.strip().strip('"') for c in df.columns]
    if df.empty:
        out["state"] = "empty_history"
        return out
    last = df.iloc[-1]

    def value(column):
        if column not in df.columns:
            return None
        v = float(last[column])
        return v

    out.update(cl=value("CL"), cd=value("CD"), cm=value("CMz"))
    iter_column = next((c for c in ("Inner_Iter", "Time_Iter") if c in df.columns), None)
    out["iterations"] = int(last[iter_column]) + 1 if iter_column else len(df)
    residual_column = RESIDUAL_COLUMNS.get(str(settings.get("CONV_FIELD", "")).strip())
    residual = value(residual_column) if residual_column else None
    out["final_residual"] = residual

    target = float(settings.get("CONV_RESIDUAL_MINVAL", -8))
    max_iter = int(float(settings.get("ITER") or 0))
    finite = all(v is not None and math.isfinite(v) for v in (out["cl"], out["cd"]))
    if not finite or (residual is not None and not math.isfinite(residual)):
        out["state"] = "diverged"
    elif residual is not None and residual <= target:
        out["state"], out["converged"] = "converged", True
    elif max_iter and out["iterations"] >= max_iter:
        out["state"] = "max_iterations"
    else:
        out["state"] = "stopped"
    return out


def run_su2_case(case, case_dir, airfoil_dir, cores, timeout, dry_run=False, su2_config=None, max_iter=None,
                 su2_settings=None):
    """
    Mesh the case (case.mesh) and run SU2 for each angle. The SU2 settings are, in order of precedence: the
    configuration file su2_config (numerical settings verbatim, write_config_from_file); su2_settings, a
    {"regime", "settings"} pair as the GUI's Solver Setup tab hands to prepare_su2_config; otherwise the settings the
    GUI selects automatically for the case's Mach and Reynolds numbers.
    """
    from Scripts.Meshing.meshing import generate_mesh
    from Scripts.Meshing.wall_spacing import design_yplus, measure_wall_spacing
    from Scripts.Solver_and_Results.su2_analyzer import SU2Runner, extract_surface_yplus, prepare_su2_config

    x, y, upper, lower, geometry = prepare_geometry(case, airfoil_dir, case_dir)
    if su2_config:
        regime, template, settings = "from file", os.path.abspath(su2_config), _read_cfg(su2_config)
    elif su2_settings:
        regime, template, settings = su2_settings["regime"], "GUI Solver Setup tab", dict(su2_settings["settings"])
    else:
        regime, template, settings = gui_recommended_settings(case.mach, case.reynolds)

    # The mesher writes airfoil.su2 into the working directory, as it does from the GUI.
    os.chdir(case_dir)
    if case.mesh == "hybrid":
        from Scripts.Meshing.hybrid import generate_hybrid
        generate_hybrid(upper, lower, case.reynolds, case.mach, y_plus=case.y_plus, show_graphics=False,
                        hide_output=True)
    else:
        generate_mesh(x, y, case.reynolds, case.mach, y_plus=case.y_plus, show_graphics=False, hide_output=True)
    mesh_path = os.path.join(case_dir, "airfoil.su2")
    mesh = {"type": case.mesh, "cells": _count_cells(mesh_path)}
    _, spacing = measure_wall_spacing(mesh_path)
    if spacing is not None and np.any(np.isfinite(spacing)):
        yp = design_yplus(spacing[np.isfinite(spacing)], case.reynolds)
        mesh.update(design_yplus_median=float(np.median(yp)), design_yplus_max=float(np.max(yp)))

    runner = None if dry_run else SU2Runner(num_procs=cores, use_mpi=cores > 1)
    if runner is not None and not runner.executables_found:
        raise RuntimeError("SU2_CFD (or mpiexec) not found on PATH")

    points = []
    for alpha in case.alphas():
        run_dir = os.path.join(case_dir, f"AoA_{alpha:.2f}")
        os.makedirs(run_dir, exist_ok=True)
        shutil.copy2(mesh_path, os.path.join(run_dir, "airfoil.su2"))
        if su2_config:
            config, settings = write_config_from_file(su2_config, run_dir, alpha, case.reynolds, case.mach, max_iter)
        else:
            config = prepare_su2_config(run_dir, settings, "airfoil.su2", regime, alpha, case.reynolds, case.mach)
        if dry_run:
            points.append({"alpha": alpha, "converged": False, "state": "dry_run", "config": config})
            continue

        fired = threading.Event()

        def stop():
            fired.set()
            proc = runner.current_process
            if proc is not None:
                kill_process_tree(proc.pid)  # mpiexec alone can leave its ranks running
            runner.stop()

        watchdog = threading.Timer(timeout, stop)
        watchdog.start()
        start = time.time()
        try:
            status, history, _, surface = runner.run_analysis(config, alpha, run_dir, None, None)
        finally:
            watchdog.cancel()

        point = {"alpha": alpha, "run_status": "TIMEOUT" if fired.is_set() else status,
                 "wall_s": round(time.time() - start, 1)}
        point.update(read_history(history, settings))
        if fired.is_set():
            point["state"], point["converged"] = "timeout", False
        yplus = extract_surface_yplus(surface) if surface else None
        if yplus:
            point.update(yplus_max=yplus["max"], yplus_median=yplus["median"])
        points.append(point)

    return {"geometry": geometry, "regime": regime, "template": template, "settings": settings,
            "mesh": mesh, "points": points, "timed_out": any(p["state"] == "timeout" for p in points),
            "dry_run": dry_run}


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def settings_fingerprint(su2_config=None, max_iter=None, su2_settings=None):
    """Short id of the SU2 settings a batch runs with: the configuration file (and its iteration cap), the settings
    from the GUI's Solver Setup tab, or - with neither - the automatic per-case selection."""
    if su2_config:
        with open(su2_config) as f:
            payload = {"file": f.read(), "max_iter": max_iter}
    elif su2_settings:
        payload = {"regime": su2_settings.get("regime"), "settings": su2_settings.get("settings")}
    else:
        payload = {"auto": True}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _status_path(out_dir, case):
    return os.path.join(out_dir, case.case_id, "status.json")


def read_status(out_dir, case):
    path = _status_path(out_dir, case)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_status(case_dir, record):
    tmp = os.path.join(case_dir, "status.json.tmp")
    with open(tmp, "w") as f:
        json.dump(record, f, indent=1, default=float)
    os.replace(tmp, os.path.join(case_dir, "status.json"))


def _mark(out_dir, case, state, message):
    """Record a case the parent had to stop, or that died without writing its own status."""
    case_dir = os.path.join(out_dir, case.case_id)
    os.makedirs(case_dir, exist_ok=True)
    record = read_status(out_dir, case) or {"case": asdict(case), "case_id": case.case_id}
    record.update(state=state, error=message, finished=datetime.now().isoformat(timespec="seconds"))
    _write_status(case_dir, record)


def _case_process(case_fields, out_dir, airfoil_dir, xfoil_path, cores, timeout, dry_run=False, su2_config=None,
                  max_iter=None, su2_settings=None, xfoil_iter=100, xfoil_angle_timeout=None):
    """Entry point of the child process that runs one case."""
    case = Case(**case_fields)
    case_dir = os.path.join(out_dir, case.case_id)
    os.makedirs(case_dir, exist_ok=True)
    log = open(os.path.join(case_dir, "log.txt"), "w", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = log

    record = {"case": asdict(case), "case_id": case.case_id, "state": "running",
              "started": datetime.now().isoformat(timespec="seconds"), "falcon_commit": _git_commit(),
              "cores": cores if case.solver == "su2" else 1}
    if case.solver == "su2":
        record["settings_id"] = settings_fingerprint(su2_config, max_iter, su2_settings)
    _write_status(case_dir, record)
    start = time.time()
    try:
        if case.solver == "xfoil":
            result = run_xfoil_case(case, case_dir, airfoil_dir, xfoil_path, timeout, xfoil_iter, xfoil_angle_timeout)
        else:
            result = run_su2_case(case, case_dir, airfoil_dir, cores, timeout, dry_run, su2_config, max_iter,
                                  su2_settings)
        record.update(result)
        n_ok = sum(1 for p in result["points"] if p["converged"])
        if result.get("dry_run"):
            record["state"] = "dry_run"  # not a final state, so a real run picks the case up again
        elif result["points"] and n_ok == len(result["points"]):
            record["state"] = "converged"
        elif n_ok:
            record["state"] = "partial"
        else:
            record["state"] = "timeout" if result.get("timed_out") else "failed"
    except Exception:
        record["state"] = "failed"
        record["error"] = traceback.format_exc()
        print(record["error"])
    finally:
        record["wall_s"] = round(time.time() - start, 1)
        record["finished"] = datetime.now().isoformat(timespec="seconds")
        _write_status(case_dir, record)
        log.close()


def kill_process_tree(pid):
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True)
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _run_pool(cases, workers, child_args, time_limit, out_dir, log, should_stop=None):
    """Run cases in child processes, at most `workers` at a time, killing any that outlive `time_limit(case)`.
    When should_stop() turns true, the running cases are killed and marked 'stopped' and the rest are not started."""
    ctx = multiprocessing.get_context("spawn")
    queue = list(cases)
    running = {}
    try:
        while queue or running:
            if should_stop is not None and should_stop():
                for case_id, (case, proc, _) in list(running.items()):
                    kill_process_tree(proc.pid)
                    proc.join(10)
                    del running[case_id]
                    _mark(out_dir, case, "stopped", "stopped by user")
                    log(f"[stopped] {case_id}")
                queue.clear()
                break
            while queue and len(running) < workers:
                case = queue.pop(0)
                proc = ctx.Process(target=_case_process, args=child_args(case))
                proc.start()
                running[case.case_id] = (case, proc, time.time() + time_limit(case))
                log(f"[start] {case.case_id}")
            time.sleep(0.5)
            for case_id, (case, proc, deadline) in list(running.items()):
                if not proc.is_alive():
                    proc.join()
                    del running[case_id]
                    state = (read_status(out_dir, case) or {}).get("state")
                    if state in (None, "running"):
                        state = "crashed"
                        _mark(out_dir, case, state, f"case process exited with code {proc.exitcode}")
                    log(f"[{state}] {case_id}")
                elif time.time() > deadline:
                    kill_process_tree(proc.pid)
                    proc.join(10)
                    del running[case_id]
                    _mark(out_dir, case, "timeout", "batch time limit reached")
                    log(f"[timeout] {case_id}")
    finally:
        for case, proc, _ in running.values():
            kill_process_tree(proc.pid)
            _mark(out_dir, case, "crashed", "batch interrupted")


def run_batch(cases, out_dir, airfoil_dir, xfoil_path=None, cores=8, workers=4, xfoil_timeout=180.0,
              su2_timeout=4 * 3600.0, rerun_failed=False, dry_run=False, su2_config=None, max_iter=None, log=print,
              should_stop=None, su2_settings=None, xfoil_iter=100, xfoil_angle_timeout=None):
    """
    Run every case that has not finished yet and write the summaries.

    XFOIL cases run `workers` at a time on one core each; SU2 cases run one at a time on `cores` MPI ranks.
    xfoil_timeout limits one XFOIL sweep, su2_timeout one SU2 angle; the parent kills any case that outlives its
    limit plus a margin for the geometry fit and meshing. dry_run stops SU2 cases after writing the mesh and
    configuration files. su2_config replaces the automatically selected SU2 settings with those of a given
    configuration file (max_iter caps its ITER); otherwise su2_settings ({"regime", "settings"}, as the GUI's Solver
    Setup tab gives them) replace them when given. should_stop, if given, is polled while cases run (the GUI's Stop
    button); cases it interrupts are marked 'stopped' and run again on the next call. Returns the statistics
    written to batch_summary.json. xfoil_iter is XFOIL's ITER (FALCON's single runs use 100); with
    xfoil_angle_timeout (seconds) XFOIL runs one angle at a time and abandons any angle that takes longer, and
    xfoil_timeout then only caps a whole case.
    """
    out_dir = os.path.abspath(out_dir)
    airfoil_dir = os.path.abspath(airfoil_dir)
    os.makedirs(out_dir, exist_ok=True)

    # A finished SU2 case is only skipped when it ran with the settings of this batch. Status files written before
    # settings were recorded count as the automatic selection.
    wanted_id = settings_fingerprint(su2_config, max_iter, su2_settings)
    auto_id = settings_fingerprint()

    pending = []
    for case in cases:
        status = read_status(out_dir, case) or {}
        state = status.get("state")
        if state in FINAL_STATES and not (rerun_failed and state in RETRY_STATES):
            if case.solver == "su2" and (status.get("settings_id") or auto_id) != wanted_id:
                log(f"[rerun] {case.case_id}: finished with different SU2 settings ({state})")
            else:
                log(f"[skip] {case.case_id} ({state})")
                continue
        pending.append(case)

    xfoil_cases = [c for c in pending if c.solver == "xfoil"]
    su2_cases = [c for c in pending if c.solver == "su2"]

    if xfoil_cases:
        resolved = shutil.which(xfoil_path or "xfoil") or (xfoil_path if xfoil_path and os.path.isfile(xfoil_path)
                                                          else None)
        if resolved is None:
            raise FileNotFoundError(f"XFOIL executable not found: {xfoil_path!r}")
        xfoil_path = os.path.abspath(resolved)
        _run_pool(xfoil_cases, workers,
                  lambda c: (asdict(c), out_dir, airfoil_dir, xfoil_path, 1, xfoil_timeout, False, None, None, None,
                             xfoil_iter, xfoil_angle_timeout),
                  lambda c: (len(c.alphas()) * (xfoil_angle_timeout + 120.0) + 300.0 if xfoil_angle_timeout
                             else xfoil_timeout + 300.0), out_dir, log, should_stop)
    if su2_cases and not (should_stop is not None and should_stop()):
        _run_pool(su2_cases, 1,
                  lambda c: (asdict(c), out_dir, airfoil_dir, xfoil_path, cores, su2_timeout, dry_run,
                             os.path.abspath(su2_config) if su2_config else None, max_iter, su2_settings),
                  lambda c: su2_timeout * len(c.alphas()) + 1800.0, out_dir, log, should_stop)

    return write_summary(cases, out_dir)


def write_summary(cases, out_dir):
    """Write cases_summary.csv (one row per case), results.csv (one row per angle) and batch_summary.json."""
    case_rows, point_rows = [], []
    for case in cases:
        status = read_status(out_dir, case) or {"state": "not_run"}
        points = status.get("points", [])
        error = (status.get("error") or "").strip().splitlines()
        case_rows.append({
            "case_id": case.case_id, "solver": case.solver, "airfoil": case.airfoil, "re": case.reynolds,
            "mach": case.mach, "n_alpha": len(case.alphas()),
            "repanel": case.repanel, "n_points": case.n_points if case.repanel else None,
            "mesh": case.mesh if case.solver == "su2" else None,
            "y_plus": case.y_plus if case.solver == "su2" else None,
            "n_converged": sum(1 for p in points if p.get("converged")), "state": status["state"],
            "wall_s": status.get("wall_s"), "cores": status.get("cores"),
            "fit_method": (status.get("geometry") or {}).get("method"),
            "rmse_parsec": (status.get("geometry") or {}).get("rmse_parsec"),
            "rmse_cst": (status.get("geometry") or {}).get("rmse_cst"),
            "mesh_cells": (status.get("mesh") or {}).get("cells"),
            "error": error[-1] if error else "",
        })
        for p in points:
            point_rows.append({
                "case_id": case.case_id, "solver": case.solver, "airfoil": case.airfoil, "re": case.reynolds,
                "mach": case.mach, "alpha": p.get("alpha"), "converged": p.get("converged"),
                "state": p.get("state"), "cl": p.get("cl"), "cd": p.get("cd"), "cdp": p.get("cdp"), "cm": p.get("cm"),
                "iterations": p.get("iterations"), "final_residual": p.get("final_residual"),
                "yplus_max": p.get("yplus_max"), "yplus_median": p.get("yplus_median"),
                "xtr_top": p.get("xtr_top"), "xtr_bottom": p.get("xtr_bottom"), "wall_s": p.get("wall_s"),
            })
    cases_df = pd.DataFrame(case_rows)
    points_df = pd.DataFrame(point_rows)
    cases_df.to_csv(os.path.join(out_dir, "cases_summary.csv"), index=False)
    points_df.to_csv(os.path.join(out_dir, "results.csv"), index=False)

    stats = {"generated": datetime.now().isoformat(timespec="seconds"), "falcon_commit": _git_commit()}
    for solver in SOLVERS:
        sub = cases_df[cases_df["solver"] == solver] if not cases_df.empty else cases_df
        if sub.empty:
            continue
        ran = sub[sub["state"] != "not_run"]
        pts = points_df[points_df["solver"] == solver] if not points_df.empty else points_df
        requested = int(ran["n_alpha"].sum())
        converged = int(ran["n_converged"].sum())
        core_hours = float((ran["wall_s"].fillna(0) * ran["cores"].fillna(1)).sum() / 3600.0)
        states = ran["state"].value_counts().to_dict()
        stats[solver] = {
            "cases": int(len(sub)), "cases_run": int(len(ran)),
            "cases_by_state": {k: int(v) for k, v in states.items()},
            "angles_requested": requested, "angles_converged": converged,
            "angle_convergence_rate": converged / requested if requested else None,
            "wall_hours": float(ran["wall_s"].fillna(0).sum() / 3600.0),
            "core_hours": core_hours,
            "cases_per_core_hour": len(ran) / core_hours if core_hours else None,
            "converged_angles_per_core_hour": converged / core_hours if core_hours else None,
            "points_recorded": int(len(pts)),
        }
    stats["excel"] = write_excel(cases_df, points_df, out_dir)
    with open(os.path.join(out_dir, "batch_summary.json"), "w") as f:
        json.dump(stats, f, indent=1)
    return stats


# Spreadsheet columns per solver: (results.csv column, heading). XFOIL writes only converged angles to its polar.
EXCEL_COLUMNS = {
    "xfoil": [("airfoil", "Airfoil"), ("alpha", "alpha"), ("cl", "CL"), ("cd", "CD"), ("cdp", "CDp"), ("cm", "CM"),
              ("xtr_top", "Top_Xtr"), ("xtr_bottom", "Bot_Xtr")],
    "su2": [("airfoil", "Airfoil"), ("alpha", "alpha"), ("cl", "CL"), ("cd", "CD"), ("cm", "CM"),
            ("converged", "converged"), ("state", "state"), ("iterations", "iterations"),
            ("final_residual", "final_residual"), ("yplus_max", "yplus_max")],
}


def _sheet_names(groups):
    """'Re 50000' per Reynolds number; solver and Mach are added only when the batch mixes them."""
    solvers = {g[0] for g in groups}
    machs = {g[2] for g in groups}
    names = {}
    for solver, reynolds, mach in groups:
        re_txt = f"{int(reynolds)}" if float(reynolds).is_integer() else f"{reynolds:g}"
        name = f"Re {re_txt}"
        if len(machs) > 1:
            name += f" M {mach:g}"
        if len(solvers) > 1:
            name = f"{solver.upper()} {name}"
        names[(solver, reynolds, mach)] = regex.sub(r"[\[\]:*?/\\]", "_", name)[:31]
    return names


def write_excel(cases_df, points_df, out_dir):
    """dataset.xlsx: one sheet per Reynolds number (rows sorted by airfoil, then angle), an Info sheet and a
    "Failed or not run" sheet. Needs openpyxl; returns the path, or a note when it could not be written."""
    if cases_df.empty:
        return None
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        return "not written: install openpyxl (pip install openpyxl) for dataset.xlsx"
    path = os.path.join(out_dir, "dataset.xlsx")
    groups = sorted({(r.solver, float(r.re), float(r.mach)) for r in cases_df.itertuples()})
    names = _sheet_names(groups)
    info = []
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        for solver, reynolds, mach in groups:
            cols = EXCEL_COLUMNS[solver]
            sub = points_df[(points_df["solver"] == solver) & (points_df["re"].astype(float) == reynolds)
                            & (points_df["mach"].astype(float) == mach)] if not points_df.empty else points_df
            if solver == "xfoil" and not sub.empty:
                sub = sub[sub["converged"] == True]  # noqa: E712 - pandas element-wise
            sheet = (sub.sort_values(["airfoil", "alpha"])[[c for c, _ in cols]].rename(columns=dict(cols))
                     if not sub.empty else pd.DataFrame(columns=[h for _, h in cols]))
            sheet.to_excel(xl, sheet_name=names[(solver, reynolds, mach)], index=False)
            xl.sheets[names[(solver, reynolds, mach)]].freeze_panes = "B2"
            cases = cases_df[(cases_df["solver"] == solver) & (cases_df["re"].astype(float) == reynolds)
                             & (cases_df["mach"].astype(float) == mach)]
            info.append({"sheet": names[(solver, reynolds, mach)], "solver": solver, "Re": reynolds, "Mach": mach,
                         "airfoils": len(cases), "airfoils with data": int((cases["n_converged"] > 0).sum()),
                         "angles converged": int(cases["n_converged"].sum()),
                         "angles requested": int(cases["n_alpha"].sum())})
        pd.DataFrame(info).to_excel(xl, sheet_name="Info", index=False)
        notes = pd.DataFrame({"note": [
            f"Generated {datetime.now().isoformat(timespec='seconds')} by FALCON (commit {_git_commit()}).",
            "XFOIL: free transition, Ncrit 9 (XFOIL defaults), PANE re-paneling; only converged angles appear.",
            "One row per airfoil and angle; CDp is the pressure drag, Top_Xtr/Bot_Xtr the transition locations.",
            "Per-case logs and status.json are in the case folders; results.csv holds every requested angle."]})
        notes.to_excel(xl, sheet_name="Info", index=False, startrow=len(info) + 2)
        failed = cases_df[~cases_df["state"].isin(["converged", "partial"])][
            ["airfoil", "solver", "re", "mach", "state", "error"]].sort_values(["re", "airfoil"])
        failed.to_excel(xl, sheet_name="Failed or not run", index=False)
    return path