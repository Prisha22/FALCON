# FALCON tests

FALCON now has an automated test suite of 95 tests, written with pytest and run on every push by GitHub Actions. The fast tests take about 3 minutes. Tests that need external solvers are marked slow, and they skip themselves when SU2 or XFOIL isn’t installed. These tests verify if the code is working as intended for the current version. These only check the workflow on the software side and do not check accuracy of the results here. Head to Examples directory for validation cases and aerodynamic accuracy.

## Run the tests

    pip install -r requirements.txt -r requirements-dev.txt
    python -m pytest -m "not slow"      # fast tests, about 30 s
    python -m pytest -m slow            # hybrid mesh, plus real SU2 / XFOIL runs if those programs are installed

Passing tests print a dot, failing tests print what differed. Skipped tests (s) need something that is not
installed, for example SU2_CFD or XFOIL.

## What is covered

| File | What it tests |
|---|---|
| test_su2_config.py | SU2 config generation for low-Re, high-Re, transonic and supersonic; defaults; user values win |
| test_solver_parsing.py | SU2 history files: coefficients, converged / diverged / max-iterations / missing |
| test_batch_logic.py | case list, alpha ranges, config-from-file, re-run when settings change |
| test_geometry.py | coordinate reading, normalization, CST and PARSEC fits, spline re-paneling |
| test_xfoil.py | XFOIL geometry cleaning, argument checks, polar and Cp parsing (with a stand-in xfoil) |
| test_live_plotter.py | the live-plot history reader, including SU2 8 file naming |
| test_mesh.py | first-cell formula, wall spacing, structured and hybrid mesh files |
| test_smoke.py | small end-to-end dry run; optional real SU2 and XFOIL runs |

## If a test fails on your machine

- `No module named 'Scripts.Batch'` (or `Scripts.Geometry`, ...) - Python found a different folder called `Scripts`
  (Windows: Python's own `Scripts` folder). `tests/conftest.py` now puts your repo root first on the import path; make
  sure you are using this version, and run `python -m pytest` from the repo root.
- Delete old copies of test files lying loose in the repo root (for example `test_mesh.py`, `test_smoke.py`).
- Import errors for `gmsh` on Linux - install `libglu1-mesa`.
- The import paths must match your folders: `Scripts.Batch.batch_runner`, `Scripts.Solver_and_Results.su2_analyzer`,
  `Scripts.Geometry`, `Scripts.Meshing`. Add an empty `Scripts/Batch/__init__.py` if it is missing.
