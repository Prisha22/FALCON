# FALCON datasets

These datasets were generated with FALCON's batch mode (the **Batch** tab of the GUI, which drives
`Scripts/Batch/batch_runner.py`). For every case FALCON runs the same steps as an interactive analysis
(geometry import, re-paneling, meshing for SU2, solver configuration, solution and post-processing),
without user interaction. Each case runs in its own process with a time limit; failed or unconverged
operating points are recorded instead of stopping the batch, and a stopped batch resumes from where it
left off.

## XFOIL dataset (`FALCON_XFOIL/`)

| File | Contents |
|---|---|
| `dataset_xfoil_falcon_re50k.csv` | polars at Re = 50 000 |
| `dataset_xfoil_falcon_re100k.csv` | polars at Re = 100 000 |

- All 1624 airfoils of the coordinate database shipped with FALCON (`Airfoil_DAT_Selig/`), M = 0,
  alpha = -10 to 15 deg in 1 deg steps.
- Each airfoil re-paneled to 200 points per surface; XFOIL 6.99, at most 500 iterations and 300 s per angle.
- Only converged operating points are listed (81.8 % of the 84 448 requested).
- Columns: `Airfoil, alpha, CL, CD, CDp, CM, Top_Xtr, Bot_Xtr` (alpha in degrees; Top_Xtr/Bot_Xtr are the
  upper/lower-surface transition locations x/c).

## SU2 dataset (`FALCON_SU2/`)

RANS results for a small set of airfoils, Reynolds numbers and angles of attack.

- `cases.csv`: the case list (airfoil, solver, Re, Mach, alpha range, points per surface, target y+, mesh type).
- `results.csv`: one row per operating point: CL, CD, CM, convergence state, iterations, final residual,
  achieved y+ (max and median) and run time.
- `cases_summary.csv`, `batch_summary.json`, `dataset.xlsx`: per-case and batch totals.
- One folder per case with the mesh (`airfoil.su2`), the re-paneled geometry (`output.dat`), the run log and
  `status.json`, which records the exact SU2 settings, mesh size and per-angle results of that case.
- The per-angle SU2 output folders (`AoA_*`: configuration, convergence history, surface and flow-field files)
  are omitted from this repository because of their size.

## Using the data

The CSV files can be read with any spreadsheet program or, in Python:

```python
import pandas as pd
xfoil = pd.read_csv("Dataset/FALCON_XFOIL/dataset_xfoil_falcon_re100k.csv")
su2 = pd.read_csv("Dataset/FALCON_SU2/results.csv")
su2 = su2[su2["converged"]]          # keep converged operating points only
```

To reproduce or extend a dataset, open the **Batch** tab in FALCON, select the airfoil files, the solver,
Reynolds and Mach numbers and the angle-of-attack range, and run. To repeat the SU2 cases exactly,
use the settings stored in each case's `status.json`.

## Larger dataset

A larger FALCON-generated dataset is currently in preparation and will be released separately.
