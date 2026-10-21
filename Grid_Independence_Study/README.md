# Grid independence study

Two mesh-refinement families generated with FALCON's meshers and solved with SU2 v8.5.0. Each study uses the
same solver settings on every grid, so the differences between levels come from the mesh alone.
`python grid_study.py` recomputes the tables below from the SU2 histories.

## `NACA0012_structured/` (structured C-mesh)

NACA 0012 case of the SU2 turbulent tutorial: M = 0.15, Re = 6e6, alpha = 10 deg, compressible RANS with SST,
tutorial numerical settings. Four grids, each with about twice the cells of the previous one (refinement ratio
r = sqrt(2)); all levels were run to an RMS density residual of 1e-12.

| Grid | Cells | C_l | C_d | dC_l (%) | dC_d (%) |
|---|---|---|---|---|---|
| L1 | 114,935 | 1.06944 | 0.01683 | -0.98 | +16.50 |
| L2 | 229,376 | 1.07758 | 0.01543 | -0.22 | +6.79 |
| L3 | 459,378 | 1.08003 | 0.01476 | <0.01 | +2.11 |
| L4 | 917,504 | 1.07998 | 0.01445 | | |

C_d from L2-L4: observed order 2.29, Richardson extrapolation 0.01420, GCI on L4 2.2% (2.6% with p = 2).

- `meshes/airfoil_L*_<cells>.su2.zip`: the four meshes (unzip before use).
- `config_aoa_10.00.cfg`: SU2 configuration, identical for all grids except `MESH_FILENAME` and the output file names.
- `history_n0012_L*.csv`: convergence histories; `surf_n0012_L*.csv`: surface solutions (C_p, C_f).

## `DoubleWedge_hybrid/` (hybrid mesh)

10% symmetric double wedge (R&M 2800 geometry), M = 1.86, Re = 0.66e6, FALCON's supersonic settings (Roe, MUSCL,
Venkatakrishnan limiter, adaptive CFL). Four grids obtained by multiplying every mesh size of the hybrid mesher by
S = 2, sqrt(2), 1 and 1/sqrt(2) (L0-L3; L2 is the default mesh). These meshes have boundary-layer fans at both the
leading and the trailing edge; the v1.0 mesher places a fan at the trailing edge only.

RANS (SST) at alpha = 2, 8 and 14 deg, and an inviscid (Euler) computation at alpha = 6 deg, which has an exact
shock-expansion solution (C_l = 0.27435, C_d = 0.05536). Coefficients are averaged over the last 500 iterations;
each run was continued until lift and drag stopped changing (`iterations.txt`). Example, RANS at alpha = 8 deg:

| Grid | Cells | C_l | C_d | dC_l (%) | dC_d (%) |
|---|---|---|---|---|---|
| L0 | 125,435 | 0.36581 | 0.08578 | -0.34 | +0.37 |
| L1 | 261,129 | 0.36752 | 0.08570 | +0.13 | +0.27 |
| L2 | 504,089 | 0.36725 | 0.08562 | +0.06 | +0.19 |
| L3 | 1,013,510 | 0.36705 | 0.08546 | | |

- `meshes/airfoil_L*_<cells>.su2.zip`: the four meshes, shared by all cases.
- `<case>/config_aoa_<alpha>.cfg`: SU2 configuration (written for L3; only `MESH_FILENAME` changes between levels).
- `<case>/history_wedge_L*.csv`: convergence histories (restarts concatenated); `<case>/surf_wedge_L*.csv`: surface
  solutions.
- `gci_report.txt`: observed orders, Richardson extrapolation and GCI for every case and quantity.
