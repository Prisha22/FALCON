"""
Run the FALCON workflow headless over a list of cases (Scripts/Batch/batch_runner.py).

    python falcon_batch.py cases.csv --out output/batch --xfoil "C:/path/to/xfoil.exe" --cores 12

The case list is a CSV with columns airfoil, solver (xfoil or su2), re, mach, alpha_min, alpha_max, alpha_step and
optionally n_points and y_plus. Airfoil names are files in the airfoil directory (Airfoil_DAT_Selig by default).
Each case gets a folder under --out with its log, status.json and solver files; cases_summary.csv, results.csv
and batch_summary.json are written at the end. Running the same command again skips finished cases.
"""
import argparse
import json
import multiprocessing
import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from Scripts.Batch.batch_runner import load_cases, run_batch  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run FALCON headless over a case list.")
    parser.add_argument("cases", help="CSV case list")
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "output", "batch"), help="output folder")
    parser.add_argument("--airfoils", default=os.path.join(REPO_ROOT, "Airfoil_DAT_Selig"),
                        help="folder holding the airfoil coordinate files")
    parser.add_argument("--xfoil", default=shutil.which("xfoil") or "xfoil.exe", help="XFOIL executable")
    parser.add_argument("--cores", type=int, default=8, help="MPI ranks per SU2 run")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help="XFOIL cases run at the same time")
    parser.add_argument("--xfoil-timeout", type=float, default=180.0, help="seconds per XFOIL sweep")
    parser.add_argument("--su2-timeout", type=float, default=4 * 3600.0, help="seconds per SU2 angle")
    parser.add_argument("--rerun-failed", action="store_true", help="run failed, timed-out and crashed cases again")
    parser.add_argument("--dry-run", action="store_true",
                        help="SU2 cases: write the mesh and configuration files but do not start SU2")
    parser.add_argument("--su2-config", help="SU2 cases: use this configuration file's numerical settings for every "
                                             "case instead of the automatically selected ones")
    parser.add_argument("--max-iter", type=int, help="cap on ITER when --su2-config is used")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    print(f"{len(cases)} case(s) from {args.cases}")
    stats = run_batch(cases, args.out, args.airfoils, xfoil_path=args.xfoil, cores=args.cores,
                      workers=args.workers, xfoil_timeout=args.xfoil_timeout, su2_timeout=args.su2_timeout,
                      rerun_failed=args.rerun_failed, dry_run=args.dry_run, su2_config=args.su2_config,
                      max_iter=args.max_iter)
    print(json.dumps(stats, indent=1))

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
