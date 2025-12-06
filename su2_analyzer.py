# su2_analyzer.py
import os
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime
import shutil
import threading
import time
from queue import Queue
import io
import pyvista as pv
pv.OFF_SCREEN = True
import glob
import logging
import re
from typing import List, Tuple, Optional, Dict, Callable
import matplotlib.pyplot as plt

# Custom import for live plotting
try:
    from live_plotter import data_reader_thread
except ImportError:
    print("Warning: 'live_plotter' not found. Live plotting will be disabled.")
    data_reader_thread = None

# --- CONSTANTS ---
FIXED_CAMERA_POSITION = ((0.5, 0, 8), (0.5, 0, 0), (0, 1, 0))
PROCESS_TERMINATION_TIMEOUT = 5  # seconds
PLOTTER_WINDOW_SIZE = (1200, 800)
SCREENSHOT_WINDOW_SIZE = (1600, 1000)

COLOR_GEOMETRY = 'lightgrey'
COLOR_MESH_EDGES = 'black'
COLOR_BG_WHITE = 'white'
COLOR_BG_BLACK = 'black'

SU2_COMPRESSIBLE_SETTINGS = {
    'SOLVER': ['RANS', 'NAVIER_STOKES', 'EULER'],
    'KIND_TURB_MODEL': ['SA', 'SST', 'NONE', 'SA_NEG'],
    'NUM_METHOD_GRAD': ['WEIGHTED_LEAST_SQUARES', 'GREEN_GAUSS'],
    'CONV_NUM_METHOD_FLOW': {
        'CENTRAL': ['JST', 'JST_KE', 'JST_MAT', 'LAX_FRIEDRICH'],
        'UPWIND': ['ROE', 'L2ROE', 'LMROE', 'AUSM', 'SLAU']
    },
    'SLOPE_LIMITER_FLOW': [
        'VENKATAKRISHNAN', 'VENKATAKRISHNAN_WANG',
        'NISHIKAWA_R3', 'NISHIKAWA_R4', 'NISHIKAWA_R5',
        'VAN_ALBADA_EDGE'
    ],
    'LINEAR_SOLVER': ['FGMRES', 'BCGSTAB', 'CONJUGATE_GRADIENT', 'SMOOTHER'],
    'LINEAR_SOLVER_PREC': ['ILU', 'JACOBI', 'LU_SGS'],
    'KIND_TRANS_MODEL': ['NONE', 'LM'],
}

SU2_INCOMPRESSIBLE_SETTINGS = {
    'SOLVER': ['INC_RANS', 'INC_NAVIER_STOKES', 'INC_EULER'],
    'KIND_TURB_MODEL': ['SST', 'SA', 'SA_NEG', 'NONE'],
    'NUM_METHOD_GRAD': ['GREEN_GAUSS', 'WEIGHTED_LEAST_SQUARES'],
    'CONV_NUM_METHOD_FLOW': {
        'CENTRAL': ['JST', 'LAX_FRIEDRICH'],
        'UPWIND': ['FDS']
    },
    'LINEAR_SOLVER': ['BCGSTAB', 'FGMRES', 'CONJUGATE_GRADIENT', 'SMOOTHER'],
    'LINEAR_SOLVER_PREC': ['ILU', 'JACOBI', 'LU_SGS'],
    'KIND_TRANS_MODEL': ['NONE', 'LM'],
}


class SU2Runner:
    """Manages the execution of SU2 simulations (serial or MPI)."""

    def __init__(self, su2_cfd_path="SU2_CFD", mpi_exec_path="mpiexec",
                 num_procs=8, use_mpi=True):
        self.su2_cfd_path = su2_cfd_path
        self.mpi_exec_path = mpi_exec_path
        self.num_procs = num_procs
        self.use_mpi = use_mpi

        self.current_process: Optional[subprocess.Popen] = None
        self.stop_requested = False
        self.su2_cfd_full_path: Optional[str] = None
        self.executables_found = self._validate_executables()

    def _validate_executables(self) -> bool:
        """Validates that required executables are accessible."""
        print("\n--- Checking For Required Executables ---")

        # SU2 is always required
        su2_path = shutil.which(self.su2_cfd_path)
        if not su2_path:
 
            if os.path.isabs(self.su2_cfd_path) and os.path.exists(self.su2_cfd_path):
                su2_path = self.su2_cfd_path
        
        if su2_path:
            print(f"SU2 executable found at: {su2_path}")
            self.su2_cfd_full_path = os.path.abspath(su2_path)  # Store absolute path
        else:
            print(f"ERROR: SU2 executable '{self.su2_cfd_path}' could not be found.")
            print(f"Current PATH: {os.environ.get('PATH', 'Not set')}")
            return False

        # MPI only required if use_mpi is True
        if self.use_mpi:
            mpi_path = shutil.which(self.mpi_exec_path)
            if not mpi_path:
                if os.path.isabs(self.mpi_exec_path) and os.path.exists(self.mpi_exec_path):
                    mpi_path = self.mpi_exec_path
            
            if mpi_path:
                print(f"MPI executable found at: {mpi_path}")
                self.mpi_exec_path = os.path.abspath(mpi_path)  # Store absolute path
            else:
                print(f"ERROR: MPI executable '{self.mpi_exec_path}' not found.")
                return False
        
        return True

    def update_parallel_settings(self, use_mpi: bool, num_procs: Optional[int] = None):
        """
        Called from the GUI when the user toggles 'Use MPI parallelisation'
        or changes core count.
        """
        self.use_mpi = bool(use_mpi)
        if num_procs is not None and num_procs > 0:
            self.num_procs = int(num_procs)

        print("\n[SU2Runner] Updating parallel settings:")
        print(f"  use_mpi = {self.use_mpi}")
        print(f"  num_procs = {self.num_procs}")


        self.executables_found = self._validate_executables()

    def stop(self):
        """Stops the currently running SU2 process gracefully and sets the global stop flag."""
        self.stop_requested = True
        proc = self.current_process  

        if proc and proc.poll() is None:
            print("Sending termination signal to SU2 process...")
            try:
                proc.terminate()
                proc.wait(timeout=PROCESS_TERMINATION_TIMEOUT)
                print("SU2 process terminated gracefully.")
            except subprocess.TimeoutExpired:
                print("SU2 did not respond to terminate, sending kill signal.")
                proc.kill()
        else:
            print("No active SU2 process to stop, or process already finished.")

    def run_analysis(
        self,
        config_path: str,
        angle_deg: float,
        output_dir: str,
        data_queue: Queue,
        stop_event
    ) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
        """
        Runs a single SU2 analysis for a given angle of attack.
        Returns (status, history_csv, solution_vtu, surface_csv).
        """
        if not self.executables_found:
            print("Aborting run: executables were not found during initial check.")
            return "FAILED", None, None, None

        self.stop_requested = False

        formatted_aoa_for_output = f"{angle_deg:.2f}"
        history_file_name = f"history_aoa_{formatted_aoa_for_output}.csv"
        actual_history_file = os.path.join(output_dir, history_file_name)

        surface_file_name = f"surf_aoa_{formatted_aoa_for_output}.csv"
        actual_surface_file = os.path.join(output_dir, surface_file_name)

        solution_file_name = f"flow_aoa_{formatted_aoa_for_output}.vtu"
        actual_solution_file = os.path.join(output_dir, solution_file_name)

        reader_thread = None
        if data_queue and stop_event and data_reader_thread:
            reader_thread = threading.Thread(
                target=data_reader_thread,
                args=(actual_history_file, data_queue, stop_event)
            )
            reader_thread.daemon = True
            reader_thread.start()

        # Construct command depending on serial / MPI mode
        if self.use_mpi:
            command = [
                self.mpi_exec_path, "-n", str(self.num_procs),
                self.su2_cfd_full_path,
                config_path
            ]
        else:
            command = [self.su2_cfd_full_path, os.path.basename(config_path)]

        print("===============================================================")
        if self.use_mpi:
            print(f"Running in PARALLEL (MPI enabled) with {self.num_procs} cores")
        else:
            print("Running in SERIAL mode (MPI disabled)")
        print("===============================================================")

        print(f"--- Starting SU2 (AoA={angle_deg:.2f}) ---")
        print(f"Command: {' '.join(command)}")
        print(f"Working Dir: {output_dir}\n")
    
        try:
            creation_flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            proc = subprocess.Popen(
                command,
                cwd=output_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                creationflags=creation_flags
            )
            self.current_process = proc

            with proc.stdout:
                for line in iter(proc.stdout.readline, ''):
                    if self.stop_requested:
                        break

                    # Detect MPI activity (ranks / parallel mention)
                    if "MPI rank" in line or "parallel" in line.lower():
                        print("🔹 MPI ACTIVITY DETECTED:", line.strip())

                    print(line, end='')

            proc.wait()

            if self.stop_requested:
                print(f"SU2 run for AoA {angle_deg} was STOPPED by the user.")
                return "STOPPED", None, None, None

            if proc.returncode != 0:
                print(f"\n--- SU2 run for AoA {angle_deg} FAILED (Code: {proc.returncode}) ---")
                return "FAILED", None, None, None
            else:
                print(f"\n--- SU2 run for AoA {angle_deg} completed successfully. ---")

        except FileNotFoundError:
            print("Error: Executable not found. This should have been caught by the initial check.")
            return "FAILED", None, None, None
        except Exception as e:
            print(f"An unexpected error occurred during SU2 execution: {e}")
            return "FAILED", None, None, None
        finally:
            if stop_event:
                stop_event.set()
            if reader_thread:
                reader_thread.join()
            self.current_process = None


        surface_pattern = os.path.join(output_dir, f"surf_aoa_{formatted_aoa_for_output}*.csv")
        surface_files = glob.glob(surface_pattern)
        if not surface_files:

            exact_surface = os.path.join(output_dir, f"surf_aoa_{formatted_aoa_for_output}.csv")
            if os.path.exists(exact_surface):
                surface_files = [exact_surface]
        
        actual_surface_file = surface_files[0] if surface_files else None
        if surface_files and len(surface_files) > 1:
            surface_files.sort(key=os.path.getmtime, reverse=True)
            actual_surface_file = surface_files[0]
        solution_pattern = os.path.join(output_dir, f"flow_aoa_{formatted_aoa_for_output}*.vtu")
        solution_files = glob.glob(solution_pattern)
        if not solution_files:
            exact_solution = os.path.join(output_dir, f"flow_aoa_{formatted_aoa_for_output}.vtu")
            if os.path.exists(exact_solution):
                solution_files = [exact_solution]
        
        actual_solution_file = solution_files[0] if solution_files else None
        if solution_files and len(solution_files) > 1:
            # Multiple files found, get the latest
            solution_files.sort(key=os.path.getmtime, reverse=True)
            actual_solution_file = solution_files[0]

        return "COMPLETED", actual_history_file, actual_solution_file, actual_surface_file


def prepare_su2_config(
    output_dir: str,
    gui_settings: dict,
    mesh_filename: str,
    flow_regime: str,
    aoa: float,
    reynolds: float,
    mach: float
) -> Optional[str]:
    """
    Generates an SU2 configuration file with regime-specific settings.
    """
    print(f"Preparing SU2 config for AoA={aoa:.2f}, Re={reynolds}, M={mach}, Regime={flow_regime}")

    # Determine specific flow regime
    is_transonic = flow_regime.lower() == "compressible" and 0.7 <= mach < 1.0
    is_supersonic = flow_regime.lower() == "compressible" and mach >= 1.0
    is_incompressible = flow_regime.lower() == "incompressible"

    config_lines = [
        "% ------------- DIRECT, ADJOINT, AND LINEARIZED PROBLEM DEFINITION ------------%",
        "MATH_PROBLEM= DIRECT",
        "RESTART_SOL= NO",
        ""
    ]

    # --- Define Flow State Variables for Reference Calculation ---
    ssl_temp = 288.15
    ssl_pres = 101325.0
    ssl_dens = 1.225
    gamma = 1.4
    r_gas = 287.05

    if is_incompressible:
        freestream_dens = ssl_dens
        freestream_temp = ssl_temp
        freestream_pres = ssl_pres
        sound_speed = np.sqrt(gamma * r_gas * ssl_temp)
        velocity_magnitude = mach * sound_speed
    else:
        # Compressible
        freestream_temp = gui_settings.get(
            'FREESTREAM_TEMPERATURE',
            293.0 if (is_transonic or is_supersonic) else 288.15
        )
        freestream_pres = gui_settings.get('FREESTREAM_PRESSURE', 101325.0)
        freestream_dens = freestream_pres / (r_gas * freestream_temp)
        sound_speed = np.sqrt(gamma * r_gas * freestream_temp)
        velocity_magnitude = mach * sound_speed

    if is_incompressible:
        # ==================== INCOMPRESSIBLE FLOW ====================
        solver = gui_settings.get('SOLVER', 'INC_RANS')
        turb_model = gui_settings.get('KIND_TURB_MODEL', 'SST')
        trans_model = gui_settings.get('KIND_TRANS_MODEL', 'NONE')

        config_lines.extend([
            f"SOLVER= {solver}",
            f"KIND_TURB_MODEL= {turb_model}"
        ])

        if turb_model == 'SA':
            config_lines.append("SA_OPTIONS= NONE")

        config_lines.append(f"FREESTREAM_NU_FACTOR= {gui_settings.get('FREESTREAM_NU_FACTOR', 4.0)}")

        if trans_model != 'NONE':
            config_lines.extend([
                f"KIND_TRANS_MODEL= {trans_model}",
                f"LM_OPTIONS= {gui_settings.get('LM_OPTIONS', '(MENTER_LANGTRY, MENTER_LANGTRY, NONE)')}",
                f"FREESTREAM_TURBULENCEINTENSITY= {gui_settings.get('FREESTREAM_TURBULENCEINTENSITY', 0.10)}",
                f"FREESTREAM_TURB2LAMVISCRATIO= {gui_settings.get('FREESTREAM_TURB2LAMVISCRATIO', 4.0)}",
                f"FREESTREAM_INTERMITTENCY= {gui_settings.get('FREESTREAM_INTERMITTENCY', 1.0)}"
            ])

        config_lines.append("")

        vel_x = velocity_magnitude * np.cos(np.radians(aoa))
        vel_y = velocity_magnitude * np.sin(np.radians(aoa))

        config_lines.extend([
            "% -------------------- INCOMPRESSIBLE FREE-STREAM DEFINITION ------------------%",
            f"INC_DENSITY_INIT= {freestream_dens:.6f}",
            f"INC_VELOCITY_INIT= ( {vel_x:.6f}, {vel_y:.6f}, 0.0 )",
            "INC_NONDIM= INITIAL_VALUES",
            f"INC_DENSITY_REF = {freestream_dens:.6f}",
            ""
        ])

        config_lines.extend([
            "% --------------------------- VISCOSITY MODEL ---------------------------------%",
            "VISCOSITY_MODEL= CONSTANT_VISCOSITY"
        ])
        mu_constant = (freestream_dens * velocity_magnitude * 1.0) / reynolds
        config_lines.extend([f"MU_CONSTANT= {mu_constant:.10e}", ""])

        bc_marker = "MARKER_EULER= ( Airfoil )" if 'EULER' in solver else "MARKER_HEATFLUX= ( Airfoil, 0.0 )"
        config_lines.extend([
            "% ----------------------- BOUNDARY CONDITION DEFINITION -----------------------%",
            bc_marker,
            "MARKER_FAR= ( Inlet, Outlet )",
            ""
        ])

    else:
        # ==================== COMPRESSIBLE FLOW ====================
        solver = gui_settings.get('SOLVER', 'RANS')
        turb_model = gui_settings.get('KIND_TURB_MODEL', 'SST' if is_supersonic else 'SA')
        trans_model = gui_settings.get('KIND_TRANS_MODEL', 'NONE')

        config_lines.extend([
            f"SOLVER= {solver}",
            f"KIND_TURB_MODEL= {turb_model}"
        ])

        if is_supersonic and turb_model == 'SST':
            sst_options = gui_settings.get('SST_OPTIONS', 'V1994m')
            config_lines.append(f"SST_OPTIONS= {sst_options}")

        if trans_model != 'NONE':
            config_lines.extend([
                f"FREESTREAM_NU_FACTOR= {gui_settings.get('FREESTREAM_NU_FACTOR', 4.0)}",
                f"KIND_TRANS_MODEL= {trans_model}",
                f"LM_OPTIONS= {gui_settings.get('LM_OPTIONS', '(MENTER_LANGTRY, MENTER_LANGTRY, NONE)')}",
                f"FREESTREAM_TURBULENCEINTENSITY= {gui_settings.get('FREESTREAM_TURBULENCEINTENSITY', 0.10)}",
                f"FREESTREAM_TURB2LAMVISCRATIO= {gui_settings.get('FREESTREAM_TURB2LAMVISCRATIO', 4.0)}",
                f"FREESTREAM_INTERMITTENCY= {gui_settings.get('FREESTREAM_INTERMITTENCY', 1.0)}"
            ])

        config_lines.append("")

        ref_dim = gui_settings.get(
            'REF_DIMENSIONALIZATION',
            'DIMENSIONAL' if (is_transonic or is_supersonic) else 'FREESTREAM_PRESS_EQ_ONE'
        )

        config_lines.extend([
            "% -------------------- COMPRESSIBLE FREE-STREAM DEFINITION --------------------%",
            f"REF_DIMENSIONALIZATION= {ref_dim}",
            f"MACH_NUMBER= {mach:.6f}",
            f"AOA= {aoa:.6f}",
            f"FREESTREAM_TEMPERATURE= {freestream_temp}",
            f"FREESTREAM_PRESSURE= {freestream_pres}",
            f"REYNOLDS_NUMBER= {reynolds:.0f}",
            "REYNOLDS_LENGTH= 1.0",
            ""
        ])

        bc_marker = "MARKER_EULER= ( Airfoil )" if 'EULER' in solver else "MARKER_HEATFLUX= ( Airfoil, 0.0 )"
        config_lines.extend([
            "% ----------------------- BOUNDARY CONDITION DEFINITION -----------------------%",
            bc_marker,
            "MARKER_FAR= ( Inlet, Outlet )",
            ""
        ])

    # ==================== REFERENCE VALUES ====================
    ref_area = gui_settings.get('REF_AREA', 1.0 if is_supersonic else 0 if is_transonic else 1.0)

    config_lines.extend([
        "% ---------------------- REFERENCE VALUE DEFINITION ---------------------------%",
        "REF_ORIGIN_MOMENT_X = 0.25",
        "REF_ORIGIN_MOMENT_Y = 0.00",
        "REF_ORIGIN_MOMENT_Z = 0.00",
        "REF_LENGTH= 1.0",
        f"REF_AREA= {ref_area}",
        ""
    ])

    # ==================== SURFACE MARKERS ====================
    config_lines.extend([
        "% ------------------------ SURFACES IDENTIFICATION ----------------------------%",
        "MARKER_PLOTTING= ( Airfoil )",
        "MARKER_MONITORING= ( Airfoil )",
    ])
    if is_transonic:
        config_lines.append("MARKER_DESIGNING= ( Airfoil )")
    config_lines.append("")

    # ==================== DISCRETIZATION ====================
    if is_transonic:
        config_lines.extend([
            "% DISCRETIZATION",
            "TIME_DOMAIN= YES",
            "TIME_MARCHING= DUAL_TIME_STEPPING-2ND_ORDER",
            f"TIME_STEP= {gui_settings.get('TIME_STEP', 5e-4)}",
            ""
        ])

    # ==================== NUMERICAL METHODS ====================
    config_lines.extend([
        "% ---------------------- NUMERICAL METHODS ------------------------------------%",
        f"NUM_METHOD_GRAD= {gui_settings.get('NUM_METHOD_GRAD', 'WEIGHTED_LEAST_SQUARES')}"
    ])

    conv_method = gui_settings.get('CONV_NUM_METHOD_FLOW', 'ROE' if is_supersonic else 'JST')

    if is_transonic and conv_method == 'JST':
        jst_coeff = gui_settings.get('JST_SENSOR_COEFF', '( 0.5, 0.005 )')
        config_lines.append(f"JST_SENSOR_COEFF= {jst_coeff}")

    config_lines.append(f"CONV_NUM_METHOD_FLOW= {conv_method}")

    is_central = conv_method in ['JST', 'JST_KE', 'JST_MAT', 'LAX_FRIEDRICH']
    if is_supersonic:
        muscl_flow_setting = "YES"
    else:
        muscl_flow_setting = "NO" if is_central else "YES"

    config_lines.append(f"MUSCL_FLOW= {muscl_flow_setting}")

    slope_limiter = gui_settings.get('SLOPE_LIMITER_FLOW', 'VENKATAKRISHNAN')
    config_lines.append(f"SLOPE_LIMITER_FLOW= {slope_limiter}")
    config_lines.append("CONV_NUM_METHOD_TURB= SCALAR_UPWIND")

    muscl_turb = gui_settings.get('MUSCL_TURB', 'NO' if is_supersonic else 'YES')
    config_lines.append(f"MUSCL_TURB= {muscl_turb}")
    config_lines.append("")

    # ==================== SOLUTION METHODS ====================
    config_lines.extend([
        "% SOLUTION METHODS",
        "TIME_DISCRE_FLOW= EULER_IMPLICIT",
        "TIME_DISCRE_TURB= EULER_IMPLICIT"
    ])

    if is_supersonic:
        cfl_number = gui_settings.get('CFL_NUMBER', 0.5)
        cfl_adapt_param = gui_settings.get('CFL_ADAPT_PARAM', '( 0.5, 1.1, 0.5, 15.0 )')
        config_lines.extend([
            f"CFL_NUMBER= {cfl_number}",
            "CFL_ADAPT= YES",
            f"CFL_ADAPT_PARAM= {cfl_adapt_param}"
        ])
    elif is_transonic:
        config_lines.extend([
            f"CFL_NUMBER= {gui_settings.get('CFL_NUMBER', 1.0)}",
            "CFL_ADAPT= YES"
        ])
    else:
        config_lines.extend([
            f"CFL_NUMBER= {gui_settings.get('CFL_NUMBER', 25.0 if is_incompressible else 40.0)}",
            "CFL_ADAPT= NO",
            "RK_ALPHA_COEFF= (0.66667, 0.66667, 1.0)"
        ])
    config_lines.append("")

    # ==================== LINEAR SOLVER ====================
    linear_solver = gui_settings.get('LINEAR_SOLVER', 'FGMRES')
    linear_solver_prec = gui_settings.get('LINEAR_SOLVER_PREC', 'ILU')

    if is_supersonic:
        linear_solver_error = gui_settings.get('LINEAR_SOLVER_ERROR', 1e-4)
        linear_solver_iter = gui_settings.get('LINEAR_SOLVER_ITER', 15)
    elif is_transonic:
        linear_solver_error = gui_settings.get('LINEAR_SOLVER_ERROR', 0.1)
        linear_solver_iter = gui_settings.get('LINEAR_SOLVER_ITER', 50)
    else:
        linear_solver_error = gui_settings.get('LINEAR_SOLVER_ERROR', 1e-8)
        linear_solver_iter = gui_settings.get('LINEAR_SOLVER_ITER', 10)

    config_lines.extend([
        "% ------------------------ LINEAR SOLVER DEFINITION ---------------------------%",
        f"LINEAR_SOLVER= {linear_solver}",
        f"LINEAR_SOLVER_PREC= {linear_solver_prec}",
        f"LINEAR_SOLVER_ERROR= {linear_solver_error}",
        f"LINEAR_SOLVER_ITER= {linear_solver_iter}",
        ""
    ])

    if is_supersonic:
        config_lines.extend([
            "% ------------- NO MULTIGRID (problematic for hybrid) -------------%",
            "MGLEVEL= 0",
            ""
        ])

    # ==================== CONVERGENCE ====================

    # Iterations (GUI may have written both ITER and EXT_ITER)
    iter_val = gui_settings.get('ITER', gui_settings.get('EXT_ITER', 3000))

    # User convergence settings (this is the critical bit)
    user_conv_field = gui_settings.get('CONV_FIELD')
    user_min_val_raw = gui_settings.get('CONV_RESIDUAL_MINVAL', "-8.0")

    if isinstance(user_min_val_raw, str):
        user_min_val = user_min_val_raw.strip()
    else:
        user_min_val = str(user_min_val_raw)

    print(f"[DEBUG] Using CONV_RESIDUAL_MINVAL from GUI: {user_min_val}")

    if is_transonic:
        conv_field = user_conv_field if user_conv_field else "REL_RMS_DENSITY"
        config_lines.extend([
            "% INNER CONVERGENCE",
            f"INNER_ITER= {gui_settings.get('INNER_ITER', 30)}",
            f"CONV_FIELD= {conv_field}",
            f"CONV_RESIDUAL_MINVAL= {user_min_val}",
            "CONV_STARTITER= 0",
            "",
            "% TIME CONVERGENCE",
            f"TIME_ITER= {iter_val}",
            "",
            "WINDOW_CAUCHY_CRIT= YES",
            f"WINDOW_START_ITER= {gui_settings.get('WINDOW_START_ITER', 500)}",
            f"WINDOW_FUNCTION= {gui_settings.get('WINDOW_FUNCTION', 'HANN_SQUARE')}",
            "",
            f"CONV_WINDOW_FIELD= {gui_settings.get('CONV_WINDOW_FIELD', '( TAVG_DRAG, TAVG_LIFT )')}",
            "CONV_WINDOW_STARTITER= 0",
            f"CONV_WINDOW_CAUCHY_EPS= {gui_settings.get('CONV_WINDOW_CAUCHY_EPS', 1e-4)}",
            f"CONV_WINDOW_CAUCHY_ELEMS= {gui_settings.get('CONV_WINDOW_CAUCHY_ELEMS', 10)}",
            ""
        ])
    else:
        if is_supersonic:
            default_field = 'RMS_DENSITY'
            conv_startiter = gui_settings.get('CONV_STARTITER', 25)
        else:
            default_field = 'RMS_PRESSURE' if is_incompressible else 'RMS_DENSITY'
            conv_startiter = gui_settings.get('CONV_STARTITER', 10)

        conv_field = user_conv_field if user_conv_field else default_field

        config_lines.extend([
            "% ------------- CONVERGENCE ------------%",
            f"ITER= {iter_val}",
            f"CONV_FIELD= {conv_field}",
            f"CONV_RESIDUAL_MINVAL= {user_min_val}",
            f"CONV_STARTITER= {conv_startiter}",
            ""
        ])

    # ==================== OUTPUT ====================
    if is_incompressible:
        screen_output = "INNER_ITER, RMS_PRESSURE, RMS_MOMENTUM-X, RMS_MOMENTUM-Y, RMS_ENERGY, LIFT, DRAG, MOMENT"
    elif is_supersonic:
        screen_output = gui_settings.get(
            'SCREEN_OUTPUT',
            "INNER_ITER, RMS_DENSITY, RMS_TKE, RMS_DISSIPATION, LIFT, DRAG"
        )
    elif is_transonic:
        screen_output = (
            "INNER_ITER, RMS_DENSITY, RMS_MOMENTUM-X, RMS_MOMENTUM-Y, "
            "RMS_PRESSURE, LIFT, DRAG, Y_PLUS"
        )
    else:
        screen_output = (
            "INNER_ITER, RMS_DENSITY, RMS_MOMENTUM-X, RMS_MOMENTUM-Y, "
            "RMS_ENERGY, LIFT, DRAG, MOMENT"
        )


    output_files = gui_settings.get('OUTPUT_FILES', '(RESTART, TECPLOT_ASCII, PARAVIEW, SURFACE_CSV)')


    config_lines.extend([
        "% ------------------------- INPUT/OUTPUT INFORMATION --------------------------%",
        f"OUTPUT_FILES= {output_files}",
        f"MESH_FILENAME= {mesh_filename}",
        "MESH_FORMAT= SU2",
        "MESH_OUT_FILENAME= mesh_out.su2",
        "SOLUTION_FILENAME= solution_flow.dat",
    ])

    if is_transonic:
        config_lines.append("SOLUTION_ADJ_FILENAME= solution_adj.dat")

    config_lines.append("TABULAR_FORMAT= CSV")

    conv_filename = f"history_aoa_{aoa:.2f}"
    restart_filename = "restart_flow.dat"

    config_lines.extend([
        f"CONV_FILENAME= {conv_filename}",
        f"RESTART_FILENAME= {restart_filename}",
    ])

    if is_transonic:
        config_lines.append("RESTART_ADJ_FILENAME= restart_adj.dat")

    volume_filename = f"flow_aoa_{aoa:.2f}"
    config_lines.append(f"VOLUME_FILENAME= {volume_filename}")

    if is_transonic:
        config_lines.extend([
            "VOLUME_ADJ_FILENAME= adjoint",
            "GRAD_OBJFUNC_FILENAME= of_grad.dat"
        ])

    surface_filename = f"surf_aoa_{aoa:.2f}"
    config_lines.append(f"SURFACE_FILENAME= {surface_filename}")

    if is_transonic:
        config_lines.append("SURFACE_ADJ_FILENAME= surface_adjoint")

    history_output = gui_settings.get('HISTORY_OUTPUT', '(ITER, RMS_RES, AERO_COEFF, AOA)')
    config_lines.extend([
        f"SCREEN_OUTPUT = ({screen_output})",
        f"HISTORY_OUTPUT = {history_output}",
        ""
    ])

    final_lines = []
    prev_empty = False
    for line in config_lines:
        if line == "":
            if not prev_empty:
                final_lines.append(line)
                prev_empty = True
        else:
            final_lines.append(line)
            prev_empty = False

    config_filename = f"config_aoa_{aoa:.2f}.cfg"
    config_path = os.path.join(output_dir, config_filename)

    try:
        with open(config_path, 'w') as f:
            f.write('\n'.join(final_lines))
        print(f"Successfully wrote config to {config_path}")
        return config_path
    except Exception as e:
        print(f"Error writing config file {config_path}: {e}")
        return None

def _save_screenshot(
    mesh,
    output_path: str,
    title: str,
    show_edges: bool,
    scalars: Optional[str] = None,
    cmap: str = 'viridis',
    show_scalar_bar: bool = False,
    mesh_color: Optional[str] = None
):
    """Internal helper to create and save a single PyVista screenshot."""
    plotter = pv.Plotter(off_screen=True, window_size=PLOTTER_WINDOW_SIZE)
    bg_color = COLOR_BG_WHITE if not scalars else COLOR_BG_BLACK
    plotter.set_background(bg_color)

    scalar_args = None
    if show_scalar_bar and scalars:
        scalar_args = {
            'title': scalars.replace('_', ' ').title(),
            'vertical': True,
            'label_font_size': 18,
            'title_font_size': 24
        }

    plotter.add_mesh(
        mesh,
        scalars=scalars,
        cmap=cmap,
        show_edges=show_edges,
        color=mesh_color,
        scalar_bar_args=scalar_args,
    )

    plotter.camera_position = FIXED_CAMERA_POSITION
    plotter.add_title(title, font_size=30, color='black' if not scalars else 'white')
    plotter.show_axes()
    plotter.screenshot(output_path, window_size=SCREENSHOT_WINDOW_SIZE)
    
    try:
        plotter.close()
        plotter.deep_clean()
    except Exception as e:
        print(f"Warning: Error closing PyVista plotter: {e}")
    
    try:
        print(f"Saved screenshot for '{title}' to: {output_path}")
    except Exception as e:
        print(f"Warning: Error printing screenshot confirmation: {e}")

def save_cp_vs_chord_plot(aoa: float, output_folder: str):
    """
    Reads surface data from Tecplot ASCII (.dat) file and saves a Cp vs. x/c plot.
    Uses surface CSV file to identify airfoil surface coordinates.
    """
    
    formatted_aoa = f"{aoa:.2f}"
    csv_pattern = os.path.join(output_folder, f"surf_aoa_{formatted_aoa}_*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        csv_pattern_fallback = os.path.join(output_folder, f"surf_aoa_{formatted_aoa}.csv")
        if os.path.exists(csv_pattern_fallback):
            csv_files = [csv_pattern_fallback]
    
    if not csv_files:
        print(f"ERROR: No surface CSV files found in {output_folder}")
        print(f"Searched pattern: surf_aoa_{formatted_aoa}_*.csv")
        return
    
    def extract_iteration(filepath):
        match = re.search(r'_(\d+)\.csv$', filepath)
        return int(match.group(1)) if match else 0
    
    csv_files.sort(key=extract_iteration, reverse=True)
    csv_file = csv_files[0]
    
    print(f"Found surface CSV file: {os.path.basename(csv_file)}")
    if len(csv_files) > 1:
        print(f"  (Using latest iteration: {len(csv_files)} files found)")
    
    try:
        surface_df = pd.read_csv(csv_file)
        surface_df.columns = surface_df.columns.str.strip().str.replace('"', '').str.replace("'", "")
        
        x_col_csv = None
        y_col_csv = None
        for col in surface_df.columns:
            col_lower = col.lower()
            if col_lower in ['x']:
                x_col_csv = col
            elif col_lower in ['y']:
                y_col_csv = col
        
        if not x_col_csv or not y_col_csv:
            print(f"ERROR: Could not find x, y coordinates in surface CSV")
            print(f"Available columns: {surface_df.columns.tolist()}")
            return
        
        surface_x = pd.to_numeric(surface_df[x_col_csv], errors='coerce').values
        surface_y = pd.to_numeric(surface_df[y_col_csv], errors='coerce').values
        
        valid_mask = ~(np.isnan(surface_x) | np.isnan(surface_y))
        surface_x = surface_x[valid_mask]
        surface_y = surface_y[valid_mask]
        
        print(f"Found {len(surface_x)} surface points from CSV")
        
    except Exception as e:
        print(f"ERROR reading surface CSV: {e}")
        import traceback
        traceback.print_exc()
        return
    
    dat_pattern = os.path.join(output_folder, "*.dat")
    dat_files = glob.glob(dat_pattern)
    
    if not dat_files:
        print(f"ERROR: No .dat files found in {output_folder}")
        return
    
    dat_files.sort(key=os.path.getmtime, reverse=True)
    file_to_read = dat_files[0]
    
    print(f"Found Tecplot ASCII file: {os.path.basename(file_to_read)}")

    try:
        df = parse_tecplot_ascii(file_to_read)
        
        if df.empty:
            print(f"ERROR: No data parsed from {os.path.basename(file_to_read)}")
            return
        
        def get_col_name(candidates):
            for col in df.columns:
                c_clean = col.strip().replace('"', '').replace("'", "").lower()
                for cand in candidates:
                    if cand.lower() == c_clean or cand.lower() in c_clean:
                        return col
            return None

        x_col = get_col_name(['x'])
        y_col = get_col_name(['y'])
        cp_col = get_col_name(['pressure_coefficient'])

        if not x_col or not y_col or not cp_col:
            print(f"ERROR: Could not identify required columns in {os.path.basename(file_to_read)}")
            print(f"Available columns: {df.columns.tolist()}")
            return

        df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
        df[cp_col] = pd.to_numeric(df[cp_col], errors='coerce')
        df = df.dropna(subset=[x_col, y_col, cp_col])
        
        print(f"DAT file has {len(df)} valid data points")
        print(f"Surface x range: [{surface_x.min():.6f}, {surface_x.max():.6f}]")
        print(f"DAT x range: [{df[x_col].min():.6f}, {df[x_col].max():.6f}]")
        
        tolerance = 1e-6
        
        matched_data = []
        for sx, sy in zip(surface_x, surface_y):
            mask = (np.abs(df[x_col] - sx) < tolerance) & (np.abs(df[y_col] - sy) < tolerance)
            matches = df[mask]
            
            if len(matches) > 0:
                matched_data.append({
                    'x': sx,
                    'y': sy,
                    'cp': matches.iloc[0][cp_col]
                })
        
        print(f"Matched {len(matched_data)} points")
        
        if not matched_data:
            print("ERROR: No matching points found between surface CSV and DAT file")
            print("Trying with larger tolerance...")
            tolerance = 1e-4
            for sx, sy in zip(surface_x[:10], surface_y[:10]):
                mask = (np.abs(df[x_col] - sx) < tolerance) & (np.abs(df[y_col] - sy) < tolerance)
                matches = df[mask]
                print(f"  Point ({sx:.6f}, {sy:.6f}): {len(matches)} matches")
            return
        
        matched_df = pd.DataFrame(matched_data)
        
        max_x = matched_df['x'].max()
        min_x = matched_df['x'].min()
        chord_length = max_x - min_x
        
        if chord_length > 0.001:
            matched_df['x_norm'] = (matched_df['x'] - min_x) / chord_length
        else:
            matched_df['x_norm'] = matched_df['x']
        
        upper_surface = matched_df[matched_df['y'] >= 0].copy().sort_values('x')
        lower_surface = matched_df[matched_df['y'] < 0].copy().sort_values('x')
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if len(upper_surface) > 0:
                ax.scatter(upper_surface['x_norm'], upper_surface['cp'], color='r', label='Upper Surface', s=20)
            if len(lower_surface) > 0:
                ax.scatter(lower_surface['x_norm'], lower_surface['cp'], color='b', label='Lower Surface', s=20)
            
            if len(upper_surface) > 0 and len(lower_surface) > 0:
                ax.legend(fontsize=12)
            
            ax.set_title(f'$C_p$ Distribution | AoA = {aoa:.2f}°', fontsize=16)
            ax.set_xlabel('Normalized Chord (x/c)', fontsize=14)
            ax.set_ylabel('Pressure Coefficient (Cp)', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.invert_yaxis()
            ax.set_xlim(0, 1)

            output_filename = os.path.join(output_folder, f"cp_vs_chord_aoa_{aoa:.2f}.png")
            plt.savefig(output_filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved Cp plot to: {output_filename}")
            
        except Exception as plot_error:
            print(f"ERROR creating Cp plot: {plot_error}")
            import traceback
            traceback.print_exc()
            plt.close('all')

    except Exception as e:
        print(f"ERROR during Cp plot generation: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')


def parse_tecplot_ascii(filepath: str) -> pd.DataFrame:
    """
    Parses a Tecplot ASCII (.dat) file and returns a pandas DataFrame.
    Handles the VARIABLES and ZONE headers specific to Tecplot format.
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    

    variables = []
    data_start_idx = 0
    variables_complete = False
    
    for i, line in enumerate(lines):
        line_upper = line.strip().upper()
        
        if 'VARIABLES' in line_upper or (variables and not variables_complete):
            var_line = line.strip()
            if 'VARIABLES' in var_line.upper():
                var_line = re.sub(r'VARIABLES\s*=\s*', '', var_line, flags=re.IGNORECASE)
        
            quoted_vars = re.findall(r'"([^"]+)"', var_line)
            variables.extend(quoted_vars)
            
            if var_line.rstrip().endswith(')') or not var_line.rstrip().endswith(','):
                variables_complete = True
                
        elif line_upper.startswith('ZONE'):
            data_start_idx = i + 1
            break
        elif line_upper.startswith('TITLE'):
            continue
    
    if not variables:
        print("Warning: Could not find VARIABLES declaration in Tecplot file.")
        print("First few lines of file:")
        for line in lines[:10]:
            print(f"  {line.rstrip()}")
    
    # Read numerical data
    data_lines = []
    for line in lines[data_start_idx:]:
        line = line.strip()
        if not line or line.startswith('#') or any(kw in line.upper() for kw in ['TITLE', 'VARIABLES', 'ZONE', 'AUXDATA']):
            continue
        try:
            values = [float(x) for x in line.split()]
            if values: 
                data_lines.append(values)
        except (ValueError, IndexError):
            continue
    
    if not data_lines:
        print("ERROR: No numerical data found in Tecplot file!")
        return pd.DataFrame()
    
    num_cols = len(data_lines[0])
    
    if variables and len(variables) == num_cols:
        df = pd.DataFrame(data_lines, columns=variables)
        #print(f"Successfully parsed {len(data_lines)} data rows with {num_cols} columns")
    else:
        if variables:
            print(f"Warning: Variable count mismatch. Expected {len(variables)}, got {num_cols} data columns")
        df = pd.DataFrame(data_lines, columns=[f'Col_{i}' for i in range(num_cols)])
        print(f"Using generic column names for {num_cols} columns")
    
    #print(f"Available columns: {df.columns.tolist()}"
    return df

def find_column(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    """Find a DataFrame column matching any of the keywords (case-insensitive)."""
    for col in df.columns:
        col_upper = col.upper()
        if any(kw in col_upper for kw in keywords):
            return col
    return None


def find_and_visualize_vtk_file(output_dir: str):
    """Recursively searches for VTK/VTU files and visualizes the most recent one."""
    pyvista_logger = logging.getLogger('pyvista')
    pyvista_logger.setLevel(logging.ERROR)

    print(f"\n{'=' * 60}")
    print(f"Recursively searching for flow files in: {output_dir}")
    print(f"{'=' * 60}")

    vtk_paths = glob.glob(os.path.join(output_dir, '**', '*.vtk'), recursive=True)
    vtu_paths = glob.glob(os.path.join(output_dir, '**', '*.vtu'), recursive=True)
    all_files = vtk_paths + vtu_paths

    if not all_files:
        print("-" * 60)
        print("ERROR: No .vtk or .vtu files found after simulation run.")
        print(f"Verified directory and subdirectories: {output_dir}")
        print("-" * 60)
        return

    all_files.sort(key=os.path.getmtime, reverse=True)
    vtk_file_path = all_files[0]

    print("\n--- Found VTK/VTU Files ---")
    for i, path in enumerate(all_files):
        relative_path = os.path.relpath(path, output_dir)
        is_latest = "(LATEST - Will be visualized)" if i == 0 else ""
        print(f"[{i + 1}] {relative_path} {is_latest}")
    print("-" * 60)

    print(f"\nSelected file for visualization: {os.path.basename(vtk_file_path)}")
    print("Generating PyVista screenshots for geometry, domain, and scalars...")
    print("-" * 60)

    try:
        mesh = pv.read(vtk_file_path)
        output_folder = os.path.dirname(vtk_file_path)
        base_filename = os.path.splitext(os.path.basename(vtk_file_path))[0]
        folder_name = os.path.basename(output_folder)
        plot_title_base = f"AoA: {folder_name}"

        # 1. Geometry View
        print("\n[1/3] Generating Geometry View...")
        surface_mesh = mesh.extract_surface()
        geometry_output_filename = os.path.join(output_folder, f"{base_filename}_Geometry.png")
        _save_screenshot(
            surface_mesh,
            geometry_output_filename,
            f"{plot_title_base} | Geometry View",
            show_edges=False,
            scalars=None,
            mesh_color=COLOR_GEOMETRY,
        )

        # 2. Domain Mesh View
        print("\n[2/3] Generating Domain Mesh View...")
        domain_output_filename = os.path.join(output_folder, f"{base_filename}_DomainMesh.png")
        _save_screenshot(
            mesh.extract_all_edges(),
            domain_output_filename,
            f"{plot_title_base} | Domain Mesh Edges",
            show_edges=False,
            scalars=None,
            mesh_color=COLOR_MESH_EDGES,
        )

        print("\n[3/3] Generating Scalar Field Visualizations...")
        scalar_arrays = list(mesh.point_data.keys()) + list(mesh.cell_data.keys())
        common_names = ['Pressure', 'Mach', 'Density', 'Velocity', 'vorticity_magnitude', 'Intermittency', 'Mach']
        scalars_to_plot = []

        for name in common_names:
            if name in scalar_arrays:
                scalars_to_plot.append(name)

        if not scalars_to_plot and scalar_arrays:
            scalars_to_plot.extend(scalar_arrays[:3])
            print(f"   Note: Common scalar names not found. Using first {len(scalars_to_plot)} available scalars.")

        if not scalars_to_plot:
            print("   Warning: No scalar arrays found in the VTK/VTU file.")
        else:
            print(f"   Found {len(scalars_to_plot)} scalar field(s) to visualize: {', '.join(scalars_to_plot)}")

        for idx, name in enumerate(scalars_to_plot, 1):
            print(f"   [{idx}/{len(scalars_to_plot)}] Creating plot for: {name}")
            flow_output_filename = os.path.join(output_folder, f"{base_filename}_{name}.png")
            _save_screenshot(
                mesh,
                flow_output_filename,
                f"{plot_title_base} | Scalar: {name.replace('_', ' ').title()}",
                show_edges=False,
                scalars=name,
                cmap='turbo',
                show_scalar_bar=True,
                mesh_color=None
            )

        print("\n" + "=" * 60)
        print("PyVista visualization complete!")
        print(f"All images saved to: {output_folder}")
        print("=" * 60)

    except FileNotFoundError:
        print(f"\n--- Visualization Error ---")
        print(f"VTK file not found: {vtk_file_path}")
        print("-" * 60)
    except ValueError as e:
        print(f"\n--- Visualization Error ---")
        print(f"Invalid data in VTK file: {e}")
        print("-" * 60)
    except Exception as e:
        print(f"\n--- PyVista Visualization Error ---")
        print(f"An error occurred: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        print("-" * 60)


def visualize_single_vtk_file(vtk_file_path: str, aoa: float):
    """
    Generate PyVista screenshots (geometry, mesh, scalar fields) for a SINGLE VTK/VTU file.
    Used to post-process every AoA separately.
    """
    pyvista_logger = logging.getLogger('pyvista')
    pyvista_logger.setLevel(logging.ERROR)

    if not os.path.exists(vtk_file_path):
        print(f"[Post] VTK/VTU file not found for AoA {aoa:.2f}: {vtk_file_path}")
        return

    try:
        mesh = pv.read(vtk_file_path)
        output_folder = os.path.dirname(vtk_file_path)
        base_filename = os.path.splitext(os.path.basename(vtk_file_path))[0]

        plot_title_base = f"AoA = {aoa:.2f} deg"
        print(f"[Post] AoA {aoa:.2f}: Geometry view...")
        surface_mesh = mesh.extract_surface()
        geometry_output_filename = os.path.join(output_folder, f"{base_filename}_Geometry.png")
        _save_screenshot(
            surface_mesh,
            geometry_output_filename,
            f"{plot_title_base} | Geometry View",
            show_edges=False,
            scalars=None,
            mesh_color=COLOR_GEOMETRY,
        )
        print(f"[Post] AoA {aoa:.2f}: Domain mesh...")
        domain_output_filename = os.path.join(output_folder, f"{base_filename}_DomainMesh.png")
        _save_screenshot(
            mesh.extract_all_edges(),
            domain_output_filename,
            f"{plot_title_base} | Domain Mesh Edges",
            show_edges=False,
            scalars=None,
            mesh_color=COLOR_MESH_EDGES,
        )
        print(f"[Post] AoA {aoa:.2f}: Scalar fields...")
        scalar_arrays = list(mesh.point_data.keys()) + list(mesh.cell_data.keys())
        common_names = ['Pressure', 'Mach', 'Density', 'Velocity',
                        'vorticity_magnitude', 'Intermittency']
        scalars_to_plot = []

        for name in common_names:
            if name in scalar_arrays:
                scalars_to_plot.append(name)

        if not scalars_to_plot and scalar_arrays:
            scalars_to_plot.extend(scalar_arrays[:3])
            print(f"   Note: common scalar names not found. Using first {len(scalars_to_plot)} arrays.")

        if not scalars_to_plot:
            print("   Warning: no scalar arrays found in the VTK/VTU file.")
        else:
            for idx, name in enumerate(scalars_to_plot, 1):
                print(f"   [{idx}/{len(scalars_to_plot)}] {name}")
                flow_output_filename = os.path.join(output_folder, f"{base_filename}_{name}.png")
                _save_screenshot(
                    mesh,
                    flow_output_filename,
                    f"{plot_title_base} | Scalar: {name.replace('_', ' ').title()}",
                    show_edges=False,
                    scalars=name,
                    cmap='turbo',
                    show_scalar_bar=True,
                    mesh_color=None
                )

        print(f"[Post] AoA {aoa:.2f}: PyVista screenshots saved in {output_folder}")

    except Exception as e:
        print(f"[Post] Error visualizing AoA {aoa:.2f} ({vtk_file_path}): {e}")


def execute_su2_analysis_workflow(
    su2_runner: SU2Runner,
    reynolds: float,
    mach: float,
    alpha_min: float,
    alpha_max: float,
    alpha_step: float,
    base_output_dir: str,
    flow_regime: str,
    gui_settings: dict,
    mesh_filepath: str,
    gui_update_callback,
    polar_filename: str = "aerodynamic_polar.csv",
    pv_filename: str = "pv.csv",
    enable_live_plotting: bool = False,
    plot_window_callback=None
):
    print("Starting SU2 analysis workflow...")
    su2_runner.stop_requested = False
    all_sim_results = []

    is_transonic = flow_regime.lower() == "compressible" and 0.7 <= mach < 1.0
    is_unsteady = is_transonic

    current_run_timestamp_dir = os.path.join(
        base_output_dir,
        f"SU2_Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(current_run_timestamp_dir, exist_ok=True)
    aoa_values = (
        np.arange(alpha_min, alpha_max + (alpha_step / 2.0), alpha_step)
        if alpha_step > 0 else [alpha_min]
    )
    mesh_filename_for_su2 = os.path.basename(mesh_filepath)

    for aoa in aoa_values:
        if su2_runner.stop_requested:
            print("Global STOP requested. Aborting remaining AoA runs.")
            break

        data_queue = stop_event = plot_window = None

        if enable_live_plotting and plot_window_callback and data_reader_thread:
            data_queue = Queue()
            stop_event = threading.Event()
            plot_window = plot_window_callback(data_queue, aoa, is_unsteady)
            time.sleep(0.5)

        aoa_run_dir = os.path.join(current_run_timestamp_dir, f"AoA_{aoa:.2f}")
        os.makedirs(aoa_run_dir, exist_ok=True)
        try:
            shutil.copy2(mesh_filepath, os.path.join(aoa_run_dir, mesh_filename_for_su2))
        except Exception as e:
            print(f"Error copying mesh file: {e}")
            continue

        config_path = prepare_su2_config(
            aoa_run_dir, gui_settings, mesh_filename_for_su2,
            flow_regime, aoa, reynolds, mach
        )
        if config_path:
            status, history, solution, surface = su2_runner.run_analysis(
                config_path, aoa, aoa_run_dir, data_queue, stop_event
            )

            if status == "COMPLETED":
                if history and os.path.exists(history):
                    all_sim_results.append((aoa, history, solution, surface))

                    if surface and os.path.exists(surface):
                        save_cp_vs_chord_plot(aoa, aoa_run_dir)
                    else:
                        print(f"[Post] Surface CSV missing for AoA {aoa:.2f}, skipping Cp plot.")

    
                    if solution and os.path.exists(solution):
                        try:
                            visualize_single_vtk_file(solution, aoa)
                        except Exception as e:
                            print(f"[Post] PyVista vis failed for AoA {aoa:.2f}: {e}")
                    else:
                        print(f"[Post] Volume solution missing for AoA {aoa:.2f}, skipping PyVista vis.")

                else:
                    print(f"SU2 run for AoA {aoa} completed but history file is missing.")
            elif status == "STOPPED":
                print("Stop signal detected in workflow. Halting AoA sweep.")
                break
            else:
                print(f"SU2 run for AoA {aoa} failed. Moving to next AoA.")


    print("\nAll SU2 simulations complete or were stopped.")

    aoa_data, cl_data, cd_data, _ = extract_su2_polar_data(all_sim_results)
    if not polar_filename.lower().endswith('.csv'):
        polar_filename += '.csv'
    polar_csv_path = os.path.join(current_run_timestamp_dir, polar_filename)
    save_polar_data_to_csv(polar_csv_path, aoa_data, cl_data, cd_data)

    if gui_update_callback:
        gui_update_callback(all_sim_results)
    return all_sim_results


def extract_su2_polar_data(results_list):
    """
    Extracts final AoA, Cl, Cd, and Cm from a list of completed simulation results.
    """
    AoA_values = [res[0] for res in results_list]

    Cl_values, Cd_values, Cm_values = [], [], []
    for aoa, history_file, _, _ in results_list:
        try:
            if history_file and os.path.exists(history_file):
                df = pd.read_csv(history_file)
                df.columns = df.columns.str.strip()
                if not df.empty:
                    last_row = df.dropna().iloc[-1]
                    cl_col = next((c for c in df.columns if 'LIFT' in c.upper() or 'CL' in c.upper()), None)
                    cd_col = next((c for c in df.columns if 'DRAG' in c.upper() or 'CD' in c.upper()), None)
                    cm_col = next((c for c in df.columns if 'MOMENT' in c.upper() or 'CM' in c.upper()), None)
                    Cl_values.append(last_row[cl_col] if cl_col else np.nan)
                    Cd_values.append(last_row[cd_col] if cd_col else np.nan)
                    Cm_values.append(last_row[cm_col] if cm_col else np.nan)
                else:
                    Cl_values.append(np.nan)
                    Cd_values.append(np.nan)
                    Cm_values.append(np.nan)
            else:
                Cl_values.append(np.nan)
                Cd_values.append(np.nan)
                Cm_values.append(np.nan)
        except Exception as e:
            print(f"Error reading {history_file}: {e}")
            Cl_values.append(np.nan)
            Cd_values.append(np.nan)
            Cm_values.append(np.nan)
    return AoA_values, Cl_values, Cd_values, Cm_values


def save_polar_data_to_csv(filepath: str, aoa_data: list, cl_data: list, cd_data: list):
    """
    Saves aerodynamic polar data (AoA, Cl, Cd) to a CSV file.
    """
    if not any(aoa_data):
        print("No successful simulation results to save.")
        return
    data_dict = {'AoA': aoa_data, 'Cl': cl_data, 'Cd': cd_data}
    df = pd.DataFrame(data_dict)
    try:
        df.to_csv(filepath, index=False)
        print(f"Polar data successfully saved to {filepath}")
    except Exception as e:
        print(f"Error saving polar data to {filepath}: {e}")
