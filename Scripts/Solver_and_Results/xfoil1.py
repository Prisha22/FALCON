import os
import subprocess
import shutil
import numpy as np
import sys
import time
CURRENT_XFOIL_PROCESS = None


def stop_xfoil():
    """Kills the current XFOIL process."""
    global CURRENT_XFOIL_PROCESS
    if CURRENT_XFOIL_PROCESS is not None:
        print("[XFOIL] Stop signal received. Killing process...")
        try:
            CURRENT_XFOIL_PROCESS.kill()
        except Exception as e:
            print(f"[XFOIL] Error killing process: {e}")
        CURRENT_XFOIL_PROCESS = None


def clean_airfoil_geometry(input_path, output_path):
    """Clean and standardizes airfoil coordinates."""
    try:
        coords = []
        with open(input_path, 'r') as f:
            lines = f.readlines()

        start_idx = 0
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    float(parts[0])
                    start_idx = i
                    break
                except ValueError:
                    continue

        for line in lines[start_idx:]:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    coords.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    continue

        coords = np.array(coords)
        if len(coords) < 5: return False

        le_idx = np.argmin(coords[:, 0])
        part1 = coords[:le_idx + 1]
        part2 = coords[le_idx:]

        if np.mean(part1[:, 1]) > np.mean(part2[:, 1]):
            upper, lower = part1, part2
        else:
            upper, lower = part2, part1

        upper = upper[np.argsort(upper[:, 0])[::-1]]
        lower = lower[np.argsort(lower[:, 0])]

        clean_coords = np.concatenate((upper, lower[1:]))

        mask = np.ones(len(clean_coords), dtype=bool)
        for i in range(1, len(clean_coords)):
            if np.linalg.norm(clean_coords[i] - clean_coords[i - 1]) < 1e-6:
                mask[i] = False
        clean_coords = clean_coords[mask]

        with open(output_path, 'w') as f:
            f.write("Clean_Airfoil\n")
            for x, y in clean_coords:
                f.write(f" {x:.6f}  {y:.6f}\n")

        print(f"[XFOIL] Geometry cleaned. {len(coords)} -> {len(clean_coords)} points.")
        return True
    except Exception as e:
        print(f"[XFOIL] Geometry cleaning failed: {e}")
        return False


def run_xfoil_logic(xfoil_path, int_path, airfoil_full_path,
                    Re, alpha_max, alpha_min, alpha_step, M,
                    **kwargs):

    global CURRENT_XFOIL_PROCESS

    run_airfoil = "run_airfoil.dat"
    polar_file = "polar_output.txt"
    input_file = "xfoil_input.txt"

    run_airfoil_path = os.path.join(int_path, run_airfoil)
    polar_path = os.path.join(int_path, polar_file)
    input_file_path = os.path.join(int_path, input_file)

    polar_data = []
    cp_files = []

    try:
        if M >= 1.0: raise ValueError(f"Mach {M} is too high for XFOIL (Max ~0.85).")

        # Resolve through PATH first so a bare "xfoil.exe" works after setup.ps1;
        # os.path.exists alone only ever checks the working directory. Resolving
        # to an absolute path also survives the cwd changes made downstream.
        resolved = shutil.which(xfoil_path)
        if resolved is None and os.path.isfile(xfoil_path):
            resolved = os.path.abspath(xfoil_path)
        if resolved is None:
            raise FileNotFoundError(
                f"XFoil exe not found: {xfoil_path}. Either run setup.ps1 to put xfoil.exe "
                f"on PATH, or enter the full path in the 'XFOIL Executable' field."
            )
        xfoil_path = resolved

        os.makedirs(int_path, exist_ok=True)

        if os.path.exists(polar_path): os.remove(polar_path)
        for f in os.listdir(int_path):
            if f.startswith("cp_") and f.endswith(".dat"):
                try:
                    os.remove(os.path.join(int_path, f))
                except:
                    pass

        if not clean_airfoil_geometry(airfoil_full_path, run_airfoil_path):
            shutil.copy(airfoil_full_path, run_airfoil_path)

        print(f"[XFOIL] Running Analysis in: {int_path}")
        print(f"[XFOIL] Re={Re}, M={M}")

        if alpha_step <= 0: alpha_step = 1.0
        alphas = np.arange(alpha_min, alpha_max + 0.0001, alpha_step)

        commands = [
            "PLOP", "G", "",
            f"LOAD {run_airfoil}",
            "PANE",
            "OPER",
            f"Visc {Re}",
            f"Mach {M}",
            "ITER 100",
            "PACC",
            polar_file,
            "",
        ]

        for i, alpha in enumerate(alphas):
            fname = f"cp_{i}.dat"
            commands.append(f"ALFA {alpha:.2f}")
            commands.append(f"CPWR {fname}")
            cp_files.append((alpha, fname))

        commands.append("PACC")
        commands.append("")
        commands.append("QUIT")

        with open(input_file_path, 'w') as f:
            f.write("\n".join(commands))

        # EXECUTE
        with open(input_file_path, 'r') as input_f:
            CURRENT_XFOIL_PROCESS = subprocess.Popen(
                [xfoil_path],
                stdin=input_f,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Deadlock Fix
                cwd=int_path,
                text=True,
                bufsize=1
            )

            while True:
                line = CURRENT_XFOIL_PROCESS.stdout.readline()
                if not line and CURRENT_XFOIL_PROCESS.poll() is not None:
                    break
                pass

        CURRENT_XFOIL_PROCESS = None


        if os.path.exists(polar_path):
            with open(polar_path, 'r') as f:
                for line in f:
                    try:
                        nums = [float(p) for p in line.split()]
                        if len(nums) >= 5: polar_data.append(nums)
                    except ValueError:
                        continue

        valid_cp = []
        for alpha, fname in cp_files:
            if os.path.exists(os.path.join(int_path, fname)):
                valid_cp.append((alpha, fname))

        cp_files = valid_cp
        print(f"[XFOIL] Success. Generated {len(polar_data)} polar points and {len(cp_files)} Cp files.")

    except Exception as e:
        print(f"[XFOIL ERROR] {e}")

    return polar_data, cp_files