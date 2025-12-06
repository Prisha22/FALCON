import os
import subprocess
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Label, ttk, Button, Frame

# Global variable to hold the running process
CURRENT_XFOIL_PROCESS = None


def stop_xfoil():
    """Kills the current XFOIL process if it is running."""
    global CURRENT_XFOIL_PROCESS
    if CURRENT_XFOIL_PROCESS is not None:
        print("[XFOIL] Stop signal received. Killing process...")
        try:
            CURRENT_XFOIL_PROCESS.kill()
        except Exception as e:
            print(f"[XFOIL] Error killing process: {e}")
        CURRENT_XFOIL_PROCESS = None


def clean_airfoil_geometry(input_path, output_path):
    """
    Reads airfoil coordinates, removes duplicates, fixes order,
    and saves to a clean file for XFOIL.
    """
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
        if len(coords) < 10:
            raise ValueError("Too few points in airfoil file.")

        # Sort Points (TE Top -> LE -> TE Bottom)
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

        # Remove Duplicates
        mask = np.ones(len(clean_coords), dtype=bool)
        for i in range(1, len(clean_coords)):
            dist = np.linalg.norm(clean_coords[i] - clean_coords[i - 1])
            if dist < 1e-6:
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


# FIXED ARGUMENT ORDER BELOW: xfoil_path FIRST, frame SECOND
def perform_xfoil_analysis(xfoil_path, xfoil_result_frame, int_path,
                           airfoil_full_path, Re, alpha_max, alpha_min, alpha_step, M):
    global CURRENT_XFOIL_PROCESS

    # 1. Clear previous GUI content
    for widget in xfoil_result_frame.winfo_children():
        widget.destroy()

    # 2. Add STOP Button
    control_frame = Frame(xfoil_result_frame)
    control_frame.pack(side="top", fill="x", padx=5, pady=5)

    lbl_status = Label(control_frame, text="Running XFOIL Sweep...", fg="blue")
    lbl_status.pack(side="left")

    btn_stop = ttk.Button(control_frame, text="STOP ANALYSIS", command=stop_xfoil)
    btn_stop.pack(side="right")

    fig_polar = None
    fig_cp = None

    # File Setup
    run_airfoil = "run_airfoil.dat"
    polar_file = "polar_output.txt"
    input_file = "xfoil_input.txt"

    run_airfoil_path = os.path.join(int_path, run_airfoil)
    polar_path = os.path.join(int_path, polar_file)
    input_file_path = os.path.join(int_path, input_file)

    try:
        # --- Sanity Checks ---
        if M >= 1.0:
            raise ValueError(f"Mach {M} is too high for XFOIL (Max ~0.85).")
        if not os.path.isfile(xfoil_path):
            raise FileNotFoundError(f"XFoil exe not found: {xfoil_path}")

        os.makedirs(int_path, exist_ok=True)

        # Cleanup ALL old CP files
        try:
            for f in os.listdir(int_path):
                if f.startswith("cp_") or f == polar_file or f == input_file:
                    try:
                        os.remove(os.path.join(int_path, f))
                    except:
                        pass
        except:
            pass

        # --- STEP 1: Clean Geometry ---
        success = clean_airfoil_geometry(airfoil_full_path, run_airfoil_path)
        if not success:
            shutil.copy(airfoil_full_path, run_airfoil_path)

        print(f"[XFOIL] Running Analysis in: {int_path}")
        print(f"[XFOIL] Re={Re}, M={M}")

        # --- STEP 2: Create List of All Angles ---
        if alpha_step <= 0: alpha_step = 1.0
        alphas = np.arange(alpha_min, alpha_max + 0.0001, alpha_step)
        cp_files = []

        # --- STEP 3: Create Input Commands ---
        commands = [
            "PLOP", "G", "",  # Disable Graphics
            f"LOAD {run_airfoil}",
            "PANE",  # Smooth Geometry
            "OPER",
            f"Visc {Re}",
            f"Mach {M}",
            "ITER 200",  # Max iterations
            "PACC",  # Start Polar Output
            polar_file,
            "",
        ]

        # LOOP THROUGH EVERY ANGLE MANUALLY
        for i, alpha in enumerate(alphas):
            fname = f"cp_{i}.dat"
            commands.append(f"ALFA {alpha:.2f}")
            commands.append(f"CPWR {fname}")
            cp_files.append((alpha, fname))

        commands.append("PACC")  # Stop Polar
        commands.append("QUIT")

        with open(input_file_path, 'w') as f:
            f.write("\n".join(commands))

        # --- STEP 4: Run XFOIL ---
        with open(input_file_path, 'r') as input_f:
            CURRENT_XFOIL_PROCESS = subprocess.Popen(
                [xfoil_path],
                stdin=input_f,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=int_path,
                text=True
            )

            try:
                stdout, stderr = CURRENT_XFOIL_PROCESS.communicate()
            except Exception as e:
                if CURRENT_XFOIL_PROCESS is None:
                    raise RuntimeError("Analysis stopped by user.")
                raise e

            if CURRENT_XFOIL_PROCESS.returncode != 0:
                if CURRENT_XFOIL_PROCESS is None:
                    raise RuntimeError("Analysis stopped by user.")

        CURRENT_XFOIL_PROCESS = None
        control_frame.destroy()

        # --- STEP 5: Create Tabs ---
        notebook = ttk.Notebook(xfoil_result_frame)
        notebook.pack(fill='both', expand=True)

        tab_polar = ttk.Frame(notebook)
        tab_cp = ttk.Frame(notebook)

        notebook.add(tab_polar, text='  Drag Polar  ')
        notebook.add(tab_cp, text='  Cp Distribution  ')

        # --- STEP 6: Parse Polar Data ---
        polar_data = []
        if os.path.exists(polar_path):
            with open(polar_path, 'r') as f:
                for line in f:
                    try:
                        parts = line.split()
                        nums = [float(p) for p in parts]
                        if len(nums) >= 5:
                            polar_data.append(nums)
                    except ValueError:
                        continue

        # --- STEP 7: Plot Polar ---
        fig_polar, ax1 = plt.subplots(figsize=(5, 4))
        if polar_data:
            polar_data = np.array(polar_data)
            cls = polar_data[:, 1]
            cds = polar_data[:, 2]
            ax1.plot(cds, cls, 'o-', color='blue', markersize=4, label=f'Re={Re:.1e}')
            ax1.legend()
            ax1.grid(True)
        else:
            ax1.text(0.5, 0.5, "Convergence Failed (No Data)", ha='center', color='red')

        ax1.set_title("Drag Polar")
        ax1.set_xlabel("Cd")
        ax1.set_ylabel("Cl")

        plt.tight_layout()
        canvas_polar = FigureCanvasTkAgg(fig_polar, master=tab_polar)
        canvas_polar.get_tk_widget().pack(fill='both', expand=True)
        canvas_polar.draw()

        # --- STEP 8: Plot Cp Distribution (ALL ANGLES) ---
        fig_cp, ax2 = plt.subplots(figsize=(5, 4))
        has_cp = False

        # Create a color map (Rainbow)
        colormap = cm.get_cmap('jet')
        num_files = len(cp_files)

        for idx, (alpha, fname) in enumerate(cp_files):
            fpath = os.path.join(int_path, fname)
            if os.path.exists(fpath):
                try:
                    cp_x, cp_y = [], []
                    with open(fpath, 'r') as f:
                        lines = f.readlines()
                    for line in lines[3:]:  # Skip XFOIL Header
                        parts = line.split()
                        if len(parts) >= 3:
                            cp_x.append(float(parts[0]))
                            cp_y.append(float(parts[2]))

                    if cp_x:
                        # Pick color based on position in loop
                        c = colormap(idx / max(1, num_files - 1))
                        ax2.plot(cp_x, cp_y, '-', color=c, linewidth=1, label=f"{alpha:.1f}")
                        has_cp = True
                except:
                    pass

        ax2.set_title("Cp Distribution (All Angles)")
        ax2.set_xlabel("x/c")
        ax2.set_ylabel("Cp")
        ax2.grid(True)

        if has_cp:
            ax2.invert_yaxis()
            # If too many legends, shrink font or hide
            if num_files <= 12:
                ax2.legend(fontsize='x-small', ncol=2)
            # Add colorbar logic could go here, but keeping it simple for Tkinter
        else:
            ax2.text(0.5, 0.5, "No Cp data available", ha='center')

        plt.tight_layout()
        canvas_cp = FigureCanvasTkAgg(fig_cp, master=tab_cp)
        canvas_cp.get_tk_widget().pack(fill='both', expand=True)
        canvas_cp.draw()

        print(
            f"[XFOIL] Success. Plotted {len(polar_data)} polar points and {sum(1 for _, f in cp_files if os.path.exists(os.path.join(int_path, f)))} Cp files.")

    except Exception as e:
        print(f"ERROR in XFOIL Thread: {e}")
        for widget in xfoil_result_frame.winfo_children():
            widget.destroy()
        lbl = Label(xfoil_result_frame, text=f"Error / Stopped:\n{e}", fg="red", justify="left", wraplength=300)
        lbl.pack(fill='both', expand=True)

    finally:
        if fig_polar: plt.close(fig_polar)
        if fig_cp: plt.close(fig_cp)
        CURRENT_XFOIL_PROCESS = None