import os, sys
import tkinter as tk
from tkinter import Menu, Label, Entry, Button, ttk, messagebox
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import numpy as np
from queue import Queue
import multiprocessing
import re

# Local imports
from su2_analyzer import (
    SU2Runner,
    execute_su2_analysis_workflow,
    extract_su2_polar_data,
    SU2_INCOMPRESSIBLE_SETTINGS,
    SU2_COMPRESSIBLE_SETTINGS
)

from live_plotter import LivePlotterWindow
import read_airfoil
from parsec import Parsec
from cst import CST
from interpolate import Interpolate
from xfoil1 import perform_xfoil_analysis
from hybrid import generate_hybrid
from meshing import generate_mesh

script_dir = os.path.dirname(os.path.abspath(__file__))


class ConvergenceSettings:
    def __init__(self, parent_frame, start_row):
        """
        Integrates directly into the parent frame using grid layout.
        """
        self.parent = parent_frame
        sep = ttk.Separator(parent_frame, orient='horizontal')
        sep.grid(row=start_row, column=0, columnspan=3, sticky="ew", pady=(15, 10))
        header_lbl = ttk.Label(parent_frame, text="Convergence Criteria", font=('Segoe UI', 9, 'bold'))
        header_lbl.grid(row=start_row + 1, column=0, columnspan=3, sticky="w", padx=5, pady=(0, 5))
        ttk.Label(parent_frame, text="Min log10 Residual:").grid(
            row=start_row + 2, column=0, padx=(10, 5), pady=5, sticky="w"
        )
        self.res_min_var = tk.StringVar(value="-8.0")
        ttk.Entry(parent_frame, textvariable=self.res_min_var).grid(
            row=start_row + 2, column=1, columnspan=2, padx=5, pady=5, sticky="ew"
        )
        ttk.Label(parent_frame, text="Max Iterations:").grid(
            row=start_row + 3, column=0, padx=(10, 5), pady=5, sticky="w"
        )
        self.max_iter_var = tk.StringVar(value="5000")
        ttk.Entry(parent_frame, textvariable=self.max_iter_var).grid(
            row=start_row + 3, column=1, columnspan=2, padx=5, pady=5, sticky="ew"
        )

    def toggle_turbulence(self, is_turbulent):
        """Kept for compatibility; no per-equation residuals now."""
        pass

    def get_config_string(self, is_inviscid=False):
        """
        Returns:
            residual_threshold_str, max_iter_str
        """
        return self.res_min_var.get(), self.max_iter_var.get()


def error(foil1, foil2, ycoords):
    rmse1 = np.sqrt(np.mean((foil1 - ycoords) ** 2))
    rmse2 = np.sqrt(np.mean((foil2 - ycoords) ** 2))
    print(f'PARSEC Root Mean Square Error (RMSE): {rmse1}\n')
    print(f'CST Root Mean Square Error (RMSE): {rmse2}\n')
    return ("PARSEC" if rmse1 < rmse2 else "CST"), rmse1, rmse2


class TextRedirector(object):
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        try:
            self.widget.after(0, self.insert_text, str)
        except:
            pass

    def insert_text(self, str):
        try:
            self.widget.configure(state='normal')
            self.widget.insert('end', str, (self.tag,))
            self.widget.configure(state='disabled')
            self.widget.see('end')
        except:
            pass

    def flush(self):
        pass


class App:
    def __init__(self, root):
        self.root = root
        root.title("FALCON : Framework for Airfoil CFD and anaLysis OptimizatioN")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.script_dir = script_dir
        print(f"Working Directory: {script_dir}")
        self.folder_name = r"Airfoil_DAT_Selig"
        self.directory = os.path.join(self.script_dir, self.folder_name)
        self.loaded_cfg_settings = {}
        self.conv_settings = None  # Will be initialized in update_su2_settings_display

        style = ttk.Style(self.root)
        style.theme_use('clam')
        main_paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        main_paned_window.pack(expand=True, fill='both', padx=10, pady=10)

        self.tabControl = ttk.Notebook(main_paned_window)
        main_paned_window.add(self.tabControl, weight=3)

        self.tab1 = ttk.Frame(self.tabControl)
        self.tab2 = ttk.Frame(self.tabControl)
        self.tab3 = ttk.Frame(self.tabControl)
        self.tabControl.add(self.tab1, text='  Airfoil Parameterization  ')
        self.tabControl.add(self.tab2, text='  XFOIL Analysis  ')
        self.tabControl.add(self.tab3, text='  SU2 Analysis  ')

        log_frame = ttk.LabelFrame(main_paned_window, text="Live Log Output")
        main_paned_window.add(log_frame, weight=1)

        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL)
        self.log_text = tk.Text(
            log_frame,
            height=20,
            wrap=tk.WORD,
            state='disabled',
            yscrollcommand=log_scrollbar.set,
            bg="#2B2B2B",
            fg="#F8F8F2"
        )
        log_scrollbar.config(command=self.log_text.yview)

        log_scrollbar.grid(row=0, column=1, sticky='ns')
        self.log_text.grid(row=0, column=0, sticky='nsew')

        self.log_text.tag_configure("stdout", foreground="#F8F8F2")
        self.log_text.tag_configure("stderr", foreground="#FF5555")

        self.stdout_redirector = TextRedirector(self.log_text, "stdout")
        sys.stdout = self.stdout_redirector
        self.stderr_redirector = TextRedirector(self.log_text, "stderr")
        sys.stderr = self.stderr_redirector

        print("--- Log initialized ---")

        self.tab1.grid_columnconfigure(1, weight=1)
        self.tab1.grid_rowconfigure(0, weight=1)
        self.tab1.grid_rowconfigure(1, weight=1)

        self.list_frame = ttk.LabelFrame(self.tab1, text="Select Airfoil", labelanchor="n")
        self.list_frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky='nsew')
        self.list_frame.grid_columnconfigure(0, weight=1)
        self.list_frame.grid_columnconfigure(1, weight=0)
        self.list_frame.grid_rowconfigure(1, weight=1)

        ttk.Label(self.list_frame, text="Search:").grid(
            row=0, column=0, padx=(10, 5), pady=(10, 5), sticky="w"
        )
        self.airfoil_search_var = tk.StringVar()
        self.airfoil_search_entry = ttk.Entry(self.list_frame, textvariable=self.airfoil_search_var)
        self.airfoil_search_entry.grid(
            row=0, column=1, padx=(0, 10), pady=(10, 5), sticky="ew"
        )

        listbox_frame = ttk.Frame(self.list_frame)
        listbox_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="nsew")
        listbox_frame.grid_columnconfigure(0, weight=1)
        listbox_frame.grid_rowconfigure(0, weight=1)

        self.listbox = tk.Listbox(listbox_frame, height=20, width=40)
        self.listbox.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.listbox.configure(yscrollcommand=scrollbar.set)
        if os.path.exists(self.directory):
            self.all_airfoils = [f for f in os.listdir(self.directory) if f.endswith('.dat')]
        else:
            self.all_airfoils = []
            print(f"Warning: Directory {self.directory} not found.")

        self._update_airfoil_listbox()
        self.listbox.bind('<<ListboxSelect>>', self.on_airfoil_change)
        self.airfoil_search_var.trace_add("write", lambda *args: self._update_airfoil_listbox())
        self.input_frame = ttk.LabelFrame(self.list_frame, text="Controls", labelanchor="n")
        self.input_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky='ew')
        self.input_frame.grid_columnconfigure(1, weight=1)
        self.n_points_label = ttk.Label(self.input_frame, text="Desired points:")
        self.n_points_label.grid(row=2, column=0, padx=(10, 5), pady=5, sticky='w')
        self.n_points_entry = ttk.Entry(self.input_frame, width=10)
        self.n_points_entry.grid(row=2, column=1, padx=(0, 10), pady=5, sticky='ew')
        self.n_points_entry.insert(0, "200")

        self.int_path_label = ttk.Label(self.input_frame, text="Output Path:")
        self.int_path_label.grid(row=3, column=0, padx=10, pady=5, sticky='w')
        self.int_path_entry = ttk.Entry(self.input_frame)
        self.int_path_entry.insert(0, os.path.join(os.getcwd(), "output"))
        self.int_path_entry.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky='ew')

        self.open_dir_button = ttk.Button(
            self.input_frame,
            text='Browse...',
            command=lambda: self.open_dir(self.int_path_entry)
        )
        self.open_dir_button.grid(row=3, column=1, padx=(0, 10), pady=5, sticky='e')

        self.analyze_button = ttk.Button(
            self.input_frame,
            text="Analyze & Generate Interpolated Airfoil",
            command=self.analyze,
            style='Accent.TButton'
        )
        self.analyze_button.grid(row=5, column=0, columnspan=2, padx=10, pady=10, sticky='ew')
        style.configure('Accent.TButton', font=('Segoe UI', 10, 'bold'))

        self.canvas_result_frame = ttk.LabelFrame(self.tab1, text="Airfoil Plot", labelanchor="n")
        self.canvas_result_frame.grid(row=0, column=1, padx=(0, 10), pady=(10, 5), sticky='nsew')
        self.canvas_result_frame.grid_rowconfigure(0, weight=1)
        self.canvas_result_frame.grid_columnconfigure(0, weight=1)

        self.result_frame = ttk.LabelFrame(self.tab1, text="Parameterization Results", labelanchor="n")
        self.result_frame.grid(row=1, column=1, padx=(0, 10), pady=(5, 10), sticky='nsew')
        self.result_frame.grid_columnconfigure(0, weight=1)

        # --- Tab 2: XFOIL Analysis ---
        self.tab2.grid_columnconfigure(1, weight=1)
        self.tab2.grid_rowconfigure(0, weight=1)

        self.xfoil_input_frame = ttk.LabelFrame(self.tab2, text="Input Parameters", labelanchor="n")
        self.xfoil_input_frame.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        self.xfoil_input_frame.grid_columnconfigure(1, weight=1)

        self.xfoil_path_label = ttk.Label(self.xfoil_input_frame, text="XFOIL Executable:")
        self.xfoil_path_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.xfoil_path_entry = ttk.Entry(self.xfoil_input_frame, width=40)
        self.xfoil_path_entry.grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 5), sticky='ew')
        self.xfoil_dir_button = ttk.Button(
            self.xfoil_input_frame,
            text="Browse...",
            command=lambda: self.open_file(self.xfoil_path_entry)
        )
        self.xfoil_dir_button.grid(row=1, column=2, padx=(5, 10), pady=(0, 5), sticky='e')

        self.re_label = ttk.Label(self.xfoil_input_frame, text="Reynolds Number:")
        self.re_label.grid(row=2, column=0, padx=10, pady=5, sticky='w')
        self.re_entry = ttk.Entry(self.xfoil_input_frame)
        self.re_entry.insert(0, "1000000")
        self.re_entry.grid(row=2, column=1, columnspan=2, padx=(0, 10), pady=5, sticky='ew')

        self.mach_label = ttk.Label(self.xfoil_input_frame, text="Mach Number:")
        self.mach_label.grid(row=3, column=0, padx=10, pady=5, sticky='w')
        self.mach_entry = ttk.Entry(self.xfoil_input_frame)
        self.mach_entry.insert(0, "0.0")
        self.mach_entry.grid(row=3, column=1, columnspan=2, padx=(0, 10), pady=5, sticky='ew')

        self.alpha_min_label = ttk.Label(self.xfoil_input_frame, text="Minimum Alpha:")
        self.alpha_min_label.grid(row=4, column=0, padx=10, pady=5, sticky='w')
        self.alpha_min_entry = ttk.Entry(self.xfoil_input_frame)
        self.alpha_min_entry.insert(0, "0.0")
        self.alpha_min_entry.grid(row=4, column=1, columnspan=2, padx=(0, 10), pady=5, sticky='ew')

        self.alpha_max_label = ttk.Label(self.xfoil_input_frame, text="Maximum Alpha:")
        self.alpha_max_label.grid(row=5, column=0, padx=10, pady=5, sticky='w')
        self.alpha_max_entry = ttk.Entry(self.xfoil_input_frame)
        self.alpha_max_entry.insert(0, "5.0")
        self.alpha_max_entry.grid(row=5, column=1, columnspan=2, padx=(0, 10), pady=5, sticky='ew')

        self.alpha_step_label = ttk.Label(self.xfoil_input_frame, text="Step:")
        self.alpha_step_label.grid(row=6, column=0, padx=10, pady=5, sticky='w')
        self.alpha_step_entry = ttk.Entry(self.xfoil_input_frame)
        self.alpha_step_entry.insert(0, "1.0")
        self.alpha_step_entry.grid(row=6, column=1, columnspan=2, padx=(0, 10), pady=5, sticky='ew')

        self.xfoil_analyze_button = ttk.Button(
            self.xfoil_input_frame,
            text="Analyze with XFOIL",
            command=self.xfoil_analyze_threaded,
            style='Accent.TButton'
        )
        self.xfoil_analyze_button.grid(row=7, column=0, columnspan=3, padx=10, pady=10, sticky='ew')

        self.xfoil_result_frame = ttk.LabelFrame(self.tab2, text="Polar Plot", labelanchor="n")
        self.xfoil_result_frame.grid(row=0, column=1, padx=(0, 10), pady=10, sticky='nsew')
        self.xfoil_result_frame.grid_rowconfigure(0, weight=1)
        self.xfoil_result_frame.grid_columnconfigure(0, weight=1)

        # --- Tab 3: SU2 Analysis ---
        su2_notebook = ttk.Notebook(self.tab3)
        su2_notebook.pack(expand=True, fill='both', padx=0, pady=0)

        su2_tab_mesh = ttk.Frame(su2_notebook)
        su2_tab_setup = ttk.Frame(su2_notebook)
        su2_tab_results = ttk.Frame(su2_notebook)

        su2_notebook.add(su2_tab_mesh, text='  Meshing & Conditions  ')
        su2_notebook.add(su2_tab_setup, text='  Solver Setup & Run  ')
        su2_notebook.add(su2_tab_results, text='  Results  ')

        # --- "Meshing & Conditions" Sub-Tab ---
        su2_tab_mesh.grid_columnconfigure(0, weight=1)
        su2_tab_mesh.grid_columnconfigure(1, weight=1)
        su2_tab_mesh.grid_rowconfigure(0, weight=1)

        # 1. Mesh Generation
        self.mesh_frame = ttk.LabelFrame(su2_tab_mesh, text="1. Mesh Generation", labelanchor="n")
        self.mesh_frame.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')

        self.structured_mesh_button = ttk.Button(self.mesh_frame, text="Generate Structured Mesh", command=self.mesh)
        self.structured_mesh_button.pack(padx=10, pady=10, fill='x')
        self.hybrid_mesh_button = ttk.Button(self.mesh_frame, text="Generate Hybrid Mesh", command=self.hybrid)
        self.hybrid_mesh_button.pack(padx=10, pady=(0, 10), fill='x')

        y_plus_frame = ttk.Frame(self.mesh_frame)
        y_plus_frame.pack(fill='x', padx=10, pady=5)
        ttk.Label(y_plus_frame, text="Target y+:").pack(side=tk.LEFT, padx=(0, 5))
        self.yplus_entry = ttk.Entry(y_plus_frame, width=10)
        self.yplus_entry.insert(0, "1.0")
        self.yplus_entry.pack(side=tk.LEFT, expand=True, fill='x')

        self.show_gmsh_var = tk.BooleanVar(value=False)
        self.show_gmsh_check = ttk.Checkbutton(
            self.mesh_frame,
            text="Show Interactive Gmsh Window",
            variable=self.show_gmsh_var
        )
        self.show_gmsh_check.pack(padx=10, pady=10)

        # 2. Flow Conditions
        self.su2_flow_conditions_frame = ttk.LabelFrame(su2_tab_mesh, text="2. Flow Conditions", labelanchor="n")
        self.su2_flow_conditions_frame.grid(row=0, column=1, padx=(0, 10), pady=10, sticky='nsew')
        self.su2_flow_conditions_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(self.su2_flow_conditions_frame, text="Reynolds Number:").grid(
            row=0, column=0, padx=(10, 5), pady=5, sticky='w'
        )
        self.su2_re_entry = ttk.Entry(self.su2_flow_conditions_frame)
        self.su2_re_entry.insert(0, "1000000")
        self.su2_re_entry.grid(row=0, column=1, padx=(0, 10), pady=5, sticky='ew')

        ttk.Label(self.su2_flow_conditions_frame, text="Mach Number:").grid(
            row=1, column=0, padx=(10, 5), pady=5, sticky='w'
        )
        self.su2_mach_entry = ttk.Entry(self.su2_flow_conditions_frame)
        self.su2_mach_entry.insert(0, "0.15")
        self.su2_mach_entry.grid(row=1, column=1, padx=(0, 10), pady=5, sticky='ew')

        ttk.Label(self.su2_flow_conditions_frame, text="Min AoA:").grid(
            row=2, column=0, padx=(10, 5), pady=5, sticky='w'
        )
        self.su2_alpha_min_entry = ttk.Entry(self.su2_flow_conditions_frame, width=10)
        self.su2_alpha_min_entry.insert(0, "0.0")
        self.su2_alpha_min_entry.grid(row=2, column=1, padx=(0, 10), pady=5, sticky='ew')

        ttk.Label(self.su2_flow_conditions_frame, text="Max AoA:").grid(
            row=3, column=0, padx=(10, 5), pady=5, sticky='w'
        )
        self.su2_alpha_max_entry = ttk.Entry(self.su2_flow_conditions_frame, width=10)
        self.su2_alpha_max_entry.insert(0, "5.0")
        self.su2_alpha_max_entry.grid(row=3, column=1, padx=(0, 10), pady=5, sticky='ew')

        ttk.Label(self.su2_flow_conditions_frame, text="AoA Step:").grid(
            row=4, column=0, padx=(10, 5), pady=5, sticky='w'
        )
        self.su2_alpha_step_entry = ttk.Entry(self.su2_flow_conditions_frame, width=10)
        self.su2_alpha_step_entry.insert(0, "1.0")
        self.su2_alpha_step_entry.grid(row=4, column=1, padx=(0, 10), pady=5, sticky='ew')

        self.load_settings_button = ttk.Button(
            self.su2_flow_conditions_frame,
            text="Load Recommended Settings",
            command=self.load_recommended_settings
        )
        self.load_settings_button.grid(row=5, column=0, columnspan=2, padx=10, pady=10, sticky='ew')

        # --- "Solver Setup & Run" Sub-Tab ---
        su2_tab_setup.grid_columnconfigure(0, weight=1)
        su2_tab_setup.grid_rowconfigure(0, weight=1)
        su2_tab_setup.grid_rowconfigure(1, weight=0)

        # 3. SU2 Configuration (Main)
        self.SU2_main_settings_frame = ttk.LabelFrame(
            su2_tab_setup, text="3. SU2 Configuration", labelanchor="n"
        )
        self.SU2_main_settings_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Flow type row (always visible at the top)
        self.SU2_main_settings_frame.grid_columnconfigure(1, weight=1)
        self.SU2_main_settings_frame.grid_rowconfigure(1, weight=1)

        ttk.Label(self.SU2_main_settings_frame, text="Flow Type:").grid(
            row=0, column=0, padx=(10, 5), pady=10, sticky="w"
        )
        self.flow_regime_var = tk.StringVar(value="Compressible")
        self.flow_regime_combo = ttk.Combobox(
            self.SU2_main_settings_frame,
            textvariable=self.flow_regime_var,
            values=["Incompressible", "Compressible"],
            state="readonly",
            width=15,
        )
        self.flow_regime_combo.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="ew")
        self.flow_regime_combo.bind("<<ComboboxSelected>>", self.update_su2_settings_display)

        self.su2_cfg_canvas = tk.Canvas(
            self.SU2_main_settings_frame, highlightthickness=0
        )
        self.su2_cfg_scrollbar = ttk.Scrollbar(
            self.SU2_main_settings_frame,
            orient="vertical",
            command=self.su2_cfg_canvas.yview,
        )
        self.su2_cfg_canvas.configure(yscrollcommand=self.su2_cfg_scrollbar.set)

        self.su2_cfg_canvas.grid(row=1, column=0, columnspan=2, sticky="nsew")
        self.su2_cfg_scrollbar.grid(row=1, column=2, sticky="ns")

        # Inner frame that actually holds the widgets
        self.su2_cfg_inner = ttk.Frame(self.su2_cfg_canvas)

        # --- FIXED: Store window ID ---
        self.su2_cfg_window_id = self.su2_cfg_canvas.create_window((0, 0), window=self.su2_cfg_inner, anchor="nw")

        def _on_cfg_inner_config(event):
            self.su2_cfg_canvas.configure(scrollregion=self.su2_cfg_canvas.bbox("all"))

        self.su2_cfg_inner.bind("<Configure>", _on_cfg_inner_config)
        def _on_canvas_configure(event):
            self.su2_cfg_canvas.itemconfig(self.su2_cfg_window_id, width=event.width)

        self.su2_cfg_canvas.bind("<Configure>", _on_canvas_configure)
        self.su2_cfg_inner.grid_columnconfigure(0, weight=1)

        self.su2_settings_frame = ttk.LabelFrame(
            self.su2_cfg_inner, text="Solver Settings", labelanchor="nw"
        )
        self.su2_settings_frame.grid(
            row=0, column=0, padx=10, pady=(0, 10), sticky="nsew"
        )
        self.su2_settings_frame.grid_columnconfigure(1, weight=1)
        self.su2_settings_frame.grid_columnconfigure(2, weight=1)
        self.su2_setting_vars = {}

        #Run Analysis (Bottom Spanning)
        run_controls_frame = ttk.LabelFrame(su2_tab_setup, text="4. Run Analysis", labelanchor="n")
        run_controls_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky='nsew')
        run_controls_frame.grid_columnconfigure(0, weight=1)

        # Boolean Options Group
        bool_options_frame = ttk.Frame(run_controls_frame)
        bool_options_frame.grid(row=0, column=0, columnspan=3, padx=10, pady=5, sticky='ew')
        bool_options_frame.grid_columnconfigure(2, weight=1)

        # Live residual plotting
        self.su2_live_plotting_var = tk.BooleanVar(value=True)
        self.live_plot_check = ttk.Checkbutton(
            bool_options_frame,
            text="Enable Live Residual Plotting",
            variable=self.su2_live_plotting_var
        )
        self.live_plot_check.pack(side="left", padx=(0, 15))

        self.use_mpi_var = tk.BooleanVar(value=True)
        self.use_mpi_check = ttk.Checkbutton(
            bool_options_frame,
            text="Use MPI parallelisation",
            variable=self.use_mpi_var,
            command=self._on_parallel_toggle
        )
        self.use_mpi_check.pack(side="left")

        #Cores input
        cores_frame = ttk.Frame(bool_options_frame)
        cores_frame.pack(side="right")
        ttk.Label(cores_frame, text="Number of cores:").pack(side="left", padx=(5, 5))
        self.num_cores_var = tk.IntVar(value=8)
        self.num_cores_entry = ttk.Entry(cores_frame, textvariable=self.num_cores_var, width=6)
        self.num_cores_entry.pack(side="left")

        # Filename input
        polar_name_frame = ttk.Frame(run_controls_frame)
        polar_name_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=5, sticky='ew')
        polar_name_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(polar_name_frame, text="Polar Filename:").grid(row=0, column=0, padx=(0, 5), sticky='w')
        self.polar_filename_entry = ttk.Entry(polar_name_frame)
        self.polar_filename_entry.insert(0, "aerodynamic_polar.csv")
        self.polar_filename_entry.grid(row=0, column=1, sticky='ew')

        #Run Buttons
        run_stop_frame = ttk.Frame(run_controls_frame)
        run_stop_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky='ew')
        run_stop_frame.grid_columnconfigure(0, weight=1)

        self.su2_analysis_button = ttk.Button(
            run_stop_frame,
            text="Run SU2 Analysis",
            command=self.run_su2_workflow_in_thread,
            style='Accent.TButton'
        )
        self.su2_analysis_button.grid(row=0, column=0, sticky='ew')
        self.stop_button = ttk.Button(run_stop_frame, text="STOP", command=self.stop_su2_analysis)
        self.stop_button.grid(row=0, column=1, padx=(10, 0), sticky='e')
        self.stop_button.config(state=tk.DISABLED)

        #Results Sub-Tab
        self.res_notebook = ttk.Notebook(su2_tab_results)
        self.res_notebook.pack(expand=True, fill='both', padx=10, pady=10)

        #Tab for Polar Graphs
        self.su2_plot_canvas_frame = ttk.Frame(self.res_notebook)
        self.res_notebook.add(self.su2_plot_canvas_frame, text="Drag Polar")

        #Tab for Visualizations (Images)
        self.su2_visual_frame = ttk.Frame(self.res_notebook)
        self.res_notebook.add(self.su2_visual_frame, text="Flow Visualization")

        #Class variables
        self.plotter_windows = {}
        self.current_su2_thread = None
        self.xfoil_thread = None
        su2_cfd_path_env = os.environ.get('SU2_CFD_PATH')
        mpiexec_path_env = os.environ.get('MPIEXEC_PATH')

        self.su2_runner = SU2Runner(
            su2_cfd_path=su2_cfd_path_env if su2_cfd_path_env else "SU2_CFD",
            mpi_exec_path=mpiexec_path_env if mpiexec_path_env else "mpiexec",
            num_procs=8,
            use_mpi=True
        )

        # Initial GUI Update
        self.update_su2_settings_display()
        self.xfoil_analyze_button.config(state=tk.DISABLED)
        self.su2_analysis_button.config(state=tk.DISABLED)
        self.hybrid_mesh_button.config(state=tk.DISABLED)
        self.structured_mesh_button.config(state=tk.DISABLED)
        self._on_parallel_toggle()

    def on_closing(self):
        if self.current_su2_thread and self.current_su2_thread.is_alive():
            self.stop_su2_analysis()
        print("Application closing...")
        self.root.destroy()

    def _on_parallel_toggle(self):
        """Enable/disable cores entry based on MPI usage."""
        if self.use_mpi_var.get():
            self.num_cores_entry.configure(state="normal")
        else:
            self.num_cores_entry.configure(state="disabled")

    def on_su2_analysis_finish(self):
        self.su2_analysis_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def stop_su2_analysis(self):
        if self.current_su2_thread and self.current_su2_thread.is_alive():
            print("Stop button pressed. Terminating SU2...")
            self.su2_runner.stop()
            messagebox.showinfo("SU2 Control", "Termination signal sent to SU2. The current analysis loop will stop.")
        else:
            messagebox.showwarning("SU2 Control", "No SU2 analysis is currently running.")

    def create_plot_window(self, data_queue, aoa, is_unsteady):
        plotter = LivePlotterWindow(self.root, data_queue, aoa, is_unsteady)
        self.plotter_windows[aoa] = plotter
        return plotter

    def run_su2_workflow_in_thread(self):
        if self.current_su2_thread and self.current_su2_thread.is_alive():
            messagebox.showwarning("Busy", "SU2 analysis is already running.")
            return

        enable_plotting = self.su2_live_plotting_var.get()
        try:
            reynolds = float(self.su2_re_entry.get())
            mach = float(self.su2_mach_entry.get())
            alpha_min = float(self.su2_alpha_min_entry.get())
            alpha_max = float(self.su2_alpha_max_entry.get())
            alpha_step = float(self.su2_alpha_step_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please ensure all SU2 flow conditions are valid numbers.")
            return

        polar_filename = self.polar_filename_entry.get()
        if not polar_filename:
            polar_filename = "aerodynamic_polar.csv"

        base_output_dir = self.int_path_entry.get()
        flow_regime = self.flow_regime_var.get()
        gui_settings = {key: var.get() for key, var in self.su2_setting_vars.items()}

        is_inviscid = False
        if gui_settings.get('KIND_TURB_MODEL') == 'NONE' or gui_settings.get('SOLVER') == 'EULER':
            is_inviscid = True

        conv_str, max_iter = self.conv_settings.get_config_string(is_inviscid=is_inviscid)
        print(f"Using CONV_RESIDUAL_MINVAL = {conv_str}, ITER/EXT_ITER = {max_iter}")

        gui_settings['CONV_RESIDUAL_MINVAL'] = conv_str
        gui_settings['ITER'] = max_iter
        gui_settings['EXT_ITER'] = max_iter

        mesh_filepath = os.path.join(os.getcwd(), "airfoil.su2")
        if not os.path.exists(mesh_filepath):
            messagebox.showerror("Error", f"Mesh file not found at {mesh_filepath}. Please generate the mesh first.")
            return

        self.su2_analysis_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        def workflow_wrapper():
            use_mpi = self.use_mpi_var.get()
            try:
                num_cores = int(self.num_cores_var.get())
            except Exception:
                num_cores = 1

            self.su2_runner.update_parallel_settings(use_mpi=use_mpi, num_procs=num_cores)

            execute_su2_analysis_workflow(
                su2_runner=self.su2_runner,
                reynolds=reynolds,
                mach=mach,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
                alpha_step=alpha_step,
                base_output_dir=base_output_dir,
                flow_regime=flow_regime,
                gui_settings=gui_settings,
                mesh_filepath=mesh_filepath,
                gui_update_callback=lambda results: self.root.after(0, self.plot_su2_results, results),
                polar_filename=polar_filename,
                enable_live_plotting=enable_plotting,
                plot_window_callback=lambda q, a, is_unsteady: self.create_plot_window(q, a, is_unsteady)
            )
            self.root.after(0, self.on_su2_analysis_finish)

        self.current_su2_thread = threading.Thread(target=workflow_wrapper)
        self.current_su2_thread.daemon = True
        self.current_su2_thread.start()

    def analyze(self):
        if not hasattr(self, 'foil1') or not hasattr(self, 'foil2'):
            messagebox.showerror("Error", "Airfoil not selected or parameterized.")
            return
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        self.method_label = ttk.Label(self.result_frame, text="")
        self.method_label.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky='w')
        self.parsec_error_label = ttk.Label(self.result_frame, text="")
        self.parsec_error_label.grid(row=1, column=0, columnspan=2, padx=10, pady=2, sticky='w')
        self.cst_error_label = ttk.Label(self.result_frame, text="")
        self.cst_error_label.grid(row=2, column=0, columnspan=2, padx=10, pady=2, sticky='w')
        self.output_path_label = ttk.Label(self.result_frame, text="")
        self.output_path_label.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky='w')
        self.parsec_error_label.config(text=f"PARSEC Error (RMSE): {self.parsec_error:.6f}")
        self.cst_error_label.config(text=f"CST Error (RMSE): {self.cst_error:.6f}")
        self.method_label.config(text=f"More Accurate Method: {self.meth}")
        n_points = int(self.n_points_entry.get())
        int_path = self.int_path_entry.get()
        os.makedirs(int_path, exist_ok=True)
        selected_airfoil_name = os.path.basename(self.selected_airfoil_path)
        i = Interpolate(self.directory, self.selected_airfoil_path)
        if self.meth == 'PARSEC':
            i.airfoil_interpolate(n_points, self.meth, self.foil1, self.directory, selected_airfoil_name, int_path)
        else:
            i.airfoil_interpolate(n_points, self.meth, self.foil2, self.directory, selected_airfoil_name, int_path)
        self.upper_surface, self.lower_surface = i.get_surface()
        output_file_path = os.path.join(int_path, "output.dat")
        self.output_path_label.config(text=f"Interpolated airfoil saved to:\n{output_file_path}")
        try:
            print("Repaneling: Reading newly generated interpolated airfoil 'output.dat'.")
            new_x, new_y = read_airfoil.read_airfoil_coordinates(int_path, output_file_path)
            self.xcoords = new_x
            self.ycoords = new_y
            print("Internal airfoil coordinates have been updated to the new interpolated profile.")
            messagebox.showinfo(
                "Repanel Complete",
                "The internal airfoil coordinates have been updated.\n\nMeshing will now use this new profile."
            )
        except Exception as e:
            messagebox.showerror(
                "Repanel Error",
                f"Could not read the newly generated airfoil file 'output.dat'.\n\n{e}"
            )
            return
        print(f"Analyze complete. Interpolated file 'output.dat' created.")
        self.xfoil_analyze_button.config(state=tk.NORMAL)
        self.hybrid_mesh_button.config(state=tk.NORMAL)

    def open_dir(self, entry_widget):
        directory = filedialog.askdirectory()
        if directory:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, directory)

    def open_file(self, entry_widget):
        file_path = filedialog.askopenfilename(filetypes=[("Executable Files", "*.exe"), ("All files", "*.*")])
        if file_path:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, file_path)

    def on_airfoil_change(self, event):
        selected_index = self.listbox.curselection()
        if not selected_index:
            return
        for widget in self.canvas_result_frame.winfo_children():
            widget.destroy()
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        self.xfoil_analyze_button.config(state=tk.DISABLED)
        self.su2_analysis_button.config(state=tk.DISABLED)
        self.hybrid_mesh_button.config(state=tk.DISABLED)
        self.structured_mesh_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        selected_airfoil = self.listbox.get(selected_index)
        self.selected_airfoil_path = os.path.join(self.directory, selected_airfoil)
        print(f'Selected Airfoil: {selected_airfoil}')
        self.take_input_and_parameterize()
        self.structured_mesh_button.config(state=tk.NORMAL)
        self.analyze_button.config(state=tk.NORMAL)

    def _update_airfoil_listbox(self, *args):
        """Filter and refresh airfoil listbox based on search text."""
        if not hasattr(self, "all_airfoils"):
            self.all_airfoils = []

        search_text = self.airfoil_search_var.get().lower() if hasattr(self, "airfoil_search_var") else ""

        self.listbox.delete(0, tk.END)
        for name in self.all_airfoils:
            if search_text in name.lower():
                self.listbox.insert(tk.END, name)

    def mesh(self):
        if not hasattr(self, 'xcoords') or not hasattr(self, 'ycoords'):
            messagebox.showerror("Error", "Airfoil coordinates not loaded. Please select an airfoil first.")
            return
        try:
            current_re = float(self.su2_re_entry.get())
            current_m = float(self.su2_mach_entry.get())
            current_yplus = float(self.yplus_entry.get())

            if self.show_gmsh_var.get():
                args_tuple = (self.xcoords, self.ycoords, current_re, current_m)
                kwargs_dict = {'y_plus': current_yplus, 'show_graphics': True, 'hide_output': False}
                p = multiprocessing.Process(target=generate_mesh, args=args_tuple, kwargs=kwargs_dict)
                p.start()
                messagebox.showinfo(
                    "Success",
                    "Gmsh window launched in a separate process.\n"
                    "The airfoil.su2 file will be generated when you close Gmsh."
                )
            else:
                generate_mesh(
                    self.xcoords,
                    self.ycoords,
                    current_re,
                    current_m,
                    y_plus=current_yplus,
                    show_graphics=False,
                    hide_output=False
                )
                messagebox.showinfo("Success", "Mesh generation complete. 'airfoil.su2' created.")

            self.su2_analysis_button.config(state=tk.NORMAL)
        except ValueError:
            messagebox.showerror("Input Error", "Please ensure Reynolds, Mach, and y+ are valid numbers.")
        except Exception as e:
            messagebox.showerror("Meshing Error", f"An error occurred during mesh generation:\n\n{e}")

    def hybrid(self):
        if not hasattr(self, 'upper_surface') or not hasattr(self, 'lower_surface'):
            messagebox.showerror("Error", "Airfoil surfaces not generated. Please run 'Analyze' first.")
            return
        try:
            current_re = float(self.su2_re_entry.get())
            current_m = float(self.su2_mach_entry.get())
            current_yplus = float(self.yplus_entry.get())

            if self.show_gmsh_var.get():
                args_tuple = (self.upper_surface, self.lower_surface, current_re, current_m)
                kwargs_dict = {'y_plus': current_yplus, 'show_graphics': True, 'hide_output': False}
                p = multiprocessing.Process(target=generate_hybrid, args=args_tuple, kwargs=kwargs_dict)
                p.start()
                messagebox.showinfo(
                    "Success",
                    "Gmsh window launched in a separate process.\n"
                    "The airfoil.su2 file will be generated when you close Gmsh."
                )
            else:
                generate_hybrid(
                    self.upper_surface,
                    self.lower_surface,
                    current_re,
                    current_m,
                    y_plus=current_yplus,
                    show_graphics=False,
                    hide_output=False
                )
                messagebox.showinfo("Success", "Hybrid mesh generation complete. 'airfoil.su2' created.")

            self.su2_analysis_button.config(state=tk.NORMAL)
        except ValueError:
            messagebox.showerror("Input Error", "Please ensure Reynolds, Mach, and y+ are valid numbers.")
        except Exception as e:
            messagebox.showerror("Meshing Error", f"An error occurred during hybrid mesh generation:\n\n{e}")

    def _on_conv_method_category_change(self, event=None):
        settings_dict = (
            SU2_INCOMPRESSIBLE_SETTINGS
            if self.flow_regime_var.get() == "Incompressible"
            else SU2_COMPRESSIBLE_SETTINGS
        )
        category = self.conv_category_var.get()
        new_schemes = settings_dict['CONV_NUM_METHOD_FLOW'][category]
        self.conv_scheme_combo['values'] = new_schemes
        self.su2_setting_vars['CONV_NUM_METHOD_FLOW'].set(new_schemes[0])

    def update_su2_settings_display(self, event=None):
        for widget in self.su2_settings_frame.winfo_children():
            widget.destroy()
        self.su2_setting_vars = {}
        selected_regime = self.flow_regime_var.get()
        settings_to_display = (
            SU2_INCOMPRESSIBLE_SETTINGS
            if selected_regime == "Incompressible"
            else SU2_COMPRESSIBLE_SETTINGS
        )
        row_idx = 0

        self.su2_settings_frame.grid_columnconfigure(1, weight=1)
        self.su2_settings_frame.grid_columnconfigure(2, weight=1)

        for key, options in settings_to_display.items():
            if key in ['ITER', 'EXT_ITER', 'CONV_RESIDUAL_MINVAL']:
                continue

            ttk.Label(self.su2_settings_frame, text=f"{key}:").grid(
                row=row_idx, column=0, padx=(10, 5), pady=5, sticky='w'
            )

            if key == 'CONV_NUM_METHOD_FLOW' and isinstance(options, dict):
                self.conv_category_var = tk.StringVar(value=list(options.keys())[0])
                category_combo = ttk.Combobox(
                    self.su2_settings_frame,
                    textvariable=self.conv_category_var,
                    values=list(options.keys()),
                    state="readonly"
                )
                category_combo.grid(row=row_idx, column=1, padx=5, pady=5, sticky='ew')
                category_combo.bind("<<ComboboxSelected>>", self._on_conv_method_category_change)

                initial_category = self.conv_category_var.get()
                initial_schemes = options[initial_category]
                scheme_var = tk.StringVar(value=initial_schemes[0])
                self.conv_scheme_combo = ttk.Combobox(
                    self.su2_settings_frame,
                    textvariable=scheme_var,
                    values=initial_schemes,
                    state="readonly"
                )
                self.conv_scheme_combo.grid(row=row_idx, column=2, padx=(0, 10), pady=5, sticky='ew')
                self.su2_setting_vars[key] = scheme_var
            elif isinstance(options, list):
                var = tk.StringVar(value=options[0])
                combo = ttk.Combobox(
                    self.su2_settings_frame,
                    textvariable=var,
                    values=options,
                    state="readonly"
                )
                combo.grid(row=row_idx, column=1, padx=5, pady=5, sticky='ew', columnspan=2)
                self.su2_setting_vars[key] = var

                if key == 'KIND_TURB_MODEL' or key == 'SOLVER':
                    combo.bind("<<ComboboxSelected>>", self.check_turbulence_status)

            row_idx += 1

        ttk.Label(self.su2_settings_frame, text="CFL Number:").grid(
            row=row_idx, column=0, padx=(10, 5), pady=5, sticky='w'
        )
        self.su2_setting_vars['CFL_NUMBER'] = tk.DoubleVar(value=1.0)
        ttk.Entry(
            self.su2_settings_frame,
            textvariable=self.su2_setting_vars['CFL_NUMBER']
        ).grid(row=row_idx, column=1, padx=5, pady=5, sticky='ew', columnspan=2)
        row_idx += 1

        ttk.Label(self.su2_settings_frame, text="Convergence Field:").grid(
            row=row_idx, column=0, padx=(10, 5), pady=5, sticky='w'
        )
        conv_field_options = [
            "RMS_DENSITY", "RMS_PRESSURE", "RMS_MOMENTUM-X",
            "RMS_ENERGY", "LIFT", "DRAG", "RESIDUAL"
        ]
        self.su2_setting_vars['CONV_FIELD'] = tk.StringVar(value="RMS_DENSITY")
        conv_combo = ttk.Combobox(
            self.su2_settings_frame,
            textvariable=self.su2_setting_vars['CONV_FIELD'],
            values=conv_field_options,
            state="readonly"
        )
        conv_combo.grid(row=row_idx, column=1, padx=5, pady=5, sticky='ew', columnspan=2)
        row_idx += 1
        self.conv_settings = ConvergenceSettings(self.su2_settings_frame, start_row=row_idx)

        self.check_turbulence_status()

    def parse_su2_cfg(self, file_path):
        """Parses a .cfg file and returns a dictionary of settings."""
        settings = {}
        if not os.path.exists(file_path):
            messagebox.showerror(
                "Error",
                f"Config file not found: {file_path}\n\nPlease ensure this file is in the same directory as the script."
            )
            return None

        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('%'):
                        continue  

                    m = re.match(r'^\s*([A-Za-z0-9_]+)\s*=\s*(.*)', line)
                    if m:
                        key = m.group(1).strip().upper()
                        value = m.group(2).strip()
                        value = value.split('%')[0].strip()
                        settings[key] = value
        except Exception as e:
            messagebox.showerror("Parse Error", f"Error reading {file_path}:\n{e}")
            return None

        print(f"Parsed {len(settings)} settings from {file_path}.")
        return settings

    def load_recommended_settings(self):
        """
        Pick a preset .cfg based on Mach & Re,
        parse it, and apply the solver settings to the GUI.
        """
        try:
            mach = float(self.su2_mach_entry.get())
            reynolds = float(self.su2_re_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers for Mach and Reynolds.")
            return

        selected_file = ""
        new_regime = ""

        if mach > 1.0:
            selected_file = "supersonic.cfg"
            new_regime = "Compressible"
        elif 0.7 <= mach <= 1.0:
            selected_file = "Transonic.cfg"
            new_regime = "Compressible"
        else:  
            new_regime = "Incompressible"
            if reynolds >= 1e6:
                selected_file = "HighReIncomp.cfg"
            else:
                selected_file = "LowReIncomp.cfg"

        if not selected_file:
            messagebox.showinfo("Info", "Could not determine a preset for the given values.")
            return

        cfg_path = os.path.join(self.script_dir, selected_file)
        print(f"Loading recommended settings from: {cfg_path}")

        self.loaded_cfg_settings = self.parse_su2_cfg(cfg_path)
        if self.loaded_cfg_settings is None:
            self.loaded_cfg_settings = {}
            return

        self.flow_regime_var.set(new_regime)
        self.update_su2_settings_display()
        self.apply_loaded_settings()
        self.loaded_cfg_settings = {}

        messagebox.showinfo("Success", f"Successfully loaded settings from {selected_file}.")

    def apply_loaded_settings(self):
        """Apply values from self.loaded_cfg_settings to self.su2_setting_vars."""
        if not getattr(self, "loaded_cfg_settings", None):
            return

        print("Applying loaded settings to GUI...")
        for key, var in self.su2_setting_vars.items():
            upper_key = key.upper()
            if upper_key not in self.loaded_cfg_settings:
                continue

            loaded_val = self.loaded_cfg_settings[upper_key]

            try:
                if upper_key == "CONV_NUM_METHOD_FLOW":
                    settings_dict = (
                        SU2_INCOMPRESSIBLE_SETTINGS
                        if self.flow_regime_var.get() == "Incompressible"
                        else SU2_COMPRESSIBLE_SETTINGS
                    )
                    found = False
                    for category, schemes in settings_dict["CONV_NUM_METHOD_FLOW"].items():
                        if loaded_val in schemes:
                            self.conv_category_var.set(category)
                            self._on_conv_method_category_change()
                            var.set(loaded_val)
                            found = True
                            print(f"  Applied {upper_key} = {loaded_val} (Category: {category})")
                            break
                    if not found:
                        print(f"  Warning: Could not map {upper_key} = {loaded_val} to a known category.")
                else:
                    var.set(loaded_val)
                    print(f"  Applied {upper_key} = {loaded_val}")

            except Exception as e:
                print(f"  Warning: Could not apply {upper_key}={loaded_val}. Error: {e}")

    def check_turbulence_status(self, event=None):
        try:
            is_turbulent = True
            if 'SOLVER' in self.su2_setting_vars:
                if self.su2_setting_vars['SOLVER'].get() == 'EULER':
                    is_turbulent = False
            if 'KIND_TURB_MODEL' in self.su2_setting_vars:
                if self.su2_setting_vars['KIND_TURB_MODEL'].get() == 'NONE':
                    is_turbulent = False
            self.conv_settings.toggle_turbulence(is_turbulent)
        except:
            pass

    def take_input_and_parameterize(self):
        self.xcoords, self.ycoords = read_airfoil.read_airfoil_coordinates(
            self.directory,
            self.selected_airfoil_path
        )
        if self.xcoords.size == 0:
            messagebox.showerror(
                "File Error",
                f"Could not read coordinates from {self.selected_airfoil_path}.\nCheck file format."
            )
            return
        p = Parsec(self.directory, self.selected_airfoil_path)
        c = CST(self.directory, self.selected_airfoil_path)
        self.foil1 = p.foil()
        self.foil2 = c.foil()
        self.plot_airfoil(self.xcoords, self.foil1, self.ycoords, self.foil2)
        self.meth, self.parsec_error, self.cst_error = error(self.foil1, self.foil2, self.ycoords)

    def xfoil_analyze_threaded(self):
        if self.xfoil_thread and self.xfoil_thread.is_alive():
            messagebox.showwarning("Busy", "XFoil analysis is already running.")
            return
        self.xfoil_analyze_button.config(state=tk.DISABLED)
        self.xfoil_thread = threading.Thread(target=self.xfoil_analyze)
        self.xfoil_thread.daemon = True
        self.xfoil_thread.start()
        self.root.after(100, self.check_xfoil_thread)

    def check_xfoil_thread(self):
        if self.xfoil_thread.is_alive():
            self.root.after(100, self.check_xfoil_thread)
        else:
            self.xfoil_analyze_button.config(state=tk.NORMAL)
            print("XFoil analysis thread finished.")

    def sanitize_for_xfoil(self, filepath):
        """
        Reads the .dat file, ensures correct XFOIL formatting:
        1. Header on first line.
        2. Order: Trailing Edge (Top) -> Leading Edge -> Trailing Edge (Bottom).
        3. Normalized coordinates.
        """
        try:
            # Read data
            with open(filepath, 'r') as f:
                lines = f.readlines()

            coords = []
            header = "Analyzed_Airfoil"
            try:
                float(lines[0].split()[0])
            except ValueError:
                header = lines[0].strip()
                lines = lines[1:]

            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        coords.append([float(parts[0]), float(parts[1])])
                    except ValueError:
                        continue

            coords = np.array(coords)
            le_idx = np.argmin(coords[:, 0])
            part1 = coords[:le_idx + 1]
            part2 = coords[le_idx:]
            if np.mean(part1[:, 1]) > np.mean(part2[:, 1]):
                upper = part1
                lower = part2
            else:
                upper = part2
                lower = part1

            upper = upper[np.argsort(upper[:, 0])[::-1]]
            lower = lower[np.argsort(lower[:, 0])]
            final_coords = np.concatenate((upper, lower[1:]))
            with open(filepath, 'w') as f:
                f.write(f"{header}\n")
                for x, y in final_coords:
                    f.write(f" {x:.6f}  {y:.6f}\n")

            print(f"Sanitized airfoil file at {filepath}")
            return True

        except Exception as e:
            print(f"Error sanitizing XFOIL file: {e}")
            return False

    def xfoil_analyze(self):
        xfoil_path = self.xfoil_path_entry.get()
        int_path = self.int_path_entry.get()
        os.makedirs(int_path, exist_ok=True)
        airfoil_path = os.path.join(int_path, "output.dat")

        if not os.path.exists(airfoil_path):
            messagebox.showerror("Error", f"Airfoil data file not found at {airfoil_path}.")
            self.root.after(0, lambda: self.xfoil_analyze_button.config(state=tk.NORMAL))
            return
        if not self.sanitize_for_xfoil(airfoil_path):
            messagebox.showerror("Error", "Could not sanitize airfoil file format.")
            self.root.after(0, lambda: self.xfoil_analyze_button.config(state=tk.NORMAL))
            return

        try:
            Re = float(self.re_entry.get())
            M = float(self.mach_entry.get())
            alpha_max = float(self.alpha_max_entry.get())
            alpha_min = float(self.alpha_min_entry.get())
            alpha_step = float(self.alpha_step_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please ensure all XFoil inputs are valid numbers.")
            self.root.after(0, lambda: self.xfoil_analyze_button.config(state=tk.NORMAL))
            return
        try:
            perform_xfoil_analysis(
                xfoil_path,
                self.xfoil_result_frame,
                int_path,
                airfoil_path,
                Re,
                alpha_max,
                alpha_min,
                alpha_step,
                M
            )
        except FileNotFoundError:
            messagebox.showerror("XFoil Error", "Analysis failed. Could not find polar file.")
        except Exception as e:
            messagebox.showerror("XFoil Error", f"An unexpected error occurred: {e}")

    def plot_airfoil(self, xcoords, foil1, ycoords, foil2):
        for widget in self.canvas_result_frame.winfo_children():
            widget.destroy()
        fig, axs = plt.subplots(2, 1, figsize=(10, 3))
        axs[0].plot(xcoords, foil1, 'r')
        axs[0].plot(xcoords, ycoords, ':b')
        axs[0].legend(['PARSEC', 'Actual'])
        axs[0].set_title('PARSEC Parameterization')
        axs[0].grid()
        axs[0].set_aspect('equal', adjustable='box')
        axs[1].plot(xcoords, foil2, 'r')
        axs[1].plot(xcoords, ycoords, ':b')
        axs[1].legend(["CST", "Actual"])
        axs[1].set_title('CST Parameterization')
        axs[1].grid()
        axs[1].set_aspect('equal', adjustable='box')
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_result_frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.draw()
        plt.close(fig)

    def plot_su2_results(self, results_list):
        if not results_list:
            messagebox.showinfo("SU2 Results", "SU2 analysis completed, but no valid results were generated to plot.")
            return
        for widget in self.su2_plot_canvas_frame.winfo_children():
            widget.destroy()

        AoA_values, Cl_values, Cd_values, Cm_values = extract_su2_polar_data(results_list)

        if not any(v is not None and not np.isnan(v) for v in Cl_values):
            messagebox.showinfo("SU2 Results", "Could not extract valid Cl/Cd data from SU2 history files.")
            return

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].plot(AoA_values, Cl_values, 'o-b')
        axs[0].set_xlabel("Angle of Attack (deg)")
        axs[0].set_ylabel("Coefficient of Lift (Cl)")
        axs[0].set_title("Cl vs. AoA (SU2)")
        axs[0].grid(True)

        axs[1].plot(Cd_values, Cl_values, 'o-r')
        axs[1].set_xlabel("Coefficient of Drag (Cd)")
        axs[1].set_ylabel("Coefficient of Lift (Cl)")
        axs[1].set_title("Drag Polar (SU2)")
        axs[1].grid(True)

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.su2_plot_canvas_frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.draw()
        plt.close(fig)
        for widget in self.su2_visual_frame.winfo_children():
            widget.destroy()

        last_aoa, _, last_flow_file, _ = results_list[-1]
        run_dir = os.path.dirname(last_flow_file)

        images = [f for f in os.listdir(run_dir) if f.endswith('.png')]

        if not images:
            ttk.Label(self.su2_visual_frame, text="No visualization images found for the last run.").pack(pady=10)
            return

        canvas_scroll = tk.Canvas(self.su2_visual_frame)
        scrollbar = ttk.Scrollbar(self.su2_visual_frame, orient="vertical", command=canvas_scroll.yview)
        scrollable_frame = ttk.Frame(canvas_scroll)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))
        )
        canvas_scroll.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas_scroll.configure(yscrollcommand=scrollbar.set)

        canvas_scroll.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for img_name in sorted(images):
            try:
                img_path = os.path.join(run_dir, img_name)
                pil_img = Image.open(img_path)

                base_width = 600
                w_percent = (base_width / float(pil_img.size[0]))
                h_size = int((float(pil_img.size[1]) * float(w_percent)))
                pil_img = pil_img.resize((base_width, h_size), Image.Resampling.LANCZOS)

                tk_img = ImageTk.PhotoImage(pil_img)

                img_container = ttk.Frame(scrollable_frame)
                img_container.pack(pady=10)

                ttk.Label(
                    img_container,
                    text=img_name,
                    font=('Segoe UI', 10, 'bold')
                ).pack()
                lbl = ttk.Label(img_container, image=tk_img)
                lbl.image = tk_img
                lbl.pack()

            except Exception as e:
                print(f"Failed to load image {img_name}: {e}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.geometry(f"{int(screen_width * 0.8)}x{int(screen_height * 0.8)}")
    app = App(root)
    root.mainloop()