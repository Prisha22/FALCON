import sys
import os
import shutil
import threading
import multiprocessing
import numpy as np
import re
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QTabWidget, QLabel, QLineEdit, QPushButton,
    QTextEdit, QListWidget, QComboBox, QCheckBox, QSplitter,
    QFileDialog, QGroupBox, QScrollArea, QMessageBox, QFrame, QFormLayout
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import QPixmap, QImage, QFont

import matplotlib

matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib.cm as cm

from Scripts.Solver_and_Results.su2_analyzer import (
    SU2Runner, execute_su2_analysis_workflow, extract_su2_polar_data,
    SU2_INCOMPRESSIBLE_SETTINGS, SU2_COMPRESSIBLE_SETTINGS
)
from Scripts.Geometry import read_airfoil
from Scripts.Geometry.parsec import Parsec
from Scripts.Geometry.cst import CST
from Scripts.Geometry.interpolate import Interpolate
from Scripts.Meshing.hybrid import generate_hybrid
from Scripts.Meshing.meshing import generate_mesh
from Scripts.Solver_and_Results import xfoil1
from Scripts.Solver_and_Results.live_plotter import LivePlotterWindow


class ConvergenceSettings(QGroupBox):

    def __init__(self):
        super().__init__("Convergence Criteria")
        layout = QFormLayout()

        self.res_min_input = QLineEdit("-8.0")
        self.max_iter_input = QLineEdit("5000")

        layout.addRow("Min log10 Residual:", self.res_min_input)
        layout.addRow("Max Iterations:", self.max_iter_input)
        self.setLayout(layout)

    def toggle_turbulence(self, is_turbulent):
        pass

    def get_config_string(self, is_inviscid=False):
        return self.res_min_input.text(), self.max_iter_input.text()


class LoggerSignals(QObject):
    write_log = pyqtSignal(str, str)


class QtTextRedirector:

    def __init__(self, signal, tag="stdout"):
        self.signal = signal
        self.tag = tag

    def write(self, text):
        if text: self.signal.write_log.emit(str(text), self.tag)

    def flush(self): pass


def error(foil1, foil2, ycoords):
    rmse1 = np.sqrt(np.mean((foil1 - ycoords) ** 2))
    rmse2 = np.sqrt(np.mean((foil2 - ycoords) ** 2))
    print(f'PARSEC Root Mean Square Error (RMSE): {rmse1}')
    print(f'CST Root Mean Square Error (RMSE): {rmse2}')
    return ("PARSEC" if rmse1 < rmse2 else "CST"), rmse1, rmse2


class FalconApp(QMainWindow):
    xfoil_finished_signal = pyqtSignal(list, list, float, str)
    su2_finished_signal = pyqtSignal()
    plot_su2_signal = pyqtSignal(list)
    request_plot_window_signal = pyqtSignal(object, float, bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FALCON : Framework for Airfoil CFD and anaLysis OptimizatioN")
        self.resize(1300, 900)

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.directory = os.path.join(self.script_dir, "Airfoil_DAT_Selig")
        print(f"Working Directory: {self.script_dir}")

        self.su2_runner = SU2Runner()
        self.live_windows = []
        self.su2_setting_widgets = {}
        self.loaded_cfg_settings = {}

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(self.main_splitter)

        self.tabs = QTabWidget()
        self.main_splitter.addWidget(self.tabs)

        log_group = QGroupBox("Live Log Output")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #2B2B2B; color: #F8F8F2; font-family: Consolas;")
        log_layout.addWidget(self.log_text)
        self.main_splitter.addWidget(log_group)
        self.main_splitter.setSizes([700, 200])

        self.log_sig = LoggerSignals()
        self.log_sig.write_log.connect(self.append_log)
        sys.stdout = QtTextRedirector(self.log_sig, "stdout")
        sys.stderr = QtTextRedirector(self.log_sig, "stderr")

        print("--- Log initialized ---")

        self.init_tab1()
        self.init_tab2()
        self.init_tab3()

        self.xfoil_finished_signal.connect(self.on_xfoil_results)
        self.su2_finished_signal.connect(self.on_su2_analysis_finish)
        self.plot_su2_signal.connect(self.plot_su2_results)
        self.request_plot_window_signal.connect(self.open_live_plotter)

    def append_log(self, text, tag):
        color = "#F8F8F2" if tag == "stdout" else "#FF5555"
        self.log_text.moveCursor(self.log_text.textCursor().MoveOperation.End)
        self.log_text.insertHtml(f'<span style="color:{color};">{text}</span>'.replace("\n", "<br>"))
        self.log_text.ensureCursorVisible()

    def on_closing(self):
        if self.su2_runner.current_process:
            self.stop_su2_analysis()
        self.close()

    def init_tab1(self):
        self.tab1 = QWidget()
        self.tabs.addTab(self.tab1, "Airfoil Parameterization")
        layout = QGridLayout(self.tab1)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        sel_group = QGroupBox("Select Airfoil")
        sel_layout = QVBoxLayout(sel_group)

        h_search = QHBoxLayout()
        h_search.addWidget(QLabel("Search:"))
        self.airfoil_search_var = QLineEdit()
        self.airfoil_search_var.textChanged.connect(self._update_airfoil_listbox)
        h_search.addWidget(self.airfoil_search_var)
        sel_layout.addLayout(h_search)

        self.listbox = QListWidget()
        self.listbox.itemSelectionChanged.connect(self.on_airfoil_change)
        sel_layout.addWidget(self.listbox)
        left_layout.addWidget(sel_group)

        ctrl_group = QGroupBox("Controls")
        ctrl_layout = QFormLayout(ctrl_group)

        self.n_points_entry = QLineEdit("200")
        ctrl_layout.addRow("Desired points:", self.n_points_entry)

        self.int_path_entry = QLineEdit(os.path.join(os.getcwd(), "output"))
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(lambda: self.open_dir(self.int_path_entry))
        h_path = QHBoxLayout()
        h_path.addWidget(self.int_path_entry)
        h_path.addWidget(browse_btn)
        ctrl_layout.addRow("Output Path:", h_path)

        self.analyze_button = QPushButton("Analyze && Generate Interpolated Airfoil")
        self.analyze_button.setStyleSheet("font-weight: bold; padding: 5px;")
        self.analyze_button.clicked.connect(self.analyze)
        self.analyze_button.setEnabled(False)
        ctrl_layout.addRow(self.analyze_button)

        left_layout.addWidget(ctrl_group)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        plot_group = QGroupBox("Airfoil Plot")
        plot_layout = QVBoxLayout(plot_group)
        self.tab1_figure = Figure(figsize=(5, 4), facecolor='#1e1e1e')
        self.tab1_canvas = FigureCanvasQTAgg(self.tab1_figure)
        self.tab1_canvas.setStyleSheet("background-color: #1e1e1e;")
        plot_layout.addWidget(self.tab1_canvas)
        right_layout.addWidget(plot_group)

        res_group = QGroupBox("Parameterization Results")
        res_layout = QVBoxLayout(res_group)
        self.method_label = QLabel("Method:")
        self.parsec_error_label = QLabel("PARSEC Error:")
        self.cst_error_label = QLabel("CST Error:")
        self.output_path_label = QLabel("")

        res_layout.addWidget(self.method_label)
        res_layout.addWidget(self.parsec_error_label)
        res_layout.addWidget(self.cst_error_label)
        res_layout.addWidget(self.output_path_label)
        right_layout.addWidget(res_group)

        layout.addWidget(left_panel, 0, 0)
        layout.addWidget(right_panel, 0, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 2)

        if os.path.exists(self.directory):
            self.all_airfoils = [f for f in os.listdir(self.directory) if f.endswith('.dat')]
        else:
            self.all_airfoils = []
        self._update_airfoil_listbox()

    def _update_airfoil_listbox(self):
        term = self.airfoil_search_var.text().lower()
        self.listbox.clear()
        for name in self.all_airfoils:
            if term in name.lower():
                self.listbox.addItem(name)

    def on_airfoil_change(self):
        items = self.listbox.selectedItems()
        if not items: return

        self.xfoil_analyze_button.setEnabled(False)
        self.su2_analysis_button.setEnabled(False)
        self.hybrid_mesh_button.setEnabled(False)
        self.structured_mesh_button.setEnabled(False)

        selected_airfoil = items[0].text()
        self.selected_airfoil_path = os.path.join(self.directory, selected_airfoil)
        print(f'Selected Airfoil: {selected_airfoil}')

        self.take_input_and_parameterize()
        self.structured_mesh_button.setEnabled(True)
        self.analyze_button.setEnabled(True)

    def take_input_and_parameterize(self):
        try:
            self.xcoords, self.ycoords = read_airfoil.read_airfoil_coordinates(self.directory,
                                                                               self.selected_airfoil_path)
            if self.xcoords.size == 0: raise ValueError("Empty coordinates")

            p = Parsec(self.directory, self.selected_airfoil_path)
            c = CST(self.directory, self.selected_airfoil_path)
            self.foil1 = p.foil()
            self.foil2 = c.foil()
            self.meth, self.parsec_error, self.cst_error = error(self.foil1, self.foil2, self.ycoords)

            self.tab1_figure.clear()
            ax1 = self.tab1_figure.add_subplot(211)
            ax1.set_facecolor('#2b2b2b')
            ax1.tick_params(colors='#f8f8f2')
            ax1.xaxis.label.set_color('#f8f8f2')
            ax1.yaxis.label.set_color('#f8f8f2')
            ax1.title.set_color('#f8f8f2')
            for spine in ax1.spines.values():
                spine.set_edgecolor('#555555')
            ax1.plot(self.xcoords, self.foil1, color='#ff6b6b', linewidth=1.5, label='PARSEC fit')
            ax1.plot(self.xcoords, self.ycoords, color='#00d4ff', linewidth=1.0, linestyle='--', label='Actual', zorder=5)
            ax1.legend(facecolor='#3a3a3a', labelcolor='#f8f8f2')
            ax1.grid(True, color='#555555')
            ax1.set_title('PARSEC')

            ax2 = self.tab1_figure.add_subplot(212)
            ax2.set_facecolor('#2b2b2b')
            ax2.tick_params(colors='#f8f8f2')
            ax2.xaxis.label.set_color('#f8f8f2')
            ax2.yaxis.label.set_color('#f8f8f2')
            ax2.title.set_color('#f8f8f2')
            for spine in ax2.spines.values():
                spine.set_edgecolor('#555555')
            ax2.plot(self.xcoords, self.foil2, color='#ff6b6b', linewidth=1.5, label='CST fit')
            ax2.plot(self.xcoords, self.ycoords, color='#00d4ff', linewidth=1.0, linestyle='--', label='Actual', zorder=5)
            ax2.legend(facecolor='#3a3a3a', labelcolor='#f8f8f2')
            ax2.grid(True, color='#555555')
            ax2.set_title('CST')

            self.tab1_figure.tight_layout()
            self.tab1_canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error parameterizing airfoil: {e}")

    def analyze(self):
        if not hasattr(self, 'foil1'): return

        self.method_label.setText(f"More Accurate Method: {self.meth}")
        self.parsec_error_label.setText(f"PARSEC Error: {self.parsec_error:.6f}")
        self.cst_error_label.setText(f"CST Error: {self.cst_error:.6f}")

        try:
            n_points = int(self.n_points_entry.text())
            int_path = self.int_path_entry.text()
            os.makedirs(int_path, exist_ok=True)

            i = Interpolate(self.directory, self.selected_airfoil_path)
            target = self.foil1 if self.meth == 'PARSEC' else self.foil2

            i.airfoil_interpolate(n_points, self.meth, target, self.directory,
                                  os.path.basename(self.selected_airfoil_path), int_path)

            self.upper_surface, self.lower_surface = i.get_surface()
            output_file = os.path.join(int_path, "output.dat")
            self.output_path_label.setText(f"Saved to:\n{output_file}")

            new_x, new_y = read_airfoil.read_airfoil_coordinates(int_path, output_file)
            self.xcoords, self.ycoords = new_x, new_y
            print("[DEBUG] Internal coordinates updated from interpolated airfoil")
            print(f"[DEBUG] Total points: {len(self.xcoords)}")

            QMessageBox.information(self, "Success", "Interpolation & Repaneling Complete.")
            self.xfoil_analyze_button.setEnabled(True)
            self.hybrid_mesh_button.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def init_tab2(self):
        self.tab2 = QWidget()
        self.tabs.addTab(self.tab2, "XFOIL Analysis")
        layout = QGridLayout(self.tab2)

        input_group = QGroupBox("Input Parameters")
        form = QFormLayout(input_group)

        self.xfoil_path_entry = QLineEdit(shutil.which("xfoil") or "xfoil.exe")
        browse_xf = QPushButton("...")
        browse_xf.clicked.connect(lambda: self.open_file(self.xfoil_path_entry))
        h_xf = QHBoxLayout()
        h_xf.addWidget(self.xfoil_path_entry)
        h_xf.addWidget(browse_xf)
        form.addRow("XFOIL Executable:", h_xf)

        self.re_entry = QLineEdit("1000000")
        self.mach_entry = QLineEdit("0.0")
        self.alpha_min_entry = QLineEdit("0.0")
        self.alpha_max_entry = QLineEdit("5.0")
        self.alpha_step_entry = QLineEdit("1.0")

        form.addRow("Reynolds Number:", self.re_entry)
        form.addRow("Mach Number:", self.mach_entry)
        form.addRow("Min Alpha:", self.alpha_min_entry)
        form.addRow("Max Alpha:", self.alpha_max_entry)
        form.addRow("Step:", self.alpha_step_entry)

        self.xfoil_analyze_button = QPushButton("Analyze with XFOIL")
        self.xfoil_analyze_button.setStyleSheet("font-weight: bold; padding: 5px;")
        self.xfoil_analyze_button.clicked.connect(self.xfoil_analyze_threaded)
        self.xfoil_analyze_button.setEnabled(False)
        form.addRow(self.xfoil_analyze_button)
        self.xfoil_stop_button = QPushButton("STOP XFOIL")
        self.xfoil_stop_button.setEnabled(False)
        self.xfoil_stop_button.clicked.connect(xfoil1.stop_xfoil)
        form.addRow(self.xfoil_stop_button)

        layout.addWidget(input_group, 0, 0)

        plot_group = QGroupBox("Polar Plot")
        plot_layout = QVBoxLayout(plot_group)
        self.xfoil_figure = Figure(figsize=(5, 4), facecolor='#1e1e1e')
        self.xfoil_canvas = FigureCanvasQTAgg(self.xfoil_figure)
        self.xfoil_canvas.setStyleSheet("background-color: #1e1e1e;")
        plot_layout.addWidget(self.xfoil_canvas)
        layout.addWidget(plot_group, 0, 1)
        layout.setColumnStretch(1, 1)

    def xfoil_analyze_threaded(self):
        self.xfoil_analyze_button.setEnabled(False)
        self.xfoil_stop_button.setEnabled(True)

        params = {
            'xfoil_path': self.xfoil_path_entry.text(),
            'int_path': self.int_path_entry.text(),
            'airfoil_full_path': os.path.join(self.int_path_entry.text(), "output.dat"),
            'Re': float(self.re_entry.text()),
            'M': float(self.mach_entry.text()),
            'alpha_min': float(self.alpha_min_entry.text()),
            'alpha_max': float(self.alpha_max_entry.text()),
            'alpha_step': float(self.alpha_step_entry.text())
        }

        def worker():
            try:
                pol, cp_files = xfoil1.run_xfoil_logic(**params)
                self.xfoil_finished_signal.emit(pol, cp_files, params['Re'], params['int_path'])
            except Exception as e:
                print(f"XFOIL Error: {e}")

        threading.Thread(target=worker, daemon=True).start()

    def on_xfoil_results(self, polar_data, cp_data, Re, int_path):
        self.xfoil_analyze_button.setEnabled(True)
        self.xfoil_figure.clear()

        ax1 = self.xfoil_figure.add_subplot(211)
        ax1.set_facecolor('#2b2b2b')
        ax1.tick_params(colors='#f8f8f2')
        ax1.xaxis.label.set_color('#f8f8f2')
        ax1.yaxis.label.set_color('#f8f8f2')
        ax1.title.set_color('#f8f8f2')
        for spine in ax1.spines.values():
            spine.set_edgecolor('#555555')
        if polar_data:
            try:
                pd = np.array(polar_data)
                ax1.plot(pd[:, 2], pd[:, 1], 'o-', label=f'Re={Re}')
                ax1.set_xlabel("Cd")
                ax1.set_ylabel("Cl")
                ax1.set_title(f"Polar Plot (Re={Re})")
                ax1.grid(True, color='#555555')
                ax1.legend(facecolor='#3a3a3a', labelcolor='#f8f8f2')
            except Exception as e:
                print(f"Polar Plotting Error: {e}")
        else:
            ax1.text(0.5, 0.5, "Convergence Failed", ha='center', color='#ff5555')

        ax2 = self.xfoil_figure.add_subplot(212)
        ax2.set_facecolor('#2b2b2b')
        ax2.tick_params(colors='#f8f8f2')
        ax2.xaxis.label.set_color('#f8f8f2')
        ax2.yaxis.label.set_color('#f8f8f2')
        ax2.title.set_color('#f8f8f2')
        for spine in ax2.spines.values():
            spine.set_edgecolor('#555555')

        try:
            try:
                cmap = matplotlib.colormaps['jet']
            except AttributeError:
                cmap = cm.get_cmap('jet')

            lines_plotted = 0

            if isinstance(cp_data, dict):
                sorted_alphas = sorted(cp_data.keys())
                for i, alpha in enumerate(sorted_alphas):
                    x, cp = cp_data[alpha]
                    color = cmap(i / max(1, len(sorted_alphas) - 1))
                    ax2.plot(x, cp, linewidth=1, color=color, label=f"a={alpha:.1f}")
                    lines_plotted += 1

            elif isinstance(cp_data, list):
                for i, item in enumerate(cp_data):
                    if len(item) == 2:
                        alpha, fname = item
                    else:
                        continue

                    full_path = os.path.join(int_path, fname)
                    if os.path.exists(full_path):
                        try:
                            data = np.loadtxt(full_path, skiprows=3)
                            color = cmap(i / max(1, len(cp_data) - 1))
                            ax2.plot(data[:, 0], data[:, 2], linewidth=1, color=color, label=f"a={alpha:.1f}")
                            lines_plotted += 1
                        except Exception as read_err:
                            print(f"Could not read {fname}: {read_err}")

            if lines_plotted > 0:
                ax2.invert_yaxis()
                ax2.set_xlabel("x/c")
                ax2.set_ylabel("Cp")
                ax2.set_title("Pressure Coefficients")
                ax2.grid(True, color='#555555')
                if lines_plotted <= 10:
                    ax2.legend(fontsize='x-small', ncol=2, facecolor='#3a3a3a', labelcolor='#f8f8f2')
            else:
                ax2.text(0.5, 0.5, "No Cp Data Available", ha='center', color='#f8f8f2')

        except Exception as e:
            print(f"Cp Plotting Error: {e}")
            ax2.text(0.5, 0.5, f"Plot Error: {e}", ha='center')

        self.xfoil_figure.tight_layout()
        self.xfoil_canvas.draw()
        QMessageBox.information(self, "XFOIL", "Analysis Complete.")

    def init_tab3(self):
        self.tab3 = QWidget()
        self.tabs.addTab(self.tab3, "SU2 Analysis")
        layout = QVBoxLayout(self.tab3)

        self.su2_tabs = QTabWidget()
        layout.addWidget(self.su2_tabs)

        t_mesh = QWidget()
        self.su2_tabs.addTab(t_mesh, "Meshing && Conditions")
        mesh_layout = QGridLayout(t_mesh)

        g_mesh = QGroupBox("1. Mesh Generation")
        gm_layout = QVBoxLayout(g_mesh)
        self.structured_mesh_button = QPushButton("Generate Structured Mesh")
        self.hybrid_mesh_button = QPushButton("Generate Hybrid Mesh")
        self.structured_mesh_button.clicked.connect(self.mesh)
        self.hybrid_mesh_button.clicked.connect(self.hybrid)
        self.structured_mesh_button.setEnabled(False)
        self.hybrid_mesh_button.setEnabled(False)

        h_yp = QHBoxLayout()
        self.yplus_entry = QLineEdit("1.0")
        h_yp.addWidget(QLabel("Target y+:"));
        h_yp.addWidget(self.yplus_entry)

        self.show_gmsh_check = QCheckBox("Show Interactive Gmsh Window")

        gm_layout.addWidget(self.structured_mesh_button)
        gm_layout.addWidget(self.hybrid_mesh_button)
        gm_layout.addLayout(h_yp)
        gm_layout.addWidget(self.show_gmsh_check)

        g_flow = QGroupBox("2. Flow Conditions")
        gf_layout = QFormLayout(g_flow)

        self.su2_re_entry = QLineEdit("1000000")
        self.su2_mach_entry = QLineEdit("0.15")
        self.su2_alpha_min_entry = QLineEdit("0.0")
        self.su2_alpha_max_entry = QLineEdit("5.0")
        self.su2_alpha_step_entry = QLineEdit("1.0")

        gf_layout.addRow("Reynolds:", self.su2_re_entry)
        gf_layout.addRow("Mach:", self.su2_mach_entry)
        gf_layout.addRow("Min AoA:", self.su2_alpha_min_entry)
        gf_layout.addRow("Max AoA:", self.su2_alpha_max_entry)
        gf_layout.addRow("Step:", self.su2_alpha_step_entry)

        load_btn = QPushButton("Load Recommended Settings")
        load_btn.clicked.connect(self.load_recommended_settings)
        gf_layout.addRow(load_btn)

        mesh_layout.addWidget(g_mesh, 0, 0)
        mesh_layout.addWidget(g_flow, 0, 1)

        t_setup = QWidget()
        self.su2_tabs.addTab(t_setup, "Solver Setup && Run")
        setup_layout = QVBoxLayout(t_setup)

        g_cfg = QGroupBox("3. SU2 Configuration")
        cfg_layout = QVBoxLayout(g_cfg)

        h_regime = QHBoxLayout()
        h_regime.addWidget(QLabel("Flow Type:"))
        self.flow_regime_combo = QComboBox()
        self.flow_regime_combo.addItems(["Compressible", "Incompressible"])
        self.flow_regime_combo.currentIndexChanged.connect(self.update_su2_settings_display)
        h_regime.addWidget(self.flow_regime_combo)
        h_regime.addStretch()
        cfg_layout.addLayout(h_regime)

        self.settings_scroll = QScrollArea()
        self.settings_scroll.setWidgetResizable(True)
        self.settings_content = QWidget()
        self.settings_form = QFormLayout(self.settings_content)
        self.settings_scroll.setWidget(self.settings_content)
        cfg_layout.addWidget(self.settings_scroll)

        setup_layout.addWidget(g_cfg, stretch=2)

        g_run = QGroupBox("4. Run Analysis")
        run_layout = QVBoxLayout(g_run)

        h_opts = QHBoxLayout()
        self.live_plot_check = QCheckBox("Live Plotting")
        self.live_plot_check.setChecked(True)
        self.use_mpi_check = QCheckBox("Use MPI")
        self.use_mpi_check.setChecked(True)
        self.use_mpi_check.toggled.connect(self._on_parallel_toggle)

        self.num_cores_entry = QLineEdit("8")
        self.num_cores_entry.setFixedWidth(50)

        h_opts.addWidget(self.live_plot_check)
        h_opts.addWidget(self.use_mpi_check)
        h_opts.addWidget(QLabel("Cores:"))
        h_opts.addWidget(self.num_cores_entry)
        run_layout.addLayout(h_opts)

        h_pol = QHBoxLayout()
        self.polar_filename_entry = QLineEdit("aerodynamic_polar.csv")
        h_pol.addWidget(QLabel("Polar Filename:"));
        h_pol.addWidget(self.polar_filename_entry)
        run_layout.addLayout(h_pol)

        h_btns = QHBoxLayout()
        self.su2_analysis_button = QPushButton("Run SU2 Analysis")
        self.su2_analysis_button.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")
        self.su2_analysis_button.clicked.connect(self.run_su2_workflow_in_thread)
        self.su2_analysis_button.setEnabled(False)

        self.stop_button = QPushButton("STOP")
        self.stop_button.clicked.connect(self.stop_su2_analysis)
        self.stop_button.setEnabled(False)

        h_btns.addWidget(self.su2_analysis_button)
        h_btns.addWidget(self.stop_button)
        run_layout.addLayout(h_btns)

        setup_layout.addWidget(g_run)

        self.update_su2_settings_display()

        t_res = QWidget()
        self.su2_tabs.addTab(t_res, "Results")
        res_layout = QVBoxLayout(t_res)

        self.res_subtabs = QTabWidget()

        self.su2_polar_canv = FigureCanvasQTAgg(Figure(facecolor='#1e1e1e'))
        self.su2_polar_canv.setStyleSheet("background-color: #1e1e1e;")
        self.su2_polar_tool = NavigationToolbar2QT(self.su2_polar_canv, t_res)
        w_pol = QWidget()
        l_pol = QVBoxLayout(w_pol)
        l_pol.addWidget(self.su2_polar_tool);
        l_pol.addWidget(self.su2_polar_canv)
        self.res_subtabs.addTab(w_pol, "Drag Polar")

        self.visual_scroll = QScrollArea()
        self.visual_scroll.setWidgetResizable(True)
        self.visual_content = QWidget()
        self.visual_layout = QVBoxLayout(self.visual_content)
        self.visual_scroll.setWidget(self.visual_content)
        self.res_subtabs.addTab(self.visual_scroll, "Flow Visualization")

        res_layout.addWidget(self.res_subtabs)

    def update_su2_settings_display(self):
        while self.settings_form.count():
            child = self.settings_form.takeAt(0)
            if child.widget(): child.widget().deleteLater()

        self.su2_setting_widgets = {}
        regime = self.flow_regime_combo.currentText()
        settings = SU2_INCOMPRESSIBLE_SETTINGS if regime == "Incompressible" else SU2_COMPRESSIBLE_SETTINGS

        for key, options in settings.items():
            if key in ['ITER', 'EXT_ITER', 'CONV_RESIDUAL_MINVAL']: continue

            if isinstance(options, list):
                combo = QComboBox()
                combo.addItems(options)
                self.settings_form.addRow(f"{key}:", combo)
                self.su2_setting_widgets[key] = combo
                if key in ['KIND_TURB_MODEL', 'SOLVER']:
                    combo.currentIndexChanged.connect(self.check_turbulence_status)
            elif isinstance(options, dict) and key == 'CONV_NUM_METHOD_FLOW':
                self.conv_category_combo = QComboBox()
                self.conv_category_combo.addItems(options.keys())
                self.conv_category_combo.currentTextChanged.connect(
                    self._on_conv_method_category_change
                )
                self.settings_form.addRow("CONV_SCHEME_CATEGORY:", self.conv_category_combo)

                self.conv_scheme_combo = QComboBox()
                self.settings_form.addRow("CONV_NUM_METHOD_FLOW:", self.conv_scheme_combo)

                self.su2_setting_widgets['CONV_NUM_METHOD_FLOW'] = self.conv_scheme_combo

                self._on_conv_method_category_change()

        self.cfl_input = QLineEdit("1.0")
        self.settings_form.addRow("CFL_NUMBER:", self.cfl_input)
        self.su2_setting_widgets['CFL_NUMBER'] = self.cfl_input

        self.conv_field = QComboBox()

        if self.flow_regime_combo.currentText() == "Incompressible":
            self.conv_field.addItems([
                "RMS_PRESSURE"
            ])
        else:
            self.conv_field.addItems([
                "RMS_DENSITY",
                "REL_RMS_DENSITY",
                "RMS_ENERGY"
            ])

        self.settings_form.addRow("CONV_FIELD:", self.conv_field)
        self.su2_setting_widgets['CONV_FIELD'] = self.conv_field

        self.conv_settings = ConvergenceSettings()
        self.settings_form.addRow(self.conv_settings)

        self.check_turbulence_status()

    def _on_conv_method_category_change(self):
        category = self.conv_category_combo.currentText()

        settings = (
            SU2_INCOMPRESSIBLE_SETTINGS
            if self.flow_regime_combo.currentText() == "Incompressible"
            else SU2_COMPRESSIBLE_SETTINGS
        )

        schemes = settings['CONV_NUM_METHOD_FLOW'].get(category, [])

        self.conv_scheme_combo.blockSignals(True)
        self.conv_scheme_combo.clear()
        self.conv_scheme_combo.addItems(schemes)
        self.conv_scheme_combo.blockSignals(False)

    def check_turbulence_status(self):
        pass

    def load_recommended_settings(self):
        try:
            mach = float(self.su2_mach_entry.text())
            reynolds = float(self.su2_re_entry.text())
        except ValueError:
            QMessageBox.warning(
                self,
                "Input Error",
                "Please enter valid Mach and Reynolds numbers."
            )
            return

        selected_file = ""
        new_regime = ""

        if mach >= 1.0:
            selected_file = "supersonic.cfg"
            new_regime = "Compressible"
        elif 0.7 <= mach < 1.0:
            selected_file = "Transonic.cfg"
            new_regime = "Compressible"
        else:
            new_regime = "Incompressible"
            if reynolds >= 1e6:
                selected_file = "HighReIncomp.cfg"
            else:
                selected_file = "LowReIncomp.cfg"

        cfg_path = os.path.join(self.script_dir, "Configuration_Files", selected_file)
        print(f"Loading recommended settings from: {cfg_path}")

        self.loaded_cfg_settings = self.parse_su2_cfg(cfg_path)
        if not self.loaded_cfg_settings:
            self.loaded_cfg_settings = {}
            return

        self.flow_regime_combo.setCurrentText(new_regime)
        self.update_su2_settings_display()
        self.apply_loaded_settings()

        self.loaded_cfg_settings = {}

        QMessageBox.information(
            self,
            "Success",
            f"Successfully loaded settings from {selected_file}."
        )

    def apply_loaded_settings(self):
        if not self.loaded_cfg_settings:
            return

        print("Applying loaded settings to GUI...")

        for key, widget in self.su2_setting_widgets.items():
            ukey = key.upper()
            if ukey not in self.loaded_cfg_settings:
                continue

            value = self.loaded_cfg_settings[ukey]

            try:
                if ukey == "CONV_NUM_METHOD_FLOW":
                    settings = (
                        SU2_INCOMPRESSIBLE_SETTINGS
                        if self.flow_regime_combo.currentText() == "Incompressible"
                        else SU2_COMPRESSIBLE_SETTINGS
                    )

                    applied = False
                    for category, schemes in settings['CONV_NUM_METHOD_FLOW'].items():
                        if value in schemes:
                            self.conv_category_combo.setCurrentText(category)
                            self._on_conv_method_category_change()
                            self.conv_scheme_combo.setCurrentText(value)
                            print(f"  Applied {ukey} = {value} (Category: {category})")
                            applied = True
                            break

                    if not applied:
                        print(f"  Warning: Failed to apply {ukey}: {value}")
                    continue
            except Exception as e:
                print(f"  Error applying {ukey}: {e}")

    def parse_su2_cfg(self, file_path):
        settings = {}

        if not os.path.exists(file_path):
            QMessageBox.critical(
                self,
                "Error",
                f"Config file not found:\n{file_path}"
            )
            return None

        try:
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("%"):
                        continue

                    m = re.match(r'^([A-Za-z0-9_]+)\s*=\s*(.*)', line)
                    if m:
                        key = m.group(1).strip().upper()
                        value = m.group(2).split("%")[0].strip()
                        settings[key] = value

        except Exception as e:
            QMessageBox.critical(self, "Parse Error", str(e))
            return None

        print(f"Parsed {len(settings)} settings from {file_path}")
        return settings

    def _on_parallel_toggle(self):
        self.num_cores_entry.setEnabled(self.use_mpi_check.isChecked())

    def mesh(self):
        self.run_gmsh_process(generate_mesh)

    def hybrid(self):
        self.run_gmsh_process(generate_hybrid)

    def run_gmsh_process(self, func):
        try:
            re_val = float(self.su2_re_entry.text())
            mach = float(self.su2_mach_entry.text())
            yp = float(self.yplus_entry.text())

            if func == generate_mesh:
                args = (self.xcoords, self.ycoords, re_val, mach)
            else:
                args = (self.upper_surface, self.lower_surface, re_val, mach)

            kwargs = {'y_plus': yp, 'show_graphics': self.show_gmsh_check.isChecked(), 'hide_output': False}

            p = multiprocessing.Process(target=func, args=args, kwargs=kwargs)
            p.start()
            QMessageBox.information(self, "Meshing", "Gmsh started in background.")
            self.su2_analysis_button.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def run_su2_workflow_in_thread(self):
        self.su2_analysis_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        gui_settings = {}
        for k, w in self.su2_setting_widgets.items():
            if isinstance(w, QComboBox):
                gui_settings[k] = w.currentText()
            elif isinstance(w, QLineEdit):
                gui_settings[k] = w.text()

        res, iter_val = self.conv_settings.get_config_string()
        gui_settings['CONV_RESIDUAL_MINVAL'] = res
        gui_settings['ITER'] = iter_val
        gui_settings['EXT_ITER'] = iter_val

        try:
            params = {
                'reynolds': float(self.su2_re_entry.text()),
                'mach': float(self.su2_mach_entry.text()),
                'alpha_min': float(self.su2_alpha_min_entry.text()),
                'alpha_max': float(self.su2_alpha_max_entry.text()),
                'alpha_step': float(self.su2_alpha_step_entry.text()),
                'base_output_dir': self.int_path_entry.text(),
                'flow_regime': self.flow_regime_combo.currentText(),
                'gui_settings': gui_settings,
                'mesh_filepath': os.path.join(os.getcwd(), "airfoil.su2"),
                'polar_filename': self.polar_filename_entry.text(),
                'enable_live_plotting': self.live_plot_check.isChecked()
            }
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid numeric inputs")
            self.su2_analysis_button.setEnabled(True)
            return

        threading.Thread(target=self.su2_worker, kwargs=params, daemon=True).start()

    def su2_worker(self, **kwargs):
        def gui_cb(res):
            self.plot_su2_signal.emit(res)

        def plot_cb(q, a, u):
            self.request_plot_window_signal.emit(q, a, u)
            return None

        kwargs['gui_update_callback'] = gui_cb
        kwargs['plot_window_callback'] = plot_cb
        kwargs['su2_runner'] = self.su2_runner

        try:
            self.su2_runner.update_parallel_settings(
                self.use_mpi_check.isChecked(),
                int(self.num_cores_entry.text())
            )
            execute_su2_analysis_workflow(**kwargs)
        except Exception as e:
            print(f"Workflow Crash: {e}")
        finally:
            self.su2_finished_signal.emit()

    def on_su2_analysis_finish(self):
        self.su2_analysis_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        QMessageBox.information(self, "SU2", "Analysis Complete.")

    def stop_su2_analysis(self):
        self.su2_runner.stop()

    def open_live_plotter(self, queue, aoa, unsteady):
        win = LivePlotterWindow(queue, aoa, unsteady)
        win.show()
        self.live_windows.append(win)

    def plot_su2_results(self, results):
        self.su2_polar_canv.figure.clear()
        ax = self.su2_polar_canv.figure.add_subplot(111)
        ax.set_facecolor('#2b2b2b')
        ax.tick_params(colors='#f8f8f2')
        ax.xaxis.label.set_color('#f8f8f2')
        ax.yaxis.label.set_color('#f8f8f2')
        ax.title.set_color('#f8f8f2')
        for spine in ax.spines.values():
            spine.set_edgecolor('#555555')
        AoA, Cl, Cd, Cm = extract_su2_polar_data(results)

        if Cl and Cd and not all(np.isnan(Cl)):
            ax.plot(Cd, Cl, 'o-', color='#62b6ef')
            ax.set_xlabel("Cd");
            ax.set_ylabel("Cl");
            ax.set_title("SU2 Polar")
            ax.grid(True, color='#555555')
        self.su2_polar_canv.draw()

        while self.visual_layout.count():
            child = self.visual_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()

        if not results: return

        last_res = results[-1]
        if last_res[2]:
            run_dir = os.path.dirname(last_res[2])
            images = [f for f in os.listdir(run_dir) if f.endswith('.png')]

            for img_name in sorted(images):
                lbl_name = QLabel(img_name)
                lbl_name.setStyleSheet("font-weight: bold;")
                self.visual_layout.addWidget(lbl_name)

                lbl_img = QLabel()
                pix = QPixmap(os.path.join(run_dir, img_name))
                if not pix.isNull():
                    pix = pix.scaledToWidth(600, Qt.TransformationMode.SmoothTransformation)
                    lbl_img.setPixmap(pix)
                self.visual_layout.addWidget(lbl_img)

    def open_dir(self, line_edit):
        d = QFileDialog.getExistingDirectory(self, "Select Directory")
        if d: line_edit.setText(d)

    def open_file(self, line_edit):
        f, _ = QFileDialog.getOpenFileName(self, "Select File")
        if f: line_edit.setText(f)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    font = QFont("Segoe UI", 9)
    app.setFont(font)

    from PyQt6.QtGui import QPalette, QColor
    from PyQt6.QtCore import Qt

    dark_palette = QPalette()
    dark_palette.setColor(QPalette.ColorRole.Window,          QColor(43, 43, 43))
    dark_palette.setColor(QPalette.ColorRole.WindowText,      QColor(248, 248, 242))
    dark_palette.setColor(QPalette.ColorRole.Base,            QColor(30, 30, 30))
    dark_palette.setColor(QPalette.ColorRole.AlternateBase,   QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ToolTipBase,     QColor(248, 248, 242))
    dark_palette.setColor(QPalette.ColorRole.ToolTipText,     QColor(248, 248, 242))
    dark_palette.setColor(QPalette.ColorRole.Text,            QColor(248, 248, 242))
    dark_palette.setColor(QPalette.ColorRole.Button,          QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ButtonText,      QColor(248, 248, 242))
    dark_palette.setColor(QPalette.ColorRole.BrightText,      QColor(255, 85, 85))
    dark_palette.setColor(QPalette.ColorRole.Link,            QColor(98, 182, 239))
    dark_palette.setColor(QPalette.ColorRole.Highlight,       QColor(98, 182, 239))
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(127, 127, 127))
    dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,       QColor(127, 127, 127))
    dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(127, 127, 127))
    app.setPalette(dark_palette)
    app.setStyleSheet("QToolTip { color: #f8f8f2; background-color: #2b2b2b; border: 1px solid #62b6ef; }")

    window = FalconApp()
    window.show()
    sys.exit(app.exec())