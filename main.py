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
    QFileDialog, QGroupBox, QScrollArea, QMessageBox, QFrame, QFormLayout, QAbstractItemView
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import QPixmap, QImage, QFont

import matplotlib

matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib.cm as cm

from Scripts.Solver_and_Results.su2_analyzer import (
    SU2Runner, execute_su2_analysis_workflow, extract_su2_polar_data, extract_surface_yplus,
    SU2_INCOMPRESSIBLE_SETTINGS, SU2_COMPRESSIBLE_SETTINGS, template_passthrough
)
from Scripts.Geometry import read_airfoil
from Scripts.Geometry.parsec import Parsec
from Scripts.Geometry.cst import CST
from Scripts.Geometry.interpolate import Interpolate
# The "Generate Hybrid Mesh" button uses Scripts/Meshing/hybrid.py. An alternative two-pass mesher is kept in
# Scripts/Meshing/hybrid_v2.py; import generate_hybrid from there to use it instead.
from Scripts.Meshing.hybrid import generate_hybrid
from Scripts.Meshing.meshing import generate_mesh
from Scripts.Meshing.wall_spacing import measure_wall_spacing, design_yplus
from Scripts.Solver_and_Results import xfoil1
from Scripts.Solver_and_Results.live_plotter import LivePlotterWindow
from Scripts.Batch.batch_runner import Case, load_cases, run_batch


XFOIL_MAX_POINTS_PER_SURFACE = 500
BATCH_CSV_COLUMNS = ["airfoil", "solver", "re", "mach", "alpha_min", "alpha_max", "alpha_step", "n_points", "y_plus",
                     "mesh", "repanel"]


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
    batch_progress_signal = pyqtSignal(str)
    batch_finished_signal = pyqtSignal(object, str)

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
        self.template_extras = {}

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
        self.init_tab4()

        self.xfoil_finished_signal.connect(self.on_xfoil_results)
        self.su2_finished_signal.connect(self.on_su2_analysis_finish)
        self.plot_su2_signal.connect(self.plot_su2_results)
        self.request_plot_window_signal.connect(self.open_live_plotter)
        self.batch_progress_signal.connect(self.on_batch_progress)
        self.batch_finished_signal.connect(self.on_batch_finished)

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
        self.su2_tabs.addTab(t_mesh, "Conditions && Meshing")
        mesh_layout = QGridLayout(t_mesh)

        g_mesh = QGroupBox("2. Mesh Generation")
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

        # Two different numbers, both worth showing. Design y+ is read straight off the generated mesh and only
        # confirms it carries the spacing it was asked for. Achieved y+ comes from the solution, because it needs
        # the wall shear - the flat-plate correlation behind the target is optimistic near the suction peak.
        self._design_yplus = None
        self._achieved_yplus_max = None
        self._mesh_timers = set()
        self._achieved_yplus = None
        self.achieved_yplus_label = QLabel()
        self.achieved_yplus_label.setWordWrap(True)
        self._refresh_yplus_label()

        self.show_gmsh_check = QCheckBox("Show Interactive Gmsh Window")

        gm_layout.addWidget(self.structured_mesh_button)
        gm_layout.addWidget(self.hybrid_mesh_button)
        gm_layout.addLayout(h_yp)
        gm_layout.addWidget(self.achieved_yplus_label)
        gm_layout.addWidget(self.show_gmsh_check)

        g_flow = QGroupBox("1. Flow Conditions")
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

        # Conditions on the left, mesh generation on the right: Re, Mach and y+ set the wall spacing.
        mesh_layout.addWidget(g_flow, 0, 0)
        mesh_layout.addWidget(g_mesh, 0, 1)

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
        # The Batch tab shows this same panel (it is moved there while the Batch solver-setup tab is open).
        self.su2_cfg_group = g_cfg
        self.su2_setup_layout = setup_layout

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
        # Template values without a widget belong to the template the panel was loaded from; rebuilding the panel
        # (a regime change) drops them.
        self.template_extras = {}
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

    def load_recommended_settings(self, *, mach=None, reynolds=None):  # keyword-only: clicked passes `checked`
        """Load the template for a Mach and Reynolds number into Solver Setup & Run; by default those of the
        Conditions tab (the Batch tab passes its own)."""
        try:
            mach = float(self.su2_mach_entry.text()) if mach is None else float(mach)
            reynolds = float(self.su2_re_entry.text()) if reynolds is None else float(reynolds)
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
        used = set()

        for key, widget in self.su2_setting_widgets.items():
            ukey = key.upper()
            if ukey not in self.loaded_cfg_settings:
                continue

            value = self.loaded_cfg_settings[ukey]
            used.add(ukey)

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

                if isinstance(widget, QComboBox):
                    if widget.findText(value) >= 0:
                        widget.setCurrentText(value)
                        print(f"  Applied {ukey} = {value}")
                    else:
                        print(f"  Warning: {ukey} = {value} is not an option here, kept {widget.currentText()}")
                elif isinstance(widget, QLineEdit):
                    widget.setText(value)
                    print(f"  Applied {ukey} = {value}")
            except Exception as e:
                print(f"  Error applying {ukey}: {e}")

        # The transonic template counts physical time steps (TIME_ITER), the others iterations (ITER).
        iterations = self.loaded_cfg_settings.get("ITER") or self.loaded_cfg_settings.get("TIME_ITER")
        if iterations:
            self.conv_settings.max_iter_input.setText(iterations)
        if "CONV_RESIDUAL_MINVAL" in self.loaded_cfg_settings:
            self.conv_settings.res_min_input.setText(self.loaded_cfg_settings["CONV_RESIDUAL_MINVAL"])
        used.update({"ITER", "TIME_ITER", "EXT_ITER", "CONV_RESIDUAL_MINVAL"})

        self.template_extras = template_passthrough(self.loaded_cfg_settings, used)
        print(f"  Passed to the config writer without a widget: {', '.join(sorted(self.template_extras))}")

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

            # Read the design y+ off the mesh once gmsh has exited. Waiting on the process rather than on the
            # file avoids reading a partly written mesh, and with the interactive window on it also means the
            # file has already been through its NMARK rewrite.
            mesh_path = os.path.join(os.getcwd(), "airfoil.su2")
            timer = QTimer(self)

            def check_finished():
                if p.is_alive():
                    return
                timer.stop()
                self._mesh_timers.discard(timer)
                if os.path.exists(mesh_path):
                    self.update_design_yplus(mesh_path)

            timer.timeout.connect(check_finished)
            timer.start(1000)
            self._mesh_timers.add(timer)

            QMessageBox.information(self, "Meshing", "Gmsh started in background.")
            self.su2_analysis_button.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def collect_su2_settings(self):
        """The settings handed to prepare_su2_config: the loaded template's values that have no widget, then every
        widget (so the user's edits win), then the convergence fields."""
        gui_settings = dict(self.template_extras)
        for k, w in self.su2_setting_widgets.items():
            if isinstance(w, QComboBox):
                gui_settings[k] = w.currentText()
            elif isinstance(w, QLineEdit):
                gui_settings[k] = w.text()

        res, iter_val = self.conv_settings.get_config_string()
        gui_settings['CONV_RESIDUAL_MINVAL'] = res
        gui_settings['ITER'] = iter_val
        gui_settings['EXT_ITER'] = iter_val
        return gui_settings

    def run_su2_workflow_in_thread(self):
        self.su2_analysis_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        gui_settings = self.collect_su2_settings()

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

    def _refresh_yplus_label(self):
        design = self._design_yplus or "mesh not generated yet"
        achieved = self._achieved_yplus or "run SU2 to measure"
        self.achieved_yplus_label.setText(f"Design y+ (mesh): {design}\nAchieved y+ (solution): {achieved}")
        warn = isinstance(self._achieved_yplus, str) and self._achieved_yplus.startswith("max") and \
            self._achieved_yplus_max is not None and self._achieved_yplus_max > 5.0
        self.achieved_yplus_label.setStyleSheet("color: #ff8f5a;" if warn else "color: #999999;")

    def update_design_yplus(self, mesh_path):
        """Wall spacing read back off the generated mesh. Needs no solver."""
        try:
            coords, spacing = measure_wall_spacing(mesh_path)
        except Exception as e:
            print(f"[y+] Could not measure wall spacing: {e}")
            return
        if spacing is None:
            return
        valid = np.isfinite(spacing)
        if not np.any(valid):
            return
        try:
            re_val = float(self.su2_re_entry.text())
            target = float(self.yplus_entry.text())
        except ValueError:
            return
        yp = design_yplus(spacing[valid], re_val)
        self._design_yplus = (f"median {np.median(yp):.3f} (range {yp.min():.3f}-{yp.max():.3f}) "
                              f"against target {target:g}, first cell {np.median(spacing[valid]):.3e}")
        self._refresh_yplus_label()
        print(f"[y+] design, from the mesh: {self._design_yplus}")

    def update_achieved_yplus(self, results):
        """Worst-case achieved y+ across the AoA sweep, read from each run's surface CSV."""
        per_aoa = [(aoa, extract_surface_yplus(surface)) for aoa, _, _, surface in results]
        measured = [(aoa, yp) for aoa, yp in per_aoa if yp]

        if not measured:
            self._achieved_yplus = "unavailable (no surface VTU with Y_Plus)"
            self._achieved_yplus_max = None
            self._refresh_yplus_label()
            return

        worst_aoa, worst = max(measured, key=lambda item: item[1]['max'])
        self._achieved_yplus_max = worst['max']
        # Wall-resolved turbulence models want the first cell inside the viscous sublayer.
        self._achieved_yplus = (f"max {worst['max']:.2f} at AoA {worst_aoa:.2f}, "
                                f"99th pct {worst['p99']:.2f}, median {worst['median']:.2f}")
        self._refresh_yplus_label()

        for aoa, yp in measured:
            print(f"[y+] AoA {aoa:.2f}: max {yp['max']:.2f}, 99th pct {yp['p99']:.2f}, "
                  f"median {yp['median']:.2f}")

    def plot_su2_results(self, results):
        self.update_achieved_yplus(results)
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

    # ------------------------------------------------------------------ Dataset generation (batch runner)
    def init_tab4(self):
        """Batch tab: the same steps and settings as the other tabs, for many airfoils at once, with no
        plots or results views; one solver per batch; the SU2 solver settings are those of Solver Setup & Run."""
        self.tab4 = QWidget()
        self.tabs.addTab(self.tab4, "Batch")
        outer = QVBoxLayout(self.tab4)

        h_solver = QHBoxLayout()
        h_solver.addWidget(QLabel("Solver for this batch:"))
        self.batch_solver_combo = QComboBox()
        self.batch_solver_combo.addItems(["XFOIL", "SU2"])
        self.batch_solver_combo.currentIndexChanged.connect(self._batch_update_enabled)
        h_solver.addWidget(self.batch_solver_combo)
        h_solver.addStretch()
        outer.addLayout(h_solver)

        self.batch_tabs = QTabWidget()
        outer.addWidget(self.batch_tabs)

        # 1. Parameterization: any folder, several airfoils at once.
        t_par = QWidget()
        self.batch_tabs.addTab(t_par, "1. Parameterization")
        lp = QGridLayout(t_par)
        g_sel = QGroupBox("Select Airfoils")
        gs = QVBoxLayout(g_sel)
        h_dir = QHBoxLayout()
        h_dir.addWidget(QLabel("Airfoil folder:"))
        self.batch_folder_entry = QLineEdit(self.directory)
        self.batch_folder_entry.editingFinished.connect(self._batch_scan_folder)
        b_dir = QPushButton("Browse...")
        b_dir.clicked.connect(self._batch_browse_folder)
        h_dir.addWidget(self.batch_folder_entry)
        h_dir.addWidget(b_dir)
        gs.addLayout(h_dir)
        h_search = QHBoxLayout()
        h_search.addWidget(QLabel("Search:"))
        self.batch_search = QLineEdit()
        self.batch_search.textChanged.connect(self._batch_filter)
        h_search.addWidget(self.batch_search)
        gs.addLayout(h_search)
        self.batch_airfoil_list = QListWidget()
        # Ctrl+click, Shift+click and Ctrl+A work natively in extended selection.
        self.batch_airfoil_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.batch_airfoil_list.itemSelectionChanged.connect(self._batch_update_count)
        gs.addWidget(self.batch_airfoil_list, stretch=1)
        h_sel = QHBoxLayout()
        b_all = QPushButton("Select All (Ctrl+A)")
        b_all.clicked.connect(self._batch_select_shown)
        b_none = QPushButton("Clear")
        b_none.clicked.connect(self.batch_airfoil_list.clearSelection)
        self.batch_selected_label = QLabel()
        for wdg in (b_all, b_none, self.batch_selected_label):
            h_sel.addWidget(wdg)
        h_sel.addStretch()
        gs.addLayout(h_sel)
        lp.addWidget(g_sel, 0, 0)
        g_ctrl = QGroupBox("Controls")
        gc = QFormLayout(g_ctrl)
        self.batch_repanel_check = QCheckBox("Fit CST/PARSEC and re-panel (as Analyze does)")
        self.batch_repanel_check.setChecked(True)
        self.batch_repanel_check.setToolTip("Unchecked: each file's own coordinates are used as they are")
        self.batch_repanel_check.toggled.connect(self._batch_update_enabled)
        self.batch_npoints_entry = QLineEdit(self.n_points_entry.text())
        self.batch_npoints_entry.setToolTip(f"Points per surface. XFOIL loads at most 1000 points, so at most "
                                            f"{XFOIL_MAX_POINTS_PER_SURFACE} for XFOIL.")
        gc.addRow(self.batch_repanel_check)
        gc.addRow("Desired points:", self.batch_npoints_entry)
        lp.addWidget(g_ctrl, 0, 1, Qt.AlignmentFlag.AlignTop)
        lp.setColumnStretch(0, 2)
        lp.setColumnStretch(1, 1)

        # 2. XFOIL
        t_xf = QWidget()
        self.batch_tabs.addTab(t_xf, "2. XFOIL")
        lx = QGridLayout(t_xf)
        g_xf = QGroupBox("Input Parameters")
        fx = QFormLayout(g_xf)
        self.batch_xfoil_entry = QLineEdit(self.xfoil_path_entry.text())
        b_xf = QPushButton("...")
        b_xf.clicked.connect(lambda: self.open_file(self.batch_xfoil_entry))
        h_xf = QHBoxLayout()
        h_xf.addWidget(self.batch_xfoil_entry)
        h_xf.addWidget(b_xf)
        fx.addRow("XFOIL Executable:", h_xf)
        self.batch_xf_re_entry = QLineEdit("50000, 100000")
        self.batch_xf_mach_entry = QLineEdit("0.0")
        self.batch_xf_amin_entry = QLineEdit("-10")
        self.batch_xf_amax_entry = QLineEdit("15")
        self.batch_xf_astep_entry = QLineEdit("1")
        fx.addRow("Reynolds Numbers:", self.batch_xf_re_entry)
        fx.addRow("Mach Numbers:", self.batch_xf_mach_entry)
        fx.addRow("Min Alpha:", self.batch_xf_amin_entry)
        fx.addRow("Max Alpha:", self.batch_xf_amax_entry)
        fx.addRow("Step:", self.batch_xf_astep_entry)
        self.batch_workers_entry = QLineEdit(str(max(1, (os.cpu_count() or 2) // 2)))
        self.batch_xfoil_timeout_entry = QLineEdit("60")
        self.batch_xfoil_timeout_entry.setToolTip("Some airfoils hang XFOIL; such a case is stopped after this long")
        fx.addRow("Parallel Cases:", self.batch_workers_entry)
        fx.addRow("Time Limit per Case (s):", self.batch_xfoil_timeout_entry)
        fx.addRow(QLabel("Re and Mach take several values, separated by commas."))
        lx.addWidget(g_xf, 0, 0, Qt.AlignmentFlag.AlignTop)
        lx.setColumnStretch(1, 1)

        # 3. SU2 conditions and meshing
        t_cm = QWidget()
        self.batch_tabs.addTab(t_cm, "3. SU2 Conditions && Meshing")
        lc = QGridLayout(t_cm)
        g_flow = QGroupBox("1. Flow Conditions")
        ff = QFormLayout(g_flow)
        self.batch_su2_re_entry = QLineEdit(self.su2_re_entry.text())
        self.batch_su2_mach_entry = QLineEdit(self.su2_mach_entry.text())
        self.batch_su2_amin_entry = QLineEdit(self.su2_alpha_min_entry.text())
        self.batch_su2_amax_entry = QLineEdit(self.su2_alpha_max_entry.text())
        self.batch_su2_astep_entry = QLineEdit(self.su2_alpha_step_entry.text())
        ff.addRow("Reynolds:", self.batch_su2_re_entry)
        ff.addRow("Mach:", self.batch_su2_mach_entry)
        ff.addRow("Min AoA:", self.batch_su2_amin_entry)
        ff.addRow("Max AoA:", self.batch_su2_amax_entry)
        ff.addRow("Step:", self.batch_su2_astep_entry)
        b_rec = QPushButton("Load Recommended Settings")
        b_rec.setToolTip("Loads the template for the first Mach and Reynolds number into Solver Setup & Run")
        b_rec.clicked.connect(self._batch_load_recommended)
        ff.addRow(b_rec)
        ff.addRow(QLabel("Re and Mach take several values, separated by commas."))
        lc.addWidget(g_flow, 0, 0, Qt.AlignmentFlag.AlignTop)
        g_mesh = QGroupBox("2. Mesh Generation")
        fm = QFormLayout(g_mesh)
        self.batch_mesh_combo = QComboBox()
        self.batch_mesh_combo.addItems(["Structured Mesh", "Hybrid Mesh"])
        self.batch_yplus_entry = QLineEdit(self.yplus_entry.text())
        fm.addRow("Mesh:", self.batch_mesh_combo)
        fm.addRow("Target y+:", self.batch_yplus_entry)
        lc.addWidget(g_mesh, 0, 1, Qt.AlignmentFlag.AlignTop)

        # 4. SU2 solver setup: the SU2 Configuration panel of SU2 Analysis > Solver Setup & Run itself, moved here
        # while this tab is open, so both tabs always edit the same settings.
        t_ss = QWidget()
        self.batch_tabs.addTab(t_ss, "4. SU2 Solver Setup")
        self.batch_ss_layout = QVBoxLayout(t_ss)
        g_par = QGroupBox("Parallel Run")
        h_par = QHBoxLayout(g_par)
        self.batch_mpi_check = QCheckBox("Use MPI")
        self.batch_mpi_check.setChecked(self.use_mpi_check.isChecked())
        self.batch_cores_entry = QLineEdit(self.num_cores_entry.text())
        self.batch_cores_entry.setFixedWidth(50)
        self.batch_mpi_check.toggled.connect(self.batch_cores_entry.setEnabled)
        self.batch_su2_timeout_entry = QLineEdit("4")
        self.batch_su2_timeout_entry.setFixedWidth(50)
        h_par.addWidget(self.batch_mpi_check)
        h_par.addWidget(QLabel("Cores:"))
        h_par.addWidget(self.batch_cores_entry)
        h_par.addWidget(QLabel("Time Limit per Angle (h):"))
        h_par.addWidget(self.batch_su2_timeout_entry)
        h_par.addStretch()
        self.batch_ss_layout.addWidget(g_par)

        # 5. Run
        t_run = QWidget()
        self.batch_tabs.addTab(t_run, "5. Run")
        lr = QVBoxLayout(t_run)
        g_run = QGroupBox("4. Run Batch")
        fr = QFormLayout(g_run)
        self.batch_out_entry = QLineEdit(os.path.join(self.script_dir, "output", "dataset"))
        self.batch_out_entry.setToolTip("Running again into the same folder skips the cases that already finished")
        b_out = QPushButton("...")
        b_out.clicked.connect(lambda: self.open_dir(self.batch_out_entry))
        h_out = QHBoxLayout()
        h_out.addWidget(self.batch_out_entry)
        h_out.addWidget(b_out)
        fr.addRow("Output Folder:", h_out)
        self.batch_rerun_check = QCheckBox("Re-run failed / timed-out cases")
        self.batch_dry_check = QCheckBox("Dry run (SU2: write meshes and configs only)")
        fr.addRow(self.batch_rerun_check)
        fr.addRow(self.batch_dry_check)
        h_btn = QHBoxLayout()
        b_preview = QPushButton("Preview Cases")
        b_preview.clicked.connect(self.batch_preview)
        self.batch_start_button = QPushButton("Run Batch")
        self.batch_start_button.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")
        self.batch_start_button.clicked.connect(self.batch_start)
        self.batch_stop_button = QPushButton("STOP")
        self.batch_stop_button.setEnabled(False)
        self.batch_stop_button.clicked.connect(self.batch_stop)
        b_open = QPushButton("Open Output Folder")
        b_open.clicked.connect(self.batch_open_folder)
        for b in (b_preview, self.batch_start_button, self.batch_stop_button, b_open):
            h_btn.addWidget(b)
        fr.addRow(h_btn)
        self.batch_cases_label = QLabel("No cases yet")
        self.batch_cases_label.setWordWrap(True)
        self.batch_progress_label = QLabel("Idle")
        self.batch_progress_label.setWordWrap(True)
        fr.addRow(self.batch_cases_label)
        fr.addRow(self.batch_progress_label)
        lr.addWidget(g_run)
        lr.addStretch()

        self._batch_stop_event = threading.Event()
        self._batch_thread = None
        self._batch_counts = {}
        self._batch_total = 0
        self.batch_tabs.currentChanged.connect(lambda _: self._place_su2_cfg_panel())
        self.tabs.currentChanged.connect(lambda _: self._place_su2_cfg_panel())
        self.su2_tabs.currentChanged.connect(lambda _: self._place_su2_cfg_panel())
        self._batch_scan_folder()
        self._batch_update_enabled()

    def _batch_browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select Airfoil Folder", self.batch_folder_entry.text())
        if d:
            self.batch_folder_entry.setText(d)
            self._batch_scan_folder()

    def _batch_scan_folder(self):
        folder = self.batch_folder_entry.text().strip()
        files = sorted(f for f in os.listdir(folder)
                       if f.lower().endswith(".dat") and os.path.isfile(os.path.join(folder, f))) \
            if os.path.isdir(folder) else []
        self.batch_airfoil_list.clear()
        self.batch_airfoil_list.addItems(files)
        self._batch_filter(self.batch_search.text())
        self._batch_update_count()

    def _batch_filter(self, text):
        text = text.strip().lower()
        for i in range(self.batch_airfoil_list.count()):
            item = self.batch_airfoil_list.item(i)
            item.setHidden(bool(text) and text not in item.text().lower())

    def _batch_select_shown(self):
        for i in range(self.batch_airfoil_list.count()):
            item = self.batch_airfoil_list.item(i)
            if not item.isHidden():
                item.setSelected(True)

    def _batch_update_count(self):
        n = self.batch_airfoil_list.count()
        folder = self.batch_folder_entry.text().strip()
        if not os.path.isdir(folder):
            self.batch_selected_label.setText("Folder not found")
        else:
            self.batch_selected_label.setText(f"{len(self.batch_airfoil_list.selectedItems())} of {n} selected")

    def _batch_update_enabled(self):
        su2 = self.batch_solver_combo.currentText() == "SU2"
        self.batch_tabs.setTabEnabled(1, not su2)
        for i in (2, 3):
            self.batch_tabs.setTabEnabled(i, su2)
        self.batch_npoints_entry.setEnabled(self.batch_repanel_check.isChecked())
        self.batch_dry_check.setEnabled(su2)

    def _place_su2_cfg_panel(self):
        """Show the one SU2 Configuration panel in whichever tab is open: the Batch tab's solver-setup page or
        SU2 Analysis > Solver Setup & Run."""
        in_batch = self.tabs.currentWidget() is self.tab4 and self.batch_tabs.currentIndex() == 3
        target = self.batch_ss_layout if in_batch else self.su2_setup_layout
        if self.su2_cfg_group.parentWidget() is target.parentWidget():
            return
        source = self.su2_setup_layout if in_batch else self.batch_ss_layout
        source.removeWidget(self.su2_cfg_group)
        target.insertWidget(0, self.su2_cfg_group, stretch=2)
        self.su2_cfg_group.show()

    def _batch_load_recommended(self):
        try:
            re_ = self._parse_numbers(self.batch_su2_re_entry.text(), "Reynolds")[0]
            mach = self._parse_numbers(self.batch_su2_mach_entry.text(), "Mach")[0]
        except ValueError as e:
            QMessageBox.warning(self, "Batch", str(e))
            return
        self.load_recommended_settings(mach=mach, reynolds=re_)
        self._batch_update_enabled()
        print(f"[Batch] Recommended settings for M {mach:g}, Re {re_:g} loaded into Solver Setup & Run")

    @staticmethod
    def _parse_numbers(text, name):
        values = [v for v in re.split(r"[,\s;]+", text.strip()) if v]
        if not values:
            raise ValueError(f"{name}: enter at least one value")
        try:
            return [float(v) for v in values]
        except ValueError:
            raise ValueError(f"{name}: '{text}' is not a list of numbers") from None

    @classmethod
    def _parse_one(cls, text, name):
        values = cls._parse_numbers(text, name)
        if len(values) != 1:
            raise ValueError(f"{name}: enter a single value")
        return values[0]

    def batch_cases(self):
        """The selected airfoils at every Reynolds and Mach number, checked for what the solver can run."""
        folder = self.batch_folder_entry.text().strip()
        if not os.path.isdir(folder):
            raise ValueError(f"Airfoil folder not found: {folder}")
        files = sorted(item.text() for item in self.batch_airfoil_list.selectedItems())
        if not files:
            raise ValueError("Select airfoils in 1. Parameterization (Ctrl+A selects all)")
        su2 = self.batch_solver_combo.currentText() == "SU2"
        if su2:
            fields = (self.batch_su2_re_entry, self.batch_su2_mach_entry, self.batch_su2_amin_entry,
                      self.batch_su2_amax_entry, self.batch_su2_astep_entry)
        else:
            fields = (self.batch_xf_re_entry, self.batch_xf_mach_entry, self.batch_xf_amin_entry,
                      self.batch_xf_amax_entry, self.batch_xf_astep_entry)
        res = self._parse_numbers(fields[0].text(), "Reynolds numbers")
        machs = self._parse_numbers(fields[1].text(), "Mach numbers")
        a_min = self._parse_one(fields[2].text(), "Min AoA")
        a_max = self._parse_one(fields[3].text(), "Max AoA")
        a_step = self._parse_one(fields[4].text(), "AoA step")
        repanel = self.batch_repanel_check.isChecked()
        n_points = int(self._parse_one(self.batch_npoints_entry.text(), "Desired points")) if repanel else 200
        y_plus = self._parse_one(self.batch_yplus_entry.text(), "Target y+")
        mesh = "hybrid" if self.batch_mesh_combo.currentText().startswith("Hybrid") else "structured"
        if a_max < a_min:
            raise ValueError("Max AoA is below Min AoA")
        if a_step <= 0 and a_max > a_min:
            raise ValueError("AoA step must be positive")
        if any(r <= 0 for r in res):
            raise ValueError("Reynolds numbers must be positive")
        if repanel and n_points < 20:
            raise ValueError("Desired points: use at least 20")
        if not su2:
            if any(m >= 1.0 for m in machs):
                raise ValueError("XFOIL runs only below Mach 1 (use SU2)")
            if repanel and n_points > XFOIL_MAX_POINTS_PER_SURFACE:
                raise ValueError(f"{n_points} points per surface gives {2 * n_points - 1} points; XFOIL loads at "
                                 f"most 1000 (use {XFOIL_MAX_POINTS_PER_SURFACE} or fewer)")
        elif self.flow_regime_combo.currentText() == "Incompressible" and any(m >= 0.7 for m in machs):
            raise ValueError("Solver Setup & Run is set to Incompressible, which cannot run Mach 0.7 or above "
                             "(switch Flow Type to Compressible)")
        solver = "su2" if su2 else "xfoil"
        return [Case(f, solver, r, m, a_min, a_max, a_step, n_points, y_plus, mesh, repanel)
                for f in files for r in res for m in machs]

    def _batch_describe(self, cases):
        n_alpha = sum(len(c.alphas()) for c in cases)
        c0 = cases[0]
        geom = f"re-paneled to {c0.n_points} per surface" if c0.repanel else "file coordinates (no re-paneling)"
        text = (f"{len(cases)} cases ({len({c.airfoil for c in cases})} airfoils), {n_alpha} angles, "
                f"{c0.solver.upper()}, {geom}")
        if c0.solver == "su2":
            text += f", {c0.mesh} mesh, target y+ {c0.y_plus:g}"
        return text

    def batch_preview(self):
        try:
            cases = self.batch_cases()
        except ValueError as e:
            QMessageBox.warning(self, "Batch", str(e))
            return
        self._batch_update_enabled()
        self.batch_cases_label.setText(self._batch_describe(cases))
        print(f"[Batch] {self._batch_describe(cases)}")

    def batch_start(self):
        if self._batch_thread is not None and self._batch_thread.is_alive():
            return
        try:
            cases = self.batch_cases()
            out_dir = self.batch_out_entry.text().strip()
            if not out_dir:
                raise ValueError("Choose an output folder")
            workers = int(self._parse_one(self.batch_workers_entry.text(), "Parallel cases"))
            xfoil_timeout = self._parse_one(self.batch_xfoil_timeout_entry.text(), "XFOIL time limit")
            su2_hours = self._parse_one(self.batch_su2_timeout_entry.text(), "SU2 time limit")
            cores = int(self._parse_one(self.batch_cores_entry.text(), "Cores")) if self.batch_mpi_check.isChecked() else 1
            su2_settings = None
            if cases[0].solver == "su2":
                su2_settings = {"regime": self.flow_regime_combo.currentText(),
                                "settings": self.collect_su2_settings()}
            xfoil_path = self.batch_xfoil_entry.text().strip() or None
            if cases[0].solver == "xfoil":
                if not (shutil.which(xfoil_path or "xfoil") or (xfoil_path and os.path.isfile(xfoil_path))):
                    raise ValueError(f"XFOIL executable not found: {xfoil_path}")
        except ValueError as e:
            QMessageBox.warning(self, "Batch", str(e))
            return

        folder = self.batch_folder_entry.text().strip()
        os.makedirs(out_dir, exist_ok=True)
        # The case list and the SU2 settings go next to the results, so the batch can be resumed or re-run later.
        import csv
        import json
        with open(os.path.join(out_dir, "cases.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(BATCH_CSV_COLUMNS)
            for c in cases:
                w.writerow([c.airfoil, c.solver, f"{c.reynolds:g}", f"{c.mach:g}", f"{c.alpha_min:g}",
                            f"{c.alpha_max:g}", f"{c.alpha_step:g}", c.n_points, f"{c.y_plus:g}", c.mesh,
                            "yes" if c.repanel else "no"])
        with open(os.path.join(out_dir, "batch_settings.json"), "w") as f:
            json.dump({"airfoil_folder": folder, "su2_settings": su2_settings, "cores": cores}, f, indent=1)

        self._batch_out_dir = out_dir
        self._batch_total = len(cases)
        self._batch_counts = {}
        self._batch_stop_event.clear()
        self.batch_cases_label.setText(self._batch_describe(cases))
        self.batch_start_button.setEnabled(False)
        self.batch_stop_button.setEnabled(True)
        self._batch_refresh_label(running=True)
        print(f"[Batch] Starting: {self._batch_describe(cases)} -> {out_dir}")

        kwargs = dict(out_dir=out_dir, airfoil_dir=folder, xfoil_path=xfoil_path, cores=cores, workers=workers,
                      xfoil_timeout=float(xfoil_timeout), su2_timeout=float(su2_hours) * 3600.0,
                      rerun_failed=self.batch_rerun_check.isChecked(),
                      dry_run=self.batch_dry_check.isChecked() and cases[0].solver == "su2",
                      su2_settings=su2_settings)

        def log(message):
            print(f"[Batch] {message}")
            self.batch_progress_signal.emit(message)

        def worker():
            try:
                stats = run_batch(cases, log=log, should_stop=self._batch_stop_event.is_set, **kwargs)
                self.batch_finished_signal.emit(stats, "")
            except Exception as e:
                self.batch_finished_signal.emit(None, f"{type(e).__name__}: {e}")

        self._batch_thread = threading.Thread(target=worker, daemon=True)
        self._batch_thread.start()

    def batch_stop(self):
        self._batch_stop_event.set()
        self.batch_stop_button.setEnabled(False)
        print("[Batch] Stopping: running cases are killed and marked 'stopped'; they run again next time.")

    def on_batch_progress(self, message):
        m = re.match(r"\[(\w+)\]", message)
        if m and m.group(1) != "start":
            self._batch_counts[m.group(1)] = self._batch_counts.get(m.group(1), 0) + 1
        elif m:
            self._batch_counts["running"] = self._batch_counts.get("running", 0) + 1
        self._batch_refresh_label(running=True)

    def _batch_refresh_label(self, running):
        done = sum(v for k, v in self._batch_counts.items() if k != "running")
        parts = [f"{k} {v}" for k, v in sorted(self._batch_counts.items()) if k != "running" and v]
        active = self._batch_counts.get("running", 0) - sum(
            v for k, v in self._batch_counts.items() if k not in ("running", "skip"))
        text = f"Finished {done} / {self._batch_total}"
        if parts:
            text += "  (" + ", ".join(parts) + ")"
        if running:
            text += f"  |  running now: {max(0, active)}"
        self.batch_progress_label.setText(text)

    def on_batch_finished(self, stats, error_text):
        """The batch summary goes to the console (log panel); no plots or dialogs."""
        self.batch_start_button.setEnabled(True)
        self.batch_stop_button.setEnabled(False)
        self._batch_refresh_label(running=False)
        out_dir = getattr(self, "_batch_out_dir", "")
        if error_text:
            print(f"[Batch] Failed: {error_text}")
            return
        head = "STOPPED" if self._batch_stop_event.is_set() else "FINISHED"
        print(f"[Batch] ===================== Batch {head} =====================")
        for solver in ("xfoil", "su2"):
            st = (stats or {}).get(solver)
            if not st:
                continue
            rate = st.get("angle_convergence_rate")
            states = ", ".join(f"{k} {v}" for k, v in sorted(st.get("cases_by_state", {}).items()))
            print(f"[Batch] {solver.upper()}: {st['cases_run']} of {st['cases']} cases run ({states})")
            print(f"[Batch]   angles converged: {st['angles_converged']} of {st['angles_requested']}"
                  + (f" ({100 * rate:.0f}%)" if rate is not None else ""))
            print(f"[Batch]   wall time {st['wall_hours']:.2f} h, {st['core_hours']:.2f} core-hours")
        try:
            import pandas as pd
            cs = pd.read_csv(os.path.join(out_dir, "cases_summary.csv"))
            bad = cs[~cs["state"].isin(["converged", "partial", "dry_run"])]
            if len(bad):
                print(f"[Batch] Not converged at any angle ({len(bad)}):")
                for r in bad.head(20).itertuples():
                    print(f"[Batch]   {r.airfoil}  Re {r.re:g}  M {r.mach:g}: {r.state}"
                          + (f" - {r.error}" if isinstance(r.error, str) and r.error else ""))
                if len(bad) > 20:
                    print(f"[Batch]   ... and {len(bad) - 20} more (cases_summary.csv)")
        except (OSError, KeyError, ValueError):
            pass
        print(f"[Batch] Results in {out_dir}: results.csv (one row per angle), cases_summary.csv, "
              f"batch_summary.json")
        excel = (stats or {}).get("excel")
        if excel:
            print(f"[Batch] Spreadsheet (one sheet per Reynolds number): {excel}")

    def batch_open_folder(self):
        path = self.batch_out_entry.text().strip()
        if not os.path.isdir(path):
            QMessageBox.information(self, "Batch", f"Folder does not exist yet:\n{path}")
            return
        if os.name == "nt":
            os.startfile(path)
        else:
            import subprocess
            subprocess.Popen(["xdg-open", path])

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