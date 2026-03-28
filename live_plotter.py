import sys
import pandas as pd
import threading
import os
import time
from queue import Queue

# PyQt6 Imports
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import QTimer
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure


class LivePlotterWindow(QWidget):
    """
     plots SU2 residuals live from the csv.
    """

    def __init__(self, data_queue: Queue, aoa: float, is_unsteady: bool = False):
        super().__init__()
        self.data_queue = data_queue
        self.aoa = aoa
        self.is_unsteady = is_unsteady

        self.setWindowTitle(f'SU2 Residuals | AoA: {aoa:.2f}')
        self.resize(800, 600)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.fig = Figure(figsize=(10, 6))
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Setup Timer for updates (checks queue every 1000ms)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1000)

    def update_plot(self):
        """Read all available data from queue and plot the latest chunk."""
        if self.data_queue.empty():
            return

        try:
            df = None
            while not self.data_queue.empty():
                df = self.data_queue.get_nowait()

            if df is None: return

            self.ax.clear()
            df.columns = df.columns.str.strip().str.replace('"', '')

            if self.is_unsteady:
                x_col = next((c for c in df.columns if c.upper() == 'TIME_ITER'), None)
            else:
                x_col = next((c for c in df.columns if c.upper() == 'INNER_ITER'), None)

            if not x_col:
                x_col = next((c for c in df.columns if 'ITER' in c.upper()), None)

            if not x_col:
                return

            residual_cols = [c for c in df.columns if
                             c.upper().startswith('RMS') or
                             c.upper().startswith('RES') or
                             c.upper().startswith('REL_RMS')]

            if residual_cols and not df.empty:
                for col in residual_cols:
                    self.ax.plot(df[x_col], df[col], marker='o', markersize=3, linestyle='-')

                last_iter = df[x_col].iloc[-1]
                mode = "Unsteady" if self.is_unsteady else "Steady"
                self.ax.set_title(f"Live Convergence [{mode}] | AoA = {self.aoa:.2f}° | Iter: {last_iter:.0f}")
                self.ax.legend(residual_cols, loc='upper right')
            else:
                self.ax.set_title(f"Live Convergence | AoA = {self.aoa:.2f}° | Waiting for data...")

            self.ax.set_xlabel(x_col)
            self.ax.set_ylabel("Log10(Residual)")
            self.ax.grid(True, which="both", ls="--")
            self.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            print(f"[LivePlotter] Error: {e}")

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)


def data_reader_thread(history_filepath: str, data_queue: Queue, stop_event, polling_interval: float = 0.5):
    """Background thread that tails the CSV file."""
    last_known_rows = 0

    while not stop_event.is_set():
        if os.path.exists(history_filepath) and os.path.getsize(history_filepath) > 0:
            try:
                df = pd.read_csv(history_filepath)
                if len(df) > last_known_rows:
                    last_known_rows = len(df)
                    df.columns = df.columns.str.strip().str.replace('"', '')
                    data_queue.put(df)
            except:
                pass
        time.sleep(polling_interval)