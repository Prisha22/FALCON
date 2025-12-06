#live_plotter.py
import tkinter as tk
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import time
import os
from queue import Queue
import threading
import warnings




class LivePlotterWindow:
    """A window that contains a Matplotlib plot and updates it with data from a queue from the data in history.csv."""

    def __init__(self, parent, data_queue: Queue, aoa: float, is_unsteady: bool = False):
        self.data_queue = data_queue
        self.aoa = aoa
        self.is_unsteady = is_unsteady
        
        if threading.current_thread() is not threading.main_thread():
            print("[LivePlotter] Warning: Not in main thread, plot window may not display correctly")
            print(f"[LivePlotter] Current thread: {threading.current_thread().name}")
            print(f"[LivePlotter] Main thread: {threading.main_thread().name}")
        
        self.window = tk.Toplevel(parent)
        self.window.title(f'SU2 Residuals | AoA: {aoa:.2f}')
        self.window.geometry("800x600")

        self.fig = Figure(figsize=(10, 6))
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.update_plot()

    def update_plot(self):
        """Checks the queue for new data and redraws the plot."""
        try:
            while not self.data_queue.empty():
                df = self.data_queue.get_nowait()
                self.ax.clear()

                df.columns = df.columns.str.strip().str.replace('"', '')

                if self.is_unsteady:
                    x_axis_col = next((col for col in df.columns if col.upper() == 'TIME_ITER'), None)
                    
                    if not x_axis_col:
                        x_axis_col = next((col for col in df.columns if 'ITER' in col.upper()), None)
                else:
                    x_axis_col = next((col for col in df.columns if col.upper() == 'INNER_ITER'), None)
                    
                    if not x_axis_col:
                        x_axis_col = next((col for col in df.columns if 'ITER' in col.upper()), None)
                
                if not x_axis_col:
                    print("Error: Could not find iteration column for plotting.")
                    self.schedule_next_update()
                    return

                if self.is_unsteady:
                    time_iter_col = next((col for col in df.columns if col.upper() == 'TIME_ITER'), None)
                    inner_iter_col = next((col for col in df.columns if col.upper() == 'INNER_ITER'), None)
                    if time_iter_col and inner_iter_col:
                        df[time_iter_col] = pd.to_numeric(df[time_iter_col], errors='coerce')
                        plot_df = df.groupby(time_iter_col).last().reset_index()
                    else:
                        plot_df = df
                else:
                    plot_df = df

                residual_cols = [col for col in plot_df.columns if
                                 col.upper().startswith('RMS') or 
                                 col.upper().startswith('RES') or
                                 col.upper().startswith('REL_RMS')]

                if residual_cols and not plot_df.empty:
                    for col in residual_cols:
                        self.ax.plot(plot_df[x_axis_col], plot_df[col], marker='o', markersize=3, linestyle='-')

                    last_iter = plot_df[x_axis_col].iloc[-1]
                    mode_str = "Unsteady" if self.is_unsteady else "Steady"
                    self.ax.set_title(
                        f"Live Convergence [{mode_str}] | AoA = {self.aoa:.2f}° | Last Iteration: {last_iter:.0f}")
                else:
                    self.ax.set_title(f"Live Convergence | AoA = {self.aoa:.2f}° | Waiting for residual data...")

                x_label = "Time Iteration" if self.is_unsteady else "Inner Iteration"
                self.ax.set_xlabel(x_label)
                self.ax.set_ylabel("Log10(Residual)")
                self.ax.legend(residual_cols, loc='upper right')
                self.ax.grid(True, which="both", ls="--")
                self.fig.tight_layout()
                self.canvas.draw()

            self.schedule_next_update()

        except Exception as e:
            print(f"[Plotter Window] Error updating plot: {e}")
            import traceback
            traceback.print_exc()
            self.schedule_next_update()

    def schedule_next_update(self):
        """Schedule the next update check."""
        try:
            if self.window.winfo_exists():
                self.window.after(1000, self.update_plot)
        except tk.TclError:
            pass

    def close(self):
        """Close the plotter window."""
        try:
            if self.window and self.window.winfo_exists():
                self.window.destroy()
        except (AttributeError, tk.TclError):
            pass
def safe_close(self):
        """Thread-safe close method"""
        try:
            if self.window and self.window.winfo_exists():
                self.window.destroy()
        except (AttributeError, tk.TclError):
            pass

def data_reader_thread(history_filepath: str, data_queue: Queue, stop_event, polling_interval: float = 0.5):
    """Watches the history file, reads it, and puts the DataFrame into the queue."""
    last_known_rows = 0
    print(f"[Data Reader] Watching file: {history_filepath}")
    print(f"[Data Reader] Running in thread: {threading.current_thread().name}")
    
    while not stop_event.is_set():
        if os.path.exists(history_filepath) and os.path.getsize(history_filepath) > 0:
            try:
                df = pd.read_csv(history_filepath)
                
                if len(df) > last_known_rows:
                    last_known_rows = len(df)
                    df.columns = df.columns.str.strip().str.replace('"', '')
                    data_queue.put(df)
            except pd.errors.EmptyDataError:
                pass
            except Exception as e:
                pass
        time.sleep(polling_interval)
    print("[Data Reader Thread] Stop event received. Exiting.")