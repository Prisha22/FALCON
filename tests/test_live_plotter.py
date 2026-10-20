"""The thread that tails SU2's history file for the live convergence plot (no window is opened)."""
import queue
import threading
import time

import pytest

pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)  # live_plotter imports Qt at module level; skip if Qt libraries are missing
from Scripts.Solver_and_Results.live_plotter import data_reader_thread, history_candidates  # noqa: E402

CSV_1 = '"Inner_Iter","rms[P]","CL"\n0,-1.0,0.1\n'
CSV_2 = CSV_1 + "1,-2.0,0.2\n"


def tail_file(path, timeout=5.0):
    """Run the reader on `path`; returns (queue, stop, thread)."""
    q, stop = queue.Queue(), threading.Event()
    t = threading.Thread(target=data_reader_thread, args=(path, q, stop, 0.05), daemon=True)
    t.start()
    return q, stop, t


def get(q, timeout=5.0):
    return q.get(timeout=timeout)


def test_candidates_cover_both_spellings():
    assert [c.replace("\\", "/") for c in history_candidates("/run/AoA_2.00/history_aoa_2.00.csv")] == [
        "/run/AoA_2.00/history_aoa_2.00.csv", "/run/AoA_2.00/history_aoa_2.csv"]   # folder dots are ignored


@pytest.mark.parametrize("written", ["history_aoa_2.00.csv", "history_aoa_2.csv"])
def test_reader_follows_the_file_whichever_spelling_su2_wrote(tmp_path, written):
    (tmp_path / written).write_text(CSV_1)
    q, stop, t = tail_file(str(tmp_path / "history_aoa_2.00.csv"))
    try:
        df = get(q)
        assert list(df.columns) == ["Inner_Iter", "rms[P]", "CL"]        # quotes stripped
        assert len(df) == 1
        (tmp_path / written).write_text(CSV_2)                            # SU2 appends a row
        assert len(get(q)) == 2
    finally:
        stop.set()
        t.join(timeout=2)
    assert not t.is_alive()


def test_reader_waits_for_the_file_and_stops_on_request(tmp_path):
    q, stop, t = tail_file(str(tmp_path / "history_aoa_2.00.csv"))
    time.sleep(0.3)
    assert q.empty() and t.is_alive()                                     # nothing yet, no crash
    stop.set()
    t.join(timeout=2)
    assert not t.is_alive()