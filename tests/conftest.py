"""Shared pytest hooks."""

from __future__ import annotations

import faulthandler
import gc
import sys
from collections.abc import Generator
from typing import Any

import pytest


def _session_gui_collector() -> Any:
    from PySide6.QtWidgets import QApplication

    from nlab.views import gui_gc

    collector = getattr(QApplication.instance(), gui_gc._APPLICATION_ATTRIBUTE, None)
    if isinstance(collector, gui_gc.GuiGarbageCollector) and not collector.closed:
        return collector
    return None


@pytest.fixture(autouse=True)
def _collect_garbage_between_tests() -> Generator[None, None, None]:
    """Run GUI-owned garbage collection between tests, never inside them.

    The first test that builds MainAppWindow or WaveformAnalysisDialog installs
    nlab.views.gui_gc, which disables automatic collection for the rest of
    the session and collects from a 1 s GUI timer. Earlier tests' Qt widgets
    and plots then sit in reference cycles until one collection destroys many
    tests' worth at once: Docker and GitHub traces showed 5-30 s GUI-thread
    collections inside a later test's 5 s pytest-qt wait, which then timed
    out with its worker already finished. Holding the timer during each test
    and running the collector's own threshold check at teardown keeps the
    total collection work unchanged and moves every pause between tests.
    test_gui_gc installs fresh collectors of its own and is unaffected.
    """
    collector = _session_gui_collector()
    if collector is not None:
        collector.timer.stop()
    yield
    collector = _session_gui_collector()
    if collector is not None:
        collector.timer.stop()  # The test may have installed it.
        collector.check()


@pytest.fixture(autouse=True)
def isolated_waveform_drawing(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Keep the waveform-drawing preference in memory for each test.

    Every Scope, MCA and Waveform Analysis view reads it on construction, so
    a "lines" choice saved while using the application would otherwise change
    what the tests draw, and a test accepting General Settings would write it.
    """
    from types import SimpleNamespace

    from nlab.views import plot_viewbox

    store: dict[str, Any] = {}
    settings = SimpleNamespace(
        value=lambda key, default=None: store.get(key, default),
        setValue=store.__setitem__,
    )
    monkeypatch.setattr(plot_viewbox, "QSettings", lambda: settings)
    return store


@pytest.fixture(autouse=True)
def timestamp_chosen_names(monkeypatch: pytest.MonkeyPatch) -> dict[str, bool]:
    """Keep "add the start time to chosen file names" off unless a test turns it on.

    The option lives in the user's QSettings; with it checked while using the
    application, every test that records under a chosen name would see a
    timestamped name. Set ``["enabled"] = True`` to exercise it.
    """
    from nlab.utils import recording_paths

    state = {"enabled": False}
    monkeypatch.setattr(recording_paths, "timestamp_chosen_names", lambda: state["enabled"])
    return state


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None],
) -> Generator[None, Any, None]:
    """Dump every thread's stack when a pytest-qt wait times out.

    Such a timeout usually means a background thread never finished; the
    dump shows where it is stuck, which a CI log otherwise cannot.
    """
    outcome = yield
    report = outcome.get_result()
    if (
        report.when == "call"
        and report.failed
        and call.excinfo is not None
        and call.excinfo.typename == "TimeoutError"
        and type(call.excinfo.value).__module__.startswith("pytestqt")
    ):
        sys.stderr.write(f"\n--- thread stacks after the wait timed out in {item.nodeid} ---\n")
        sys.stderr.flush()
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        _report_qt_threads()


def _report_qt_threads() -> None:
    """List widget-owned QThreads: an idle QThread has no Python stack to dump."""
    from PySide6.QtCore import QThread
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance()
    if not isinstance(application, QApplication):
        return
    out = sys.stderr
    out.write(f"--- Qt threads (python gc enabled={gc.isenabled()}) ---\n")
    for widget in application.allWidgets():
        for thread in widget.findChildren(QThread):
            try:
                state = f"running={thread.isRunning()} finished={thread.isFinished()}"
            except RuntimeError as exc:
                state = f"deleted ({exc})"
            out.write(f"{type(widget).__name__}: QThread {state}\n")
        if hasattr(widget, "_index_thread"):
            out.write(
                f"{type(widget).__name__}: _index_thread={widget._index_thread!r} "
                f"_index_worker={widget._index_worker!r} "
                f"loaded={getattr(widget, '_index', None) is not None}\n"
            )
    out.flush()
