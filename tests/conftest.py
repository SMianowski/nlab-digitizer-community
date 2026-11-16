"""Shared pytest hooks."""

from __future__ import annotations

import faulthandler
import sys
from collections.abc import Generator
from typing import Any

import pytest


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
