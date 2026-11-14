from __future__ import annotations

import logging
import threading

from pytestqt.qtbot import QtBot

from nlab.views.log_panel import LogConsole, LogHandler

log = logging.getLogger("nlab.test.log_panel")


def test_burst_from_a_thread_is_appended_in_one_flush(qtbot: QtBot) -> None:
    console = LogConsole()
    qtbot.addWidget(console)
    previous = logging.getLogger().level
    logging.getLogger().setLevel(logging.INFO)
    ready: list[None] = []
    console.handler.ready.connect(lambda: ready.append(None))
    try:
        thread = threading.Thread(target=lambda: [log.info("line %d", i) for i in range(200)])
        thread.start()
        thread.join()
        assert console.toPlainText() == ""  # nothing is drawn per record
        qtbot.waitUntil(lambda: "line 199" in console.toPlainText())
        lines = console.toPlainText().splitlines()
        assert [line.rsplit(" ", 1)[-1] for line in lines] == [str(i) for i in range(200)]
        assert console.document().blockCount() == 200
        assert len(ready) == 1  # one GUI wake-up for the whole burst
        assert not console.isUndoRedoEnabled()
    finally:
        logging.getLogger().setLevel(previous)
        logging.getLogger().removeHandler(console.handler)


def test_overflow_drops_the_oldest_lines_and_says_so() -> None:
    handler = LogHandler(max_pending=2)
    for i in range(5):
        handler.emit(logging.makeLogRecord({"msg": f"m{i}", "levelno": logging.INFO}))
    lines, dropped = handler.drain()
    assert [text.rsplit(" ", 1)[-1] for text, _ in lines] == ["m3", "m4"]
    assert dropped == 3
    assert handler.drain() == ([], 0)


def test_console_reports_dropped_lines(qtbot: QtBot) -> None:
    console = LogConsole()
    qtbot.addWidget(console)
    logging.getLogger().removeHandler(console.handler)
    console.handler._dropped = 7
    console.handler.emit(logging.makeLogRecord({"msg": "after", "levelno": logging.ERROR}))
    console.flush()
    text = console.toPlainText()
    assert text.splitlines()[0] == "… 7 earlier log line(s) dropped"
    assert text.splitlines()[1].endswith("after")
