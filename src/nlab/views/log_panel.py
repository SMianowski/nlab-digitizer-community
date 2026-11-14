from __future__ import annotations

import html
import logging
import threading
from collections import deque

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import QPlainTextEdit, QWidget

# Lines arriving within this window are appended in one batch.
_FLUSH_MS = 50
# Matches the console's block limit: older lines would be trimmed anyway.
_MAX_PENDING = 5_000


class _LogEmitter(QObject):
    ready = Signal()


class LogHandler(logging.Handler):
    """Thread-safe logging handler that queues formatted records for the console.

    Formatting happens on the emitting thread; the GUI thread only receives
    rendered strings. Instead of one queued Qt signal per record, records are
    buffered and ``ready`` fires once when the buffer goes from empty to
    non-empty, so a log burst costs one GUI wake-up. If the GUI falls behind,
    the oldest pending lines are dropped and counted.
    """

    def __init__(self, max_pending: int = _MAX_PENDING) -> None:
        super().__init__(level=logging.DEBUG)
        self._emitter = _LogEmitter()
        self.ready = self._emitter.ready
        self._pending: deque[tuple[str, int]] = deque(maxlen=max_pending)
        self._pending_lock = threading.Lock()
        self._dropped = 0
        self.setFormatter(
            logging.Formatter(
                "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = (self.format(record), record.levelno)
            with self._pending_lock:
                notify = not self._pending
                if len(self._pending) == self._pending.maxlen:
                    self._dropped += 1
                self._pending.append(line)
            if notify:
                self._emitter.ready.emit()
        except Exception:
            self.handleError(record)

    def drain(self) -> tuple[list[tuple[str, int]], int]:
        """Take every pending (text, levelno) line and the dropped-line count."""
        with self._pending_lock:
            lines = list(self._pending)
            self._pending.clear()
            dropped, self._dropped = self._dropped, 0
        return lines, dropped


class LogConsole(QPlainTextEdit):
    """Read-only dark-themed console that displays Python log records.

    Installs its own LogHandler into the root logger on construction and
    removes it when destroyed. The handler level is always DEBUG; the root
    logger's level acts as the gate.
    """

    _COLORS: dict[int, str] = {
        logging.DEBUG:    "#7ec8e3",
        logging.INFO:     "#d4d4d4",
        logging.WARNING:  "#f0c040",
        logging.ERROR:    "#f04040",
        logging.CRITICAL: "#ff6060",
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        # A log view has nothing to undo; do not keep every append on a stack.
        self.setUndoRedoEnabled(False)
        self.setMaximumBlockCount(5_000)
        font = QFont("Consolas", 9)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(font)
        self.setStyleSheet("QPlainTextEdit { background:#1e1e1e; color:#d4d4d4; border:none; }")

        self._flush_timer = QTimer(self)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.setInterval(_FLUSH_MS)
        self._flush_timer.timeout.connect(self.flush)

        handler = LogHandler()
        self._handler = handler
        handler.ready.connect(self._flush_timer.start)
        logging.getLogger().addHandler(handler)
        self.destroyed.connect(lambda *_: logging.getLogger().removeHandler(handler))

    @property
    def handler(self) -> LogHandler:
        return self._handler

    def flush(self) -> None:
        """Append every pending line in one edit block and scroll once."""
        lines, dropped = self._handler.drain()
        if dropped:
            lines.insert(0, (f"… {dropped:,} earlier log line(s) dropped", logging.WARNING))
        if not lines:
            return
        cursor = QTextCursor(self.document())
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.beginEditBlock()
        new_block = not self.document().isEmpty()
        for text, levelno in lines:
            if new_block:
                cursor.insertBlock()
            new_block = True
            color = self._COLORS.get(levelno, "#d4d4d4")
            cursor.insertHtml(f'<span style="color:{color};">{html.escape(text)}</span>')
        cursor.endEditBlock()
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())
