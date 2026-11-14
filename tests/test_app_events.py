from __future__ import annotations

from collections.abc import Iterator

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QObject, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QWidget
from pytestqt.qtbot import QtBot

from nlab.views.app_events import ApplicationEventHub, application_event_hub


class _Recorder(QObject):
    def __init__(self, consume: bool = False) -> None:
        super().__init__()
        self.seen: list[QEvent.Type] = []
        self.consume = consume

    def handle(self, _watched: QObject, event: QEvent) -> bool:
        self.seen.append(event.type())
        return self.consume


def _key() -> QKeyEvent:
    return QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_L, Qt.KeyboardModifier.NoModifier)


_HUBS: list[ApplicationEventHub] = []


def _hub() -> ApplicationEventHub:
    application = QApplication.instance()
    assert application is not None
    hub = ApplicationEventHub(application)
    _HUBS.append(hub)
    return hub


@pytest.fixture(autouse=True)
def _remove_test_hubs() -> Iterator[None]:
    yield
    application = QApplication.instance()
    while _HUBS:
        hub = _HUBS.pop()
        if application is not None:
            application.removeEventFilter(hub)
        hub.deleteLater()


def test_only_requested_event_types_reach_a_handler(qtbot: QtBot) -> None:
    hub, target, recorder = _hub(), QWidget(), _Recorder()
    qtbot.addWidget(target)
    hub.register(recorder, recorder.handle, events=(QEvent.Type.KeyPress,))
    QCoreApplication.sendEvent(target, QEvent(QEvent.Type.User))
    QCoreApplication.sendEvent(target, _key())
    assert recorder.seen == [QEvent.Type.KeyPress]
    hub.set_events(recorder, None)
    QCoreApplication.sendEvent(target, QEvent(QEvent.Type.User))
    assert recorder.seen[-1] == QEvent.Type.User
    hub.unregister(recorder)
    QCoreApplication.sendEvent(target, _key())
    assert len(recorder.seen) == 2


def test_first_handler_runs_first_and_consumes(qtbot: QtBot) -> None:
    hub, target = _hub(), QWidget()
    qtbot.addWidget(target)
    later, help_like = _Recorder(), _Recorder(consume=True)
    hub.register(later, later.handle, events=(QEvent.Type.KeyPress,))
    hub.register(help_like, help_like.handle, events=(QEvent.Type.KeyPress,), first=True)
    assert hub.eventFilter(target, _key())
    assert help_like.seen == [QEvent.Type.KeyPress]
    assert later.seen == []


def test_destroyed_owner_is_forgotten(qtbot: QtBot) -> None:
    hub, target = _hub(), QWidget()
    qtbot.addWidget(target)
    seen: list[QEvent.Type] = []
    owner = QObject()
    hub.register(owner, lambda _w, e: bool(seen.append(e.type())), events=(QEvent.Type.KeyPress,))
    owner.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not hub.eventFilter(target, _key())
    assert seen == []


def test_application_hub_is_shared() -> None:
    assert application_event_hub() is application_event_hub()
