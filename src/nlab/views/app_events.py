"""One application-wide event filter that dispatches to registered handlers.

Every application-wide ``installEventFilter`` costs a Python call for every Qt
event of every widget (paint, timer, mouse move, ...). The GUI used to install
one per feature (window shortcuts, context help, numeric input, and one per PSD
view or offline PSD plot). This hub is the only one installed. Each handler
declares the event types it needs, so all other events are left in C++ after
one set lookup.

Order: handlers registered with ``first=True`` (context help, which swallows
input while help mode is on) run before the others; the rest run in
registration order and the first handler that returns True consumes the event.
"""

from __future__ import annotations

import inspect
import weakref
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from PySide6.QtCore import QCoreApplication, QEvent, QObject

EventHandler = Callable[[QObject, QEvent], bool]


@dataclass
class _Route:
    owner: int
    handler: Callable[[], EventHandler | None]
    events: frozenset[QEvent.Type] | None


def _weak(handler: EventHandler) -> Callable[[], EventHandler | None]:
    if inspect.ismethod(handler):
        return weakref.WeakMethod(handler)
    return lambda: handler


class ApplicationEventHub(QObject):
    def __init__(self, application: QCoreApplication) -> None:
        super().__init__(application)
        self._routes: list[_Route] = []
        self._kinds: frozenset[QEvent.Type] = frozenset()
        self._every_event = False
        application.installEventFilter(self)

    def register(
        self,
        owner: QObject,
        handler: EventHandler,
        *,
        events: Iterable[QEvent.Type] | None,
        first: bool = False,
    ) -> None:
        """Route ``events`` (None: every event) to ``handler`` until ``owner`` goes.

        Registering an owner again replaces its route (keeping its position)."""
        route = _Route(id(owner), _weak(handler), None if events is None else frozenset(events))
        # A dead route's id() may be recycled by this owner: drop those first.
        self._routes = [existing for existing in self._routes if existing.handler() is not None]
        for index, existing in enumerate(self._routes):
            if existing.owner == route.owner:
                self._routes[index] = route
                break
        else:
            if first:
                self._routes.insert(0, route)
            else:
                self._routes.append(route)
            key = route.owner
            owner.destroyed.connect(lambda *_: self._remove(key))
        self._refresh()

    def set_events(self, owner: QObject, events: Iterable[QEvent.Type] | None) -> None:
        """Change which event types reach ``owner``'s handler."""
        for route in self._routes:
            if route.owner == id(owner):
                route.events = None if events is None else frozenset(events)
        self._refresh()

    def unregister(self, owner: QObject) -> None:
        self._remove(id(owner))

    def _remove(self, owner: int) -> None:
        self._routes = [route for route in self._routes if route.owner != owner]
        self._refresh()

    def _refresh(self) -> None:
        self._every_event = any(route.events is None for route in self._routes)
        self._kinds = frozenset().union(
            *(route.events for route in self._routes if route.events is not None)
        )

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        kind = event.type()
        if not self._every_event and kind not in self._kinds:
            return False
        handled = stale = False
        for route in tuple(self._routes):
            if route.events is not None and kind not in route.events:
                continue
            handler = route.handler()
            if handler is None:
                stale = True
            elif handler(watched, event):
                handled = True
                break
        if stale:
            self._routes = [route for route in self._routes if route.handler() is not None]
            self._refresh()
        return handled


def application_event_hub() -> ApplicationEventHub | None:
    """The hub for the running application (created on first use)."""
    application = QCoreApplication.instance()
    if application is None:
        return None
    hub = getattr(application, "_nlab_event_hub", None)
    if not isinstance(hub, ApplicationEventHub):
        hub = ApplicationEventHub(application)
        # Retain the Python object for as long as the application exists.
        setattr(application, "_nlab_event_hub", hub)
    return hub
