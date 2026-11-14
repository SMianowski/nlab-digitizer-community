"""Qt-free transport-error classification shared by workers and the list-mode process."""

from __future__ import annotations

import errno

_FATAL_TRANSPORT_ERRNOS = frozenset(
    {
        errno.ECONNABORTED,
        errno.ECONNRESET,
        errno.ENETDOWN,
        errno.ENETRESET,
        errno.ENETUNREACH,
        errno.ENOTCONN,
        errno.EPIPE,
        errno.EHOSTUNREACH,
        # Python normally maps these Winsock values to ConnectionError
        # subclasses. Keep the numeric forms for libiio/OSError wrappers.
        10050,  # WSAENETDOWN
        10051,  # WSAENETUNREACH
        10052,  # WSAENETRESET
        10053,  # WSAECONNABORTED
        10054,  # WSAECONNRESET
        10058,  # WSAESHUTDOWN
        10065,  # WSAEHOSTUNREACH
    }
)


def is_fatal_transport_error(exc: BaseException) -> bool:
    """Return whether an exception means the current connection is unusable.

    Do not classify timeouts or generic I/O errors as terminal here. MCA DMA
    legitimately sees remote timeouts while waiting for an external trigger,
    and the backend has its own recovery policy for EIO/EINVAL hardware faults.
    """
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ConnectionError):
            return True
        if isinstance(current, OSError) and current.errno in _FATAL_TRANSPORT_ERRNOS:
            return True
        current = current.__cause__ or current.__context__
    return False
