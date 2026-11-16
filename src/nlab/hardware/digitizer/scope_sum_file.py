"""Raw binary container for periodic-sum recordings (NSUM version 1).

The HDF5 recording stores every packet twice (raw bytes plus decoded columns)
and spends most of its time in Python. In the reader process that competes
with the DMA reader for the interpreter lock: live on .128 (IP124, 192-sample
windows, gap 0) HDF5 recording alone cut coverage from 99.7% to 90-95%. This
container appends the packets exactly as received with one ``write()`` per
batch, which releases the lock, and writes a third of the bytes.

Layout, little-endian:

``preamble``
    ``b"NSUM"``, u16 version (1), u16 reserved (0), u32 header JSON length.
``header``
    UTF-8 JSON (format, units, creation time, capture metadata, configuration
    YAML), zero padded so the first packet starts at a multiple of 4096 bytes.
``packets``
    The 4096-byte IP123/IP124 periodic-sum packets, unchanged.
``trailer``
    One 4096-byte block written when the recording closes: a 16-byte marker,
    u32 JSON length, UTF-8 JSON (completion, cleanup error, final hardware
    missed windows, runtime metadata, packet count), zero padding. Bytes 8-16
    of the marker can never equal a packet's ABI magic, so the trailer cannot
    be mistaken for a packet.

A file without a trailer did not close (crash, power loss): the packets up to
its last whole 4096-byte block stay readable and the file reports incomplete.
A partial final block is ignored. Never append sums to an NDMA file: the NDMA
formats describe waveforms and list-mode records, not structured sums.
"""

from __future__ import annotations

import json
import os
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

PACKET_BYTES = 4096
MAGIC = b"NSUM"
VERSION = 1
SUFFIX = ".bin"
_PREAMBLE = struct.Struct("<4sHHI")
_TRAILER_MARKER = b"NSUM-END" + b"TRAILER\0"
_TRAILER = struct.Struct("<16sI")
_TRAILER_JSON_BYTES = PACKET_BYTES - _TRAILER.size
_WRITE_BUFFER_BYTES = 1 << 20


def is_binary_sum_file(path: Path) -> bool:
    """True when *path* starts with the NSUM magic (any version)."""
    try:
        with Path(path).open("rb") as handle:
            return handle.read(len(MAGIC)) == MAGIC
    except OSError:
        return False


def _json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")


def _sync(handle: BinaryIO) -> None:
    handle.flush()
    os.fsync(handle.fileno())


class SumBinaryWriter:
    """Append raw packets; write the trailer only after the data is on disk."""

    def __init__(self, path: Path, header: dict[str, Any], *, overwrite: bool = False) -> None:
        text = _json_bytes(header)
        data_offset = -(-(_PREAMBLE.size + len(text)) // PACKET_BYTES) * PACKET_BYTES
        self._file: BinaryIO = Path(path).open(
            "wb" if overwrite else "xb", buffering=_WRITE_BUFFER_BYTES,
        )
        self.packets = 0
        self._closed = False
        try:
            self._file.write(
                _PREAMBLE.pack(MAGIC, VERSION, 0, len(text))
                + text
                + bytes(data_offset - _PREAMBLE.size - len(text))
            )
        except BaseException:
            self._file.close()
            raise

    def append_raw(self, packets: Sequence[bytes]) -> None:
        if self._closed:
            raise RuntimeError("Sum binary writer is closed")
        data = b"".join(packets)
        if len(data) != PACKET_BYTES * len(packets):
            raise ValueError("Every periodic-sum packet must contain exactly 4096 bytes")
        self._file.write(data)
        self.packets += len(packets)

    def close(self, trailer: dict[str, Any]) -> None:
        """Flush all packets to disk, then write and flush the trailer."""
        if self._closed:
            return
        self._closed = True
        try:
            _sync(self._file)  # Data first: a trailer must never precede its packets.
            body = dict(trailer, packets=self.packets)
            text = _json_bytes(body)
            if len(text) > _TRAILER_JSON_BYTES:
                # Only free text can grow; keep the counts and flags intact.
                error = str(body.get("cleanup_error", ""))
                excess = len(text) - _TRAILER_JSON_BYTES
                body["cleanup_error"] = error[: max(0, len(error) - excess - 16)] + " [truncated]"
                text = _json_bytes(body)
                if len(text) > _TRAILER_JSON_BYTES:
                    body["runtime_metadata"] = None
                    text = _json_bytes(body)
            block = _TRAILER.pack(_TRAILER_MARKER, len(text)) + text
            self._file.write(block + bytes(PACKET_BYTES - len(block)))
            _sync(self._file)
        finally:
            self._file.close()


@dataclass(frozen=True)
class SumBinaryLayout:
    """What an NSUM file contains, read without touching its packets."""

    header: dict[str, Any]
    data_offset: int
    packets: int
    trailer: dict[str, Any] | None
    ignored_bytes: int  # a partial final block left by an interrupted write


def read_layout(path: Path) -> SumBinaryLayout:
    with Path(path).open("rb") as handle:
        preamble = handle.read(_PREAMBLE.size)
        if len(preamble) != _PREAMBLE.size:
            raise ValueError("Not an NSUM periodic-sum file: truncated preamble")
        magic, version, _reserved, length = _PREAMBLE.unpack(preamble)
        if magic != MAGIC:
            raise ValueError("Not an NSUM periodic-sum file")
        if version != VERSION:
            raise ValueError(f"Unsupported NSUM version {version}; this build reads {VERSION}")
        text = handle.read(length)
        if len(text) != length:
            raise ValueError("Truncated NSUM header")
        header = json.loads(text.decode("utf-8"))
        if not isinstance(header, dict):
            raise ValueError("NSUM header must contain a mapping")
        data_offset = -(-(_PREAMBLE.size + length) // PACKET_BYTES) * PACKET_BYTES
        size = os.fstat(handle.fileno()).st_size
        body = max(0, size - data_offset)
        blocks, ignored = divmod(body, PACKET_BYTES)
        trailer = None
        if blocks and not ignored:
            handle.seek(data_offset + (blocks - 1) * PACKET_BYTES)
            last = handle.read(PACKET_BYTES)
            marker, trailer_length = _TRAILER.unpack_from(last)
            if marker == _TRAILER_MARKER:
                if trailer_length > _TRAILER_JSON_BYTES:
                    raise ValueError("Invalid NSUM trailer length")
                value = json.loads(last[_TRAILER.size:_TRAILER.size + trailer_length])
                if not isinstance(value, dict):
                    raise ValueError("NSUM trailer must contain a mapping")
                trailer = value
                blocks -= 1
                if trailer.get("packets") != blocks:
                    raise ValueError("NSUM trailer packet count differs from the file")
    return SumBinaryLayout(header, data_offset, blocks, trailer, ignored)


def read_packet_bytes(
    handle: BinaryIO, layout: SumBinaryLayout, start: int, stop: int,
) -> bytes:
    """Raw bytes of packets ``[start, stop)``; the caller decodes and validates."""
    if not 0 <= start <= stop <= layout.packets:
        raise IndexError("NSUM packet range outside the file")
    handle.seek(layout.data_offset + start * PACKET_BYTES)
    data = handle.read((stop - start) * PACKET_BYTES)
    if len(data) != (stop - start) * PACKET_BYTES:
        raise ValueError("NSUM file shrank while reading")
    return data
