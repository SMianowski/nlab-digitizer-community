"""IP123/IP124 periodic-sum packets and lossless, versioned recordings.

Contract: PetaLinux bb8c57e, scope-periodic-sum-v123.md. Sums contain all N
original offset-binary ADC samples. They cannot recover waveforms or extrema.

Window sums measure the detector signal in current (integrating) mode rather
than by counting pulses (Knoll 2010).

References
----------
Knoll, G. F. (2010). Radiation Detection and Measurement, 4th ed.
    Wiley, Hoboken NJ. ISBN 978-0-470-13148-0
"""

from __future__ import annotations

import errno
import json
import logging
import struct
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING, Any, NotRequired, Protocol, TypedDict, cast

import h5py
import numpy as np
import numpy.typing as npt

from nlab.hardware.digitizer import scope_sum_file as sum_file

if TYPE_CHECKING:
    from nlab.hardware.digitizer.reader_process import ReaderSession

MASK64 = (1 << 64) - 1
MAGICS = {0x31534D5553504456: 123, 0x32534D5553504456: 124}
# Newest windows kept for the live plot (about 16 packets).
_DISPLAY_POINTS = 8192


class ScopeSumSnapshot(TypedDict):
    # Newest display windows, oldest first: shape (n, 2) float64 of
    # (elapsed seconds, mean offset ADC code). An array, not tuples, so the
    # reader process pickles it cheaply and the GUI plots it without copying.
    points: npt.NDArray[np.float64]
    packets: int
    windows: int
    samples: int
    signed_sum: int
    coverage_ns: int
    missed: int
    skipped: int
    mean: float | None
    complete: NotRequired[bool]
    final_missed_windows: NotRequired[int]
    ip_version: NotRequired[int]
    samples_per_window: NotRequired[int]
    gap_cycles: NotRequired[int]
    interval_ns: NotRequired[int]


class ScopeSumBackend(Protocol):
    def prepare_dma_capture(self) -> None: ...
    def get_scope_dma_geometry(self) -> object: ...
    def read_dma_raw_frame(self) -> bytes: ...
    def request_scope_sum_stop(self) -> None: ...
    def close_scope_sum_capture(self) -> None: ...
    def stop_scope_sum_from_reader(self) -> None: ...
    def scope_sum_status(self) -> tuple[int, int]: ...
    def scope_sum_identity(self) -> dict[str, object]: ...
    def get_scope_dma_runtime_metadata(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class ScopeSumGeometry:
    frame_samples: int
    ip_version: int
    gap_cycles: int
    buffer_samples: int = 2048
    frame_bytes: int = 4096
    waveform_samples: int = 0
    sample_decimation: int = 0
    padding_bytes: int = 0
    ndma_v1_compatible: bool = False

    def __post_init__(self) -> None:
        if (
            any(
                type(value) is not int
                for value in (self.ip_version, self.frame_samples, self.gap_cycles)
            )
            or self.ip_version not in (123, 124)
            or not 4 <= self.frame_samples <= 8188
            or self.frame_samples % 4
            or not 0 <= self.gap_cycles <= 65535
            or (
                self.buffer_samples,
                self.frame_bytes,
                self.waveform_samples,
                self.sample_decimation,
                self.padding_bytes,
            )
            != (2048, 4096, 0, 0, 0)
        ):
            raise ValueError("Invalid periodic-sum geometry")

    @property
    def interval_ticks(self) -> int:
        return self.frame_samples // 4 + self.gap_cycles


@dataclass(frozen=True, eq=False)
class ScopeSumPacket:
    """One decoded packet; per-window values are read-only uint64 arrays.

    Arrays (not tuples) keep decode and analysis vectorised: per-window
    Python work limited a reader to ~2700 packets/s (decode 203 us plus
    analysis 167 us per IP124 packet). ``records`` remains for callers that
    want (timestamp, sum) pairs.
    """

    ip_version: int
    sequence: int
    samples_per_window: int
    gap_cycles: int
    missed_windows: int
    stop_flush: bool
    timestamps: np.ndarray
    sums: np.ndarray
    raw: bytes

    @cached_property
    def records(self) -> tuple[tuple[int, int], ...]:
        return tuple(zip(self.timestamps.tolist(), self.sums.tolist(), strict=True))

    @property
    def signed_means(self) -> tuple[float, ...]:
        means = self.sums / self.samples_per_window - 32768.0
        return tuple(means.tolist())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScopeSumPacket):
            return NotImplemented
        return (
            (self.ip_version, self.sequence, self.samples_per_window, self.gap_cycles,
             self.missed_windows, self.stop_flush, self.raw)
            == (other.ip_version, other.sequence, other.samples_per_window, other.gap_cycles,
                other.missed_windows, other.stop_flush, other.raw)
            and np.array_equal(self.timestamps, other.timestamps)
            and np.array_equal(self.sums, other.sums)
        )



def decode_sum_packet(raw: bytes, geometry: ScopeSumGeometry | None = None) -> ScopeSumPacket:
    if len(raw) != 4096:
        raise ValueError("A periodic-sum packet must contain exactly 4096 bytes")
    first, magic, config, state = struct.unpack_from("<4Q", raw)
    version = MAGICS.get(magic)
    if version is None:
        raise ValueError("Unknown periodic-sum magic/ABI")
    n, gap, count = config & 0xFFFFFFFF, (config >> 32) & 0xFFFF, config >> 48
    actual = ScopeSumGeometry(n, version, gap)
    if geometry is not None and actual != geometry:
        raise ValueError("Sum packet differs from the armed IP/window/gap")
    capacity = 508 if version == 124 else 254
    stopped = bool(state >> 63)
    if not 1 <= count <= capacity or (count != capacity and not stopped):
        raise ValueError("Invalid sum valid_count or partial packet without STOP flush")
    owned = bytes(raw)
    interval = np.uint64(actual.interval_ticks)
    if version == 124:
        stored = np.frombuffer(owned, dtype="<u8", count=508, offset=32)
        if (stored[count:] != stored[count - 1]).any():
            raise ValueError("Invalid last-valid-value padding in IP124 packet")
        sums = stored[:count]
        # uint64 arithmetic wraps modulo 2**64, i.e. the original "& MASK64".
        timestamps = np.uint64(first) + np.arange(count, dtype=np.uint64) * interval
        timestamps.flags.writeable = False
    else:
        if any(owned[32 + count * 16 :]):
            raise ValueError("Nonzero IP123 packet padding")
        pairs = np.frombuffer(owned, dtype="<u8", count=2 * count, offset=32).reshape(count, 2)
        timestamps, sums = pairs[:, 0], pairs[:, 1]
        if int(timestamps[0]) != first:
            raise ValueError("Header timestamp differs from the first sum window")
    # Report the first failing window, sum check before interval check, as
    # the original per-window loop did.
    bad_sum = sums > np.uint64(n * 65535)
    deltas = timestamps[1:] - timestamps[:-1]
    bad_delta = np.zeros(count, dtype=bool)
    bad_delta[1:] = (deltas == 0) | (deltas >= np.uint64(1 << 63)) | (deltas % interval != 0)
    failing = bad_sum | bad_delta
    if failing.any():
        index = int(np.argmax(failing))
        if bad_sum[index]:
            raise ValueError("Sum exceeds N * full-scale ADC code")
        raise ValueError("Nonmonotonic or off-grid sum timestamp")
    return ScopeSumPacket(
        version,
        state & 0xFFFFFFFF,
        n,
        gap,
        (state >> 32) & 0x7FFFFFFF,
        stopped,
        timestamps,
        sums,
        owned,
    )


class ScopeSumContinuity:
    """Validate modulo-u32 sequence and modulo-u64 timestamp continuity."""

    def __init__(self) -> None:
        self.previous: ScopeSumPacket | None = None
        self.skipped_windows = 0
        # Packets accepted by the latest append_many() before any error.
        self.accepted = 0

    def append(self, packet: ScopeSumPacket) -> None:
        self.append_many((packet,))

    def append_many(self, packets: Sequence[ScopeSumPacket]) -> None:
        """Same as append() per packet; in-packet gaps are summed in one pass."""
        accepted = 0
        try:
            for packet in packets:
                self._check_boundary(packet)
                self.previous = packet
                accepted += 1
        finally:
            self.accepted = accepted
            self._add_internal_skips(packets[:accepted])

    def _add_internal_skips(self, packets: Sequence[ScopeSumPacket]) -> None:
        if not packets:
            return
        sizes = {packet.timestamps.size for packet in packets}
        intervals = {packet.samples_per_window // 4 + packet.gap_cycles for packet in packets}
        if len(sizes) == 1 and len(intervals) == 1:
            stamps = np.stack([packet.timestamps for packet in packets])
            steps = stamps[:, 1:] - stamps[:, :-1]  # wraps like & MASK64
            interval = np.uint64(intervals.pop())
            self.skipped_windows += sum((steps // interval).sum(axis=1).tolist()) - steps.size
            return
        for packet in packets:
            steps = packet.timestamps[1:] - packet.timestamps[:-1]
            if steps.size:
                interval = np.uint64(packet.samples_per_window // 4 + packet.gap_cycles)
                self.skipped_windows += sum((steps // interval).tolist()) - steps.size

    def _check_boundary(self, packet: ScopeSumPacket) -> None:
        old = self.previous
        interval = packet.samples_per_window // 4 + packet.gap_cycles
        if old is not None:
            if old.stop_flush:
                raise ValueError("Packet received after terminal STOP flush")
            if ((packet.sequence - old.sequence) & 0xFFFFFFFF) != 1:
                raise ValueError("Sum packet sequence discontinuity")
            if (old.ip_version, old.samples_per_window, old.gap_cycles) != (
                packet.ip_version,
                packet.samples_per_window,
                packet.gap_cycles,
            ):
                raise ValueError("Sum geometry changed during acquisition")
            delta = (int(packet.timestamps[0]) - int(old.timestamps[-1])) & MASK64
            if not 0 < delta < (1 << 63) or delta % interval:
                raise ValueError("Sum packet timestamps are nonmonotonic or off-grid")
            self.skipped_windows += delta // interval - 1


class ScopeSumWriter:
    """Single-threaded HDF5 sink; cache handles and resize once per batch."""

    def __init__(
        self, path: Path, metadata: dict[str, object], *, overwrite: bool = False,
    ) -> None:
        self._file = h5py.File(path, "w" if overwrite else "x")
        self._closed = False
        try:
            self._file.attrs.update(
                format="nlab-scope-sums",
                format_version=1,
                signal_origin="measurement",
                created_utc=datetime.now(UTC).isoformat(),
                timestamp_unit="8 ns",
                sum_unit="offset-binary ADC code sum",
                complete=False,
                metadata_json=json.dumps(
                    {k: v for k, v in metadata.items() if k != "configuration_yaml"},
                    sort_keys=True,
                ),
            )
            configuration = metadata.get("configuration_yaml", "")
            if configuration:
                # Use a dataset rather than an HDF5 attribute: full settings can
                # exceed attribute size limits and other capture tools use this key.
                self._file.create_dataset(
                    "configuration_yaml", data=str(configuration), dtype=h5py.string_dtype("utf-8")
                )
            self._packets = self._file.create_dataset(
                "packets", shape=(0, 4096), maxshape=(None, 4096), chunks=(16, 4096), dtype="u1"
            )
            self._records = tuple(
                self._file.create_dataset(
                    name, shape=(0,), maxshape=(None,), chunks=(8192,), dtype="<u8"
                )
                for name in ("timestamps", "sums")
            )
            self._counts = self._file.create_dataset(
                "valid_counts", shape=(0,), maxshape=(None,), chunks=(512,), dtype="<u2"
            )
        except Exception:
            self._file.close()
            raise

    def append(self, packet: ScopeSumPacket) -> None:
        self.append_many([packet])

    def append_many(self, packets: Sequence[ScopeSumPacket]) -> None:
        if self._closed:
            raise RuntimeError("Scope sum writer is closed")
        if not packets:
            return
        raw = np.frombuffer(b"".join(packet.raw for packet in packets), dtype="u1")
        counts = np.asarray([packet.timestamps.size for packet in packets], dtype="<u2")
        records = np.column_stack((
            np.concatenate([packet.timestamps for packet in packets]),
            np.concatenate([packet.sums for packet in packets]),
        )).astype("<u8", copy=False)
        row = self._packets.shape[0]
        self._packets.resize(row + len(packets), axis=0)
        self._packets[row:] = raw.reshape(-1, 4096)
        self._counts.resize(row + len(packets), axis=0)
        self._counts[row:] = counts
        for index, data in enumerate(self._records):
            start = data.shape[0]
            data.resize(start + len(records), axis=0)
            data[start:] = records[:, index]

    def close(
        self, *, complete: bool, error: str = "", missed_windows: int = 0,
        runtime_metadata: dict[str, object] | None = None,
    ) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._file.attrs.update(
                complete=False, cleanup_error=error, final_missed_windows=missed_windows
            )
            if runtime_metadata is not None:
                self._file.attrs["runtime_metadata_json"] = json.dumps(
                    runtime_metadata, sort_keys=True
                )
            self._file.flush()  # Flush all data before advertising completion.
            if complete:
                self._file.attrs["complete"] = True
                self._file.flush()
        except Exception as exc:
            # Best effort only: storage failures cannot provide transactional HDF5.
            try:
                self._file.attrs.update(
                    complete=False,
                    cleanup_error="; ".join(
                        part for part in (error, f"HDF5 flush: {type(exc).__name__}: {exc}") if part
                    ),
                )
                self._file.flush()
            except Exception:
                pass
            raise
        finally:
            self._file.close()


class ScopeSumBinaryWriter:
    """NSUM sink (``scope_sum_file``) with ScopeSumWriter's interface."""

    def __init__(
        self, path: Path, metadata: dict[str, object], *, overwrite: bool = False,
    ) -> None:
        header: dict[str, Any] = dict(
            format="nlab-scope-sums",
            format_version=1,
            container="nsum",
            signal_origin="measurement",
            created_utc=datetime.now(UTC).isoformat(),
            timestamp_unit="8 ns",
            sum_unit="offset-binary ADC code sum",
            packet_bytes=sum_file.PACKET_BYTES,
            metadata={k: v for k, v in metadata.items() if k != "configuration_yaml"},
            configuration_yaml=str(metadata.get("configuration_yaml", "") or ""),
        )
        self._writer = sum_file.SumBinaryWriter(path, header, overwrite=overwrite)

    def append(self, packet: ScopeSumPacket) -> None:
        self.append_many([packet])

    def append_many(self, packets: Sequence[ScopeSumPacket]) -> None:
        self._writer.append_raw([packet.raw for packet in packets])

    def close(
        self, *, complete: bool, error: str = "", missed_windows: int = 0,
        runtime_metadata: dict[str, object] | None = None,
    ) -> None:
        self._writer.close(dict(
            complete=complete,
            cleanup_error=error,
            final_missed_windows=missed_windows,
            runtime_metadata=runtime_metadata,
        ))


def sum_file_format(path: Path) -> str:
    """Recording format chosen by extension: ``"binary"`` (.bin) or ``"hdf5"``."""
    return "binary" if Path(path).suffix.lower() == sum_file.SUFFIX else "hdf5"


def open_sum_writer(
    path: Path, metadata: dict[str, object], *, overwrite: bool = False,
) -> ScopeSumWriter | ScopeSumBinaryWriter:
    if sum_file_format(path) == "binary":
        return ScopeSumBinaryWriter(path, metadata, overwrite=overwrite)
    return ScopeSumWriter(path, metadata, overwrite=overwrite)


class ScopeSumRecorder:
    """Bounded asynchronous sink; all file operations stay on one joined thread.

    The DMA reader only enqueues immutable decoded packets. A full queue is a
    recording error, never blocking backpressure or silent packet loss. STOP
    drains the queue and joins the writer before the capture worker can exit.
    """

    def __init__(
        self,
        path: Path,
        metadata: dict[str, object],
        *,
        overwrite: bool = False,
        queue_packets: int = 2048,
        batch_packets: int = 64,
        batch_seconds: float = 0.1,
    ) -> None:
        # 2048 packets ride out ~0.4 s of storage stall at 192-sample windows.
        # Batches of 64 halve HDF5's per-packet cost against 16 (14 vs 29 us),
        # and the writer shares the reader process's interpreter lock.
        if queue_packets < 1 or batch_packets < 1 or not 0 < batch_seconds <= 1:
            raise ValueError("Invalid Scope sum recording queue/batch limits")
        self._queue: Queue[ScopeSumPacket] = Queue(maxsize=queue_packets)
        self._batch_packets = batch_packets
        self._batch_seconds = batch_seconds
        self._label = "Binary" if sum_file_format(path) == "binary" else "HDF5"
        self._closing = threading.Event()
        self._ready = threading.Event()
        self._error = ""
        self._complete = False
        self._cleanup_error = ""
        self._missed = 0
        self._runtime_metadata: dict[str, object] | None = None
        self._thread = threading.Thread(
            target=self._run, args=(path, metadata, overwrite),
            name=f"Scope sum {self._label} writer", daemon=False,
        )
        self._thread.start()
        self._ready.wait()  # Create/validate the destination before acquisition is armed.
        if self._error:
            self._thread.join()
            raise RuntimeError(self._error)

    def append(self, packet: ScopeSumPacket) -> None:
        if self._error:
            raise RuntimeError(self._error)
        if self._closing.is_set():
            raise RuntimeError("Scope sum recorder is closed")
        try:
            self._queue.put_nowait(packet)
        except Full:
            raise RuntimeError(
                f"{self._label} recording queue is full; storage cannot keep up. "
                "Acquisition stopped; the saved prefix is incomplete."
            ) from None

    def _run(self, path: Path, metadata: dict[str, object], overwrite: bool) -> None:
        writer = None
        try:
            writer = open_sum_writer(path, metadata, overwrite=overwrite)
            self._ready.set()
            while not self._closing.is_set() or not self._queue.empty():
                batch: list[ScopeSumPacket] = []
                deadline = time.monotonic() + self._batch_seconds
                while len(batch) < self._batch_packets:
                    if self._closing.is_set() and self._queue.empty():
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        batch.append(self._queue.get(timeout=remaining))
                    except Empty:
                        break
                if batch:
                    writer.append_many(batch)
        except Exception as exc:
            self._error = f"{self._label} recording: {type(exc).__name__}: {exc}"
        finally:
            if writer is not None:
                error = "; ".join(part for part in (self._cleanup_error, self._error) if part)
                try:
                    writer.close(
                        complete=self._complete and not error,
                        error=error,
                        missed_windows=self._missed,
                        runtime_metadata=self._runtime_metadata,
                    )
                except Exception as exc:
                    self._error = "; ".join(
                        part for part in (
                            self._error, f"{self._label} close: {type(exc).__name__}: {exc}",
                        )
                        if part
                    )
            self._ready.set()

    def close(
        self, *, complete: bool, error: str = "", missed_windows: int = 0,
        runtime_metadata: dict[str, object] | None = None,
    ) -> None:
        if not self._closing.is_set():
            self._complete = complete
            self._cleanup_error = error
            self._missed = missed_windows
            self._runtime_metadata = runtime_metadata
            self._closing.set()
        self._thread.join()
        if self._error:
            raise RuntimeError(self._error)


def read_sum_capture(path: Path) -> Iterator[ScopeSumPacket]:
    """Reopen a sum recording (HDF5 or NSUM binary) and validate continuity.

    HDF5 files also check the stored decoded columns against the raw packets.
    """
    if sum_file.is_binary_sum_file(path):
        yield from _read_binary_sum_capture(path)
        return
    with h5py.File(path, "r") as source:
        if (
            source.attrs.get("format") != "nlab-scope-sums"
            or source.attrs.get("format_version") != 1
        ):
            raise ValueError("Unsupported Scope sum file")
        continuity = ScopeSumContinuity()
        offset = 0
        if len(source["packets"]) != len(source["valid_counts"]):
            raise ValueError("Truncated sum packet/count table")
        for index, row in enumerate(source["packets"]):
            packet = decode_sum_packet(row.tobytes())
            continuity.append(packet)
            count = packet.timestamps.size
            if int(source["valid_counts"][index]) != count:
                raise ValueError("Stored valid_count differs from raw packet")
            for name, expected in (("timestamps", packet.timestamps), ("sums", packet.sums)):
                if not np.array_equal(source[name][offset : offset + count], expected):
                    raise ValueError(f"Stored {name} differs from raw packet")
            offset += count
            yield packet
        if len(source["timestamps"]) != offset or len(source["sums"]) != offset:
            raise ValueError("Trailing or truncated sum record columns")


_BINARY_READ_PACKETS = 256


def _read_binary_sum_capture(path: Path) -> Iterator[ScopeSumPacket]:
    layout = sum_file.read_layout(path)
    continuity = ScopeSumContinuity()
    with Path(path).open("rb") as handle:
        for start in range(0, layout.packets, _BINARY_READ_PACKETS):
            stop = min(layout.packets, start + _BINARY_READ_PACKETS)
            data = sum_file.read_packet_bytes(handle, layout, start, stop)
            for offset in range(0, len(data), sum_file.PACKET_BYTES):
                packet = decode_sum_packet(data[offset:offset + sum_file.PACKET_BYTES])
                continuity.append(packet)
                yield packet


class ScopeSumAnalysis:
    """Bounded display history plus exact, sample-weighted session totals."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Ring of the newest display points; _point_total counts every point
        # ever written, so its remainder is the next write position.
        self._points = np.empty((_DISPLAY_POINTS, 2), dtype=np.float64)
        self._point_total = 0
        self._last_timestamp: int | None = None
        self._elapsed_ticks = 0
        self.packet_count = self.window_count = self.sample_count = self.signed_sum = 0
        self.missed_windows = self.coverage_ns = 0
        self.continuity = ScopeSumContinuity()
        # Set when this object mirrors an analysis that runs elsewhere (the
        # Scope sum reader process); snapshot() then returns it.
        self._mirror: ScopeSumSnapshot | None = None

    def adopt_snapshot(self, snapshot: ScopeSumSnapshot) -> None:
        """Mirror a snapshot produced by the reader process's own analysis."""
        with self._lock:
            self._mirror = snapshot
            self.packet_count = snapshot["packets"]
            self.window_count = snapshot["windows"]
            self.sample_count = snapshot["samples"]
            self.signed_sum = snapshot["signed_sum"]
            self.coverage_ns = snapshot["coverage_ns"]
            self.missed_windows = snapshot["missed"]

    def append(self, packet: ScopeSumPacket) -> None:
        self.append_many((packet,))

    def append_many(self, packets: Sequence[ScopeSumPacket]) -> None:
        """Same as append() per packet, in one vectorised pass per batch.

        Packets before a continuity error are accounted, then it is raised.
        """
        with self._lock:
            error: Exception | None = None
            try:
                self.continuity.append_many(packets)
            except Exception as exc:
                error = exc
            accepted = packets[: self.continuity.accepted]
            if accepted:
                self._accumulate(accepted)
            if error is not None:
                raise error

    def _accumulate(self, packets: Sequence[ScopeSumPacket]) -> None:
        # Continuity has verified one geometry and steps below 2**63.
        n = packets[0].samples_per_window
        timestamps = np.concatenate([packet.timestamps for packet in packets])
        sums = np.concatenate([packet.sums for packet in packets])
        count = int(timestamps.size)
        steps = np.empty(count, dtype=np.uint64)
        steps[0] = (
            0 if self._last_timestamp is None
            else (int(timestamps[0]) - self._last_timestamp) & MASK64
        )
        steps[1:] = timestamps[1:] - timestamps[:-1]  # wraps like & MASK64
        elapsed = self._elapsed_ticks + np.cumsum(steps.astype(np.int64))
        self._elapsed_ticks = int(elapsed[-1])
        self._last_timestamp = int(timestamps[-1])
        # Only the newest _DISPLAY_POINTS survive the bounded display history.
        tail = slice(-_DISPLAY_POINTS, None)
        self._write_points(elapsed[tail] * 8e-9, sums[tail] / n - 32768.0)
        self.signed_sum += int(sums.sum(dtype=np.uint64)) - n * 32768 * count
        self.packet_count += len(packets)
        self.window_count += count
        self.sample_count += n * count
        self.coverage_ns += n * 2 * count
        self.missed_windows = packets[-1].missed_windows

    def _write_points(
        self, seconds: npt.NDArray[np.floating[Any]], means: npt.NDArray[np.floating[Any]],
    ) -> None:
        count = seconds.size  # at most _DISPLAY_POINTS
        start = self._point_total % _DISPLAY_POINTS
        first = min(count, _DISPLAY_POINTS - start)
        self._points[start:start + first, 0] = seconds[:first]
        self._points[start:start + first, 1] = means[:first]
        self._points[:count - first, 0] = seconds[first:]
        self._points[:count - first, 1] = means[first:]
        self._point_total += count

    def _display_points(self) -> npt.NDArray[np.float64]:
        """A copy of the ring, oldest point first."""
        if self._point_total <= _DISPLAY_POINTS:
            return self._points[:self._point_total].copy()
        start = self._point_total % _DISPLAY_POINTS
        return np.concatenate((self._points[start:], self._points[:start]))

    def snapshot(self) -> ScopeSumSnapshot:
        with self._lock:
            if self._mirror is not None:
                mirrored: ScopeSumSnapshot = self._mirror.copy()
                return mirrored
            result: ScopeSumSnapshot = dict(
                points=self._display_points(),
                packets=self.packet_count,
                windows=self.window_count,
                samples=self.sample_count,
                signed_sum=self.signed_sum,
                coverage_ns=self.coverage_ns,
                missed=self.missed_windows,
                skipped=self.continuity.skipped_windows,
                mean=self.signed_sum / self.sample_count if self.sample_count else None,
            )

            previous = self.continuity.previous
            if previous is not None:
                result["ip_version"] = previous.ip_version
                result["samples_per_window"] = previous.samples_per_window
                result["gap_cycles"] = previous.gap_cycles
                result["interval_ns"] = (previous.samples_per_window // 4 + previous.gap_cycles) * 8

            return result


class ScopeSumCapture:
    """Single-reader session; STOP drains through terminal timeout before close.

    The backend's dedicated DMA context owns refills. The GUI's stop hook only
    clears the acquisition gate; it never cancels a healthy tail drain.
    """

    def __init__(
        self,
        backend: ScopeSumBackend,
        channel: int,
        *,
        configuration_yaml: str = "",
        process_uri: str | None = None,
    ) -> None:
        self.backend = backend
        self.channel = channel
        self.configuration_yaml = configuration_yaml
        self.analysis = ScopeSumAnalysis()
        self._stop = threading.Event()
        self._started = False
        # With a URI, run() reads in a reader process (scope_process.py) and
        # this object's analysis is fed from the forwarded packets.
        self._process_uri = process_uri
        self._session: ReaderSession | None = None

    def request_stop(self) -> None:
        self._stop.set()
        if self._process_uri is not None:
            session = self._session
            if session is not None:
                session.request_stop()
            return
        if self._started:
            self.backend.request_scope_sum_stop()

    def run(
        self,
        path: Path | None = None,
        ready: Callable[[], None] | None = None,
        max_packets: int | None = None,
        *,
        overwrite: bool = False,
    ) -> ScopeSumSnapshot:
        """Record sums; replacing an existing file requires explicit opt-in."""
        if self._process_uri is not None:
            return self._run_in_reader_process(path, ready, max_packets, overwrite=overwrite)
        writer = None
        error = ""
        complete = False
        missed = 0
        prepared = False
        runtime_metadata = None
        try:
            self.backend.prepare_dma_capture()
            prepared = True
            geometry = self.backend.get_scope_dma_geometry()
            if not isinstance(geometry, ScopeSumGeometry):
                raise ValueError("Periodic-sum capture requires mode 5 on IP123/IP124")
            if path is not None:
                metadata: dict[str, object] = dict(
                    channel=self.channel,
                    ip_version=geometry.ip_version,
                    samples_per_window=geometry.frame_samples,
                    gap_cycles=geometry.gap_cycles,
                    adc_sample_period_ns=2,
                    timestamp_period_ns=8,
                    calibration=None,
                    current_conversion="unqualified; ADC codes only",
                    producer=self.backend.scope_sum_identity(),
                    configuration_yaml=self.configuration_yaml,
                    configuration_snapshot="before acquisition"
                    if self.configuration_yaml else "not supplied",
                )
                writer = ScopeSumRecorder(path, metadata, overwrite=overwrite)
            deadline = None
            self._started = True
            while True:
                if self._stop.is_set() and deadline is None:
                    if self.analysis.packet_count == 0:
                        break  # Never arm a new buffer after cancellation.
                    self.backend.stop_scope_sum_from_reader()
                    deadline = time.monotonic() + 5.0
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError("Periodic-sum tail drain exceeded five seconds")
                try:
                    raw = self.backend.read_dma_raw_frame()
                except OSError as exc:
                    if self._stop.is_set() and exc.errno == errno.ECANCELED:
                        break
                    if self._stop.is_set() and exc.errno == errno.ETIMEDOUT:
                        status, missed = self.backend.scope_sum_status()
                        if status & (2 | 8):
                            raise RuntimeError(
                                f"Sum tail did not finish: status=0x{status:x}"
                            ) from None
                        if writer is not None:
                            # Snapshot after drain, on the DMA context's owner
                            # thread. Never add control RPCs to the running reader.
                            runtime_metadata = self.backend.get_scope_dma_runtime_metadata()
                        complete = True
                        break
                    raise
                packet = decode_sum_packet(raw, geometry)
                self.analysis.append(packet)
                if writer is not None:
                    writer.append(packet)
                if ready is not None:
                    ready()
                    ready = None
                if max_packets is not None and self.analysis.packet_count >= max_packets:
                    self._stop.set()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if prepared:
                try:
                    self.backend.stop_scope_sum_from_reader()
                    self.backend.close_scope_sum_capture()
                except Exception as exc:
                    error = (error + "; " if error else "") + f"Cleanup: {exc}"
                    complete = False
            self._started = False
            if writer is not None:
                try:
                    writer.close(
                        complete=complete and not error, error=error, missed_windows=missed,
                        runtime_metadata=runtime_metadata,
                    )
                except Exception as exc:
                    error = (error + "; " if error else "") + str(exc)
                    complete = False
        if error:
            raise RuntimeError(error)
        return self.analysis.snapshot() | {"complete": complete, "final_missed_windows": missed}

    def _run_in_reader_process(
        self,
        path: Path | None,
        ready: Callable[[], None] | None,
        max_packets: int | None,
        *,
        overwrite: bool,
    ) -> ScopeSumSnapshot:
        """Run the sum session in a reader process; analyse its packets here."""
        from nlab.hardware.digitizer.reader_process import ReaderSession, raise_child_error
        from nlab.hardware.digitizer.scope_process import ScopeSumRequest, sum_label, sum_main

        assert self._process_uri is not None
        latched = getattr(self.backend, "dma_fault_is_latched", None)
        if callable(latched) and latched():
            raise RuntimeError(
                "DMA fault is latched from a previous session -- call "
                "acknowledge_dma_recovery() before capturing again"
            )
        request = ScopeSumRequest(
            uri=self._process_uri,
            channel=self.channel,
            filepath=str(path) if path is not None else None,
            max_packets=max_packets,
            overwrite=overwrite,
            configuration_yaml=self.configuration_yaml,
            log_level=logging.getLogger().getEffectiveLevel(),
        )

        def note(key: str, value: object = None) -> None:
            # The reader process analyses every packet itself (exact totals,
            # continuity checks); the GUI only mirrors throttled snapshots, so
            # GUI load can never slow the reader or lose sums.
            if key == "analysis" and isinstance(value, dict):
                self.analysis.adopt_snapshot(cast(ScopeSumSnapshot, value))

        handlers: dict[str, Callable[..., None]] = {"note": note}
        if ready is not None:
            handlers["ready"] = ready
        session = ReaderSession(sum_main, request, label=sum_label(self.channel))
        self._session = session
        try:
            finished = session.run(self._stop, handlers)
        finally:
            self._session = None
            adopt = getattr(self.backend, "adopt_scope_dma_session_state", None)
            if callable(adopt):
                report = session.finished
                adopt(None if report is None else bool(report.get("fault_latched")))
        final = finished.get("snapshot")
        if isinstance(final, dict):
            self.analysis.adopt_snapshot(cast(ScopeSumSnapshot, final))
        raise_child_error(finished)
        missed = finished.get("final_missed_windows")
        return self.analysis.snapshot() | {
            "complete": bool(finished.get("complete")),
            "final_missed_windows": missed if isinstance(missed, int) else 0,
        }
