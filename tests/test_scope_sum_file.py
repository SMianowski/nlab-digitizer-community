"""NSUM binary periodic-sum container and its recorder integration."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from nlab.hardware.digitizer import scope_sum_file
from nlab.hardware.digitizer.scope_sums import (
    ScopeSumBinaryWriter,
    ScopeSumRecorder,
    ScopeSumWriter,
    decode_sum_packet,
    open_sum_writer,
    read_sum_capture,
    sum_file_format,
)


def _raw(sequence: int, *, count: int = 508, stopped: bool = False) -> bytes:
    raw = bytearray(4096)
    struct.pack_into(
        "<4Q", raw, 0, 1000 + sequence * 508 * 64, 0x32534D5553504456,
        256 | (count << 48), sequence | (int(stopped) << 63),
    )
    sums = 256 * 32768 + np.minimum(np.arange(508), count - 1)
    np.frombuffer(raw, dtype="<u8", count=508, offset=32)[:] = sums
    return bytes(raw)


def _packets(total: int, *, tail: int = 7) -> list:
    packets = [decode_sum_packet(_raw(i)) for i in range(total - 1)]
    return [*packets, decode_sum_packet(_raw(total - 1, count=tail, stopped=True))]


@pytest.mark.parametrize(
    "name,expected",
    [("a.bin", "binary"), ("a.BIN", "binary"), ("a.h5", "hdf5"), ("a", "hdf5")],
)
def test_extension_selects_the_recording_format(tmp_path, name, expected):
    assert sum_file_format(tmp_path / name) == expected
    writer = open_sum_writer(tmp_path / name, {})
    assert isinstance(writer, ScopeSumBinaryWriter if expected == "binary" else ScopeSumWriter)
    writer.close(complete=False)


def test_binary_recording_keeps_packets_unchanged_and_trailer_last(tmp_path):
    path = tmp_path / "sums.bin"
    packets = _packets(300)
    recorder = ScopeSumRecorder(path, {"channel": 1, "configuration_yaml": "x: 2"})
    for packet in packets:
        recorder.append(packet)
    runtime = {"transport": "iiod-batched", "readbuf_batch_frames": 64}
    recorder.close(complete=True, missed_windows=299, runtime_metadata=runtime)

    data = path.read_bytes()
    layout = scope_sum_file.read_layout(path)
    # One header block, every packet unchanged and block aligned, one trailer.
    assert len(data) == 4096 * (1 + len(packets) + 1) and layout.data_offset == 4096
    assert data[4096:-4096] == b"".join(packet.raw for packet in packets)
    assert layout.header["metadata"] == {"channel": 1}
    assert layout.header["configuration_yaml"] == "x: 2"
    assert layout.header["format"] == "nlab-scope-sums" and layout.header["container"] == "nsum"
    assert layout.trailer == {
        "complete": True, "cleanup_error": "", "final_missed_windows": 299,
        "runtime_metadata": runtime, "packets": len(packets),
    }
    assert [packet.raw for packet in read_sum_capture(path)] == [p.raw for p in packets]


def test_binary_header_longer_than_a_block_keeps_packets_aligned(tmp_path):
    path = tmp_path / "big.bin"
    writer = ScopeSumBinaryWriter(path, {"configuration_yaml": "k: v\n" * 2000})
    writer.append_many(_packets(3))
    writer.close(complete=True)
    layout = scope_sum_file.read_layout(path)
    assert layout.data_offset % 4096 == 0 and layout.data_offset > 4096
    assert layout.packets == 3 and len(list(read_sum_capture(path))) == 3


def test_binary_trailer_shortens_only_an_oversized_error(tmp_path):
    path = tmp_path / "error.bin"
    writer = ScopeSumBinaryWriter(path, {})
    writer.append_many(_packets(2))
    writer.close(complete=False, error="E" * 10_000, missed_windows=5)
    trailer = scope_sum_file.read_layout(path).trailer
    assert trailer is not None
    assert trailer["cleanup_error"].endswith("[truncated]")
    kept = (trailer["complete"], trailer["final_missed_windows"], trailer["packets"])
    assert kept == (False, 5, 2)


def test_binary_reader_rejects_other_files(tmp_path):
    other = tmp_path / "scope.bin"
    other.write_bytes(b"NDMA" + bytes(100))
    assert not scope_sum_file.is_binary_sum_file(other)
    with pytest.raises(ValueError, match="Not an NSUM"):
        scope_sum_file.read_layout(other)
    newer = tmp_path / "newer.bin"
    newer.write_bytes(struct.pack("<4sHHI", b"NSUM", 2, 0, 2) + b"{}")
    with pytest.raises(ValueError, match="Unsupported NSUM version 2"):
        scope_sum_file.read_layout(newer)


def test_binary_recording_never_replaces_without_overwrite(tmp_path):
    path = tmp_path / "keep.bin"
    path.write_bytes(b"original")
    with pytest.raises(RuntimeError, match="Binary recording"):
        ScopeSumRecorder(path, {})
    assert path.read_bytes() == b"original"
    recorder = ScopeSumRecorder(path, {}, overwrite=True)
    recorder.close(complete=True)
    layout = scope_sum_file.read_layout(path)
    assert layout.packets == 0 and json.loads(json.dumps(layout.trailer))["complete"]
