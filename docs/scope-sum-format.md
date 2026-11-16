# Scope periodic-sum capture format (version 1)

Periodic sums are recorded in one of two containers, chosen by file extension
(**Current > Format**): HDF5 (`.h5`, `.hdf5`) or NSUM binary (`.bin`, see
[NSUM binary container](#nsum-binary-container) below). Both store the same
original 4096-byte packets and the same metadata; Current Capture Analysis and
`read_sum_capture()` open either.

The HDF5 root attribute `format` is `nlab-scope-sums`; `format_version` is 1.
This container stores measured Scope aggregates. It is independent of NDMA
waveform and MCA event formats. IP123 and IP124 wire packets are decoded by
`nlab.hardware.digitizer.scope_sums`.

| Dataset | Type / shape | Meaning |
| --- | --- | --- |
| `packets` | uint8, `(packet_count, 4096)` | Original, complete DMA packets, including header and padding. |
| `valid_counts` | little-endian uint16, `(packet_count,)` | Valid window records in each packet; padding is excluded. |
| `timestamps` | little-endian uint64, `(window_count,)` | Native window timestamps, in 8 ns counter ticks, modulo 2^64. |
| `sums` | little-endian uint64, `(window_count,)` | Sum of all N offset-binary ADC samples in each window. |
| `configuration_yaml` | optional scalar UTF-8 string | Complete pre-acquisition Save Settings document in new GUI captures. |

For signed int16 ADC samples, unsigned code `u = x + 32768`, so the signed
window mean is `S / N - 32768`. ADC samples are 2 ns apart. Each interval
starts every `(N / 4 + gap_cycles) * 8 ns`. Exposure is the number of measured
samples times 2 ns; missed windows and gap intervals do not contribute.
Session means weight measured samples rather than packets. No waveform,
noise extrema, or absolute current calibration is reconstructed.

Root attributes include `signal_origin=measurement`, `created_utc`,
`timestamp_unit`, `sum_unit`, `metadata_json`, `complete`, `cleanup_error`,
and `final_missed_windows`. Cleanly drained new recordings also have optional
`runtime_metadata_json`: actual transport, READBUF batch size, selected kernel
depth and queue counters sampled after STOP drain and before buffer close. This
snapshot does not add control reads to the running DMA loop. Older files remain
readable without it. The UTC field is host file-creation time; it does
not synchronize the FPGA counter to UTC. `metadata_json` records channel,
IP, N, gap, ADC/counter periods, producer/endpoint identity, calibration status,
and the driver reference used by this client. Physical Scope A/B mapping
still requires independent qualification.

Remote acquisition uses the driver's qualified kernel-buffer depth and up to
64 packets per iiod READBUF request. Each 4096-byte packet is still decoded and
recorded separately; batching never changes N, window timing or the file ABI.
Local or unqualified DMA retains native libiio. A normal batched STOP keeps
receiving complete packets through an explicit server ETIMEDOUT reply, checks
pending/invalid status, then closes the stream. A socket timeout or malformed
response cannot establish a clean drain and leaves the capture incomplete.

Recording uses a separate writer thread with a 2048-packet queue (about 0.4 s
at 192-sample windows). Appends batch up to 64 packets or 100 ms, whichever
occurs first; no file calls run on the DMA reader. The writer still shares the
reader process's interpreter lock: HDF5 costs about 14 us per packet in batches
of 64 (29 us in batches of 16), NSUM binary about 1.5 us. Queue overflow is a capture error rather than blocking the reader or
silently dropping recorded packets. Stop joins the writer after draining its
queue, including a final partial batch. Raw packets and exact decoded columns
retain the version 1 layout above.

A new file starts with `complete=false`. Only successful STOP tail drain,
buffer release, and queued-file-write completion can mark it complete.
Completion describes clean finalization, not absence of missed hardware windows;
check native timestamp gaps and the missed-window counters for acquisition loss.
Unexpected read/drain/release/write errors retain an incomplete prefix and an
error string. Existing files are protected
by exclusive creation. HDF5 is not a crash-transaction log: a process or storage
failure can leave an unreadable/truncated file even with incomplete status.

`read_sum_capture(path)` iterates validated packets. It checks wire geometry,
magic, valid counts, padding, sums, modulo sequence/timestamps, and agreement
between original packets and decoded datasets. Valid incomplete prefixes can
be inspected. **Tools > Current Capture Analysis** provides a dedicated offline
browser, also opened by **Open sum capture** in Current. Time chunks use native
counter differences before converting to microseconds, retaining uint64 wrap
and native tick precision at large counter values. The viewer binary-searches
stored timestamps and validates only selected packets and their neighbors.
This is bounded offline inspection, not full-file integrity certification.
Statistics and a 128-bin histogram include every selected window mean; a sampled
trace preview is limited to 100,000 points and breaks across missing windows.
Population std uses ddof=0; adjacent noise is the RMS successive difference
divided by sqrt(2), excluding gaps. Neither estimates within-window ADC noise.

The optional `configuration_yaml` dataset uses the same versioned document as
Save Settings, including connection/topology, all hardware channels, shared and
external settings, and application settings. The GUI snapshots it after setting
the effective sum mode/N/gap and before starting the DMA reader. Snapshot failure
aborts startup and restores Scope settings. This adds no metadata RPCs to the
running DMA loop. `metadata_json.configuration_snapshot` records `before
acquisition` or `not supplied`; headless callers can pass `configuration_yaml`
to `create_sum_capture`. Large configuration text stays in a dataset rather than
an attribute. The viewer reads/exports it without applying it to hardware.
Older version 1 files without it remain supported and do not imply a recovered
configuration. This is an additive extension, with no wire-format change.

## NSUM binary container

A `.bin` sum recording starts with the 4-byte magic `NSUM`, never `NDMA`: the
NDMA formats describe waveforms and list-mode records, and sums must not reach
their readers. All integers are little-endian.

| Part | Layout |
| --- | --- |
| Preamble | `char[4] "NSUM"`, `u16 version` (1), `u16 reserved` (0), `u32 header_bytes` |
| Header | `header_bytes` of UTF-8 JSON, zero padded so the first packet starts at a multiple of 4096 bytes |
| Packets | The original 4096-byte IP123/IP124 packets, unchanged and in order |
| Trailer | One 4096-byte block: `char[16] "NSUM-END" "TRAILER\0"`, `u32 json_bytes`, UTF-8 JSON, zero padding |

The header JSON holds `format` (`nlab-scope-sums`), `format_version` (1),
`container` (`nsum`), `signal_origin`, `created_utc`, `timestamp_unit`,
`sum_unit`, `packet_bytes`, `metadata` (the same mapping as HDF5
`metadata_json`) and `configuration_yaml` (empty when not supplied). The
trailer JSON holds `complete`, `cleanup_error`, `final_missed_windows`,
`runtime_metadata` (or null) and `packets`, which must equal the number of
packet blocks. Bytes 8-16 of the trailer marker can never equal a packet's ABI
magic, so the trailer cannot be mistaken for a packet.

There are no decoded columns: timestamps and sums are decoded from the packets
when read, and valid counts follow from the producer contract that every packet
except a final STOP flush is full. Readers find packet `i` at
`data_offset + 4096 * i` without an index.

The trailer is written only after every packet has been flushed to disk, and is
itself flushed before the file closes. A file without a trailer did not close
(process or power loss): its packets up to the last whole 4096-byte block stay
readable, and readers report it incomplete. A partial final block is ignored.
Existing files are protected by exclusive creation, as for HDF5.

## Display history

The 8192-point display history is bounded independently of the recording and
sample-weighted totals. GUI updates do not discard recorded windows. IP123
and IP124 host fixtures test the declared wire assumptions; live smoke tests
exercise IP124 on both channels, without claiming detector calibration.
