from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .acquisition_profile import AcquisitionProfile
from .backends.base import DigitizerBackend, IDSBackend
from .device_information import DigitizerInformation
from .diagnostics import GlobalDiagnosticReading
from .dma import IIOMcaDmaStreamer, IIOScopeDmaStreamer
from .hv import HVSupply
from .mca import MultiChannelAnalyzer
from .scope import Scope

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DigitizerCapabilities:
    """Hardware features discovered once while a channel is connected."""

    scope_dma: bool
    mca: bool
    mca_dma: bool
    ids: bool


class Digitizer:
    """Entry point for instrument access.

    Construct with :meth:`from_iio`, then access subsystems through
    ``scope``, ``mca``, and ``hv``. Scope/MCA and IDS/HV use the digitizer's
    IIO device tree through independent context owners where threading or
    blocking I/O requires it.

    Hardware I/O uses explicit get_*/set_* methods rather than properties.
    This makes the I/O cost visible at call sites and avoids surprising side
    effects inside expressions.

    Example::

        d = Digitizer.from_iio(channel=0, uri="ip:192.168.3.1:30431")
        d.scope.set_trigger_level(5000)
        d.mca.set_energy_bin(4)
        d.hv.set_voltage(30.0)

    Example — without HV (e.g. bench testing DPP only)::

        d = Digitizer.from_iio(channel=0, with_ids=False)
        # d.hv is None
    """

    def __init__(
        self,
        backend: DigitizerBackend,
        ids_backend: IDSBackend | None = None,
        scope_dma: IIOScopeDmaStreamer | None = None,
        mca_dma: IIOMcaDmaStreamer | None = None,
    ) -> None:
        self._backend = backend
        self._ids_backend = ids_backend
        self.scope = Scope(backend)
        self.mca   = MultiChannelAnalyzer(backend)
        self.hv: HVSupply | None = HVSupply(ids_backend) if ids_backend else None
        self.scope_dma = scope_dma
        self.mca_dma = mca_dma
        mca_present = getattr(backend, "mca_hardware_present", lambda: True)()
        self.capabilities = DigitizerCapabilities(
            scope_dma=scope_dma is not None,
            mca=bool(mca_present),
            mca_dma=bool(mca_present and mca_dma is not None),
            ids=ids_backend is not None,
        )

    def close(self) -> None:
        if self.hv is not None:
            self.hv.safe_shutdown()
        self._backend.close()
        if self._ids_backend is not None:
            self._ids_backend.close()

    def mca_available(self) -> bool:
        """Whether d.mca's methods are expected to work on this channel.

        Kept as a compatibility convenience for callers written before the
        explicit :attr:`capabilities` model was introduced.
        """
        return self.capabilities.mca

    def get_global_diagnostics(self) -> list[GlobalDiagnosticReading]:
        """Read sensors shared by the complete digitizer, not one channel."""
        if self.hv is None:
            return []
        return self.hv.get_global_diagnostics()

    def get_device_information(self) -> DigitizerInformation | None:
        """Return immutable connection-time identity; never poll hardware here."""
        information = getattr(self._backend, "get_device_information", lambda: None)()
        return information if isinstance(information, DigitizerInformation) else None

    @classmethod
    def from_iio(
        cls,
        channel: int,
        uri: str = "ip:192.168.10.128:30431",
        *,
        with_ids: bool = True,
        profile: AcquisitionProfile | None = None,
        dac_channel: int | None = None,
    ) -> Digitizer:
        """Create a Digitizer backed by the on-FPGA IIO device tree.

        The HV module is accessed through its AD5686R, MCP3564(R), TMP117,
        and ADS5407 IIO devices; pass ``with_ids=False`` to leave ``d.hv``
        as ``None``. If a required IDS device is absent or its initial HV
        ADC/DAC readback reports EINVAL, EIO, ENODEV or ENXIO, construction
        logs a warning and also leaves ``d.hv`` as ``None`` so scope/MCA
        operation remains available. Transport errors still propagate.
        The removed SiPM controls are reported unavailable.
        When a channel exposes
        both vdpp_pulse_processor and vdpp_lm_frame, d.mca_dma is an
        IIOMcaDmaStreamer implementing the fixed 1024-record lifecycle in
        mca-architecture.md; it remains None on older firmware without
        lm_frame. d.mca itself is usable: IIODigitizerBackend
        implements MCABackend against vdpp-pulse-processor.c/
        vdpp-input-filter.c, except for a handful of methods with no
        matching hardware register (see iio_backend.py's module docstring)
        — and only if this channel's firmware actually has those two
        devices; if not, d.mca's methods raise RuntimeError instead of
        NotImplementedError, since that's a firmware/device-tree gap on a
        specific board, not a gap in this backend.
        d.scope_dma is an IIOScopeDmaStreamer (see dma.py) and pulls
        full-resolution frames by looping read_dma_raw_frame().
        """
        from .backends.iio_backend import IIODigitizerBackend
        from .backends.iio_ids_backend import IIOIDSBackend, IIOIDSUnavailableError

        # The DPP and IDS backends open separate IIO contexts and share no
        # state, so build them concurrently: IDS construction is dominated by
        # its required HV-feedback probe (~180 ms MCP3564R conversion, live).
        # Each backend is only handed to its owners after both have finished.
        ids_pool = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"iio-ids-ch{channel}-open")
            if with_ids
            else None
        )
        ids_future = ids_pool.submit(IIOIDSBackend, channel, uri) if ids_pool else None
        try:
            backend = IIODigitizerBackend(channel, uri, dac_channel=dac_channel, profile=profile)
        except BaseException:
            if ids_future is not None:
                try:
                    ids_future.result().close()
                except Exception:
                    pass  # the DPP failure is the error to report
            raise
        finally:
            if ids_pool is not None:
                ids_pool.shutdown(wait=True)
        ids = None
        if ids_future is not None:
            try:
                ids = ids_future.result()
            except IIOIDSUnavailableError as exc:
                # Scope and MCA are independent IIO cores. A firmware image
                # may omit an IDS converter or expose an unusable HV readback;
                # keep the acquired DPP backend and disable only this
                # channel's PSU functionality.
                log.warning(
                    "IIO channel %d: IDS/PSU unavailable; continuing without it: %s",
                    channel,
                    exc,
                )
            except BaseException:
                backend.close()
                raise
        from .reader_process import in_process_requested

        # Scope and list-mode DMA read in their own processes so GUI work
        # cannot starve them of the interpreter lock (reader_process.py).
        process_uri = None if in_process_requested() else uri
        scope_dma = IIOScopeDmaStreamer(backend, channel, process_uri=process_uri)

        mca_dma = (
            IIOMcaDmaStreamer(backend, channel, process_uri=process_uri)
            if backend.mca_dma_hardware_present()
            else None
        )
        return cls(backend, ids, scope_dma=scope_dma, mca_dma=mca_dma)
