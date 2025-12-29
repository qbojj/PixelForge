"""LiteX wrapper for IndexGenerator."""

from litex.gen import *
from litex.soc.interconnect import stream, wishbone
from litex.soc.interconnect.csr import *
from migen import *

from .stream_defs import index_layout


class LiteXIndexGenerator(LiteXModule, AutoCSR):
    """LiteX wrapper for the IndexGenerator stage."""

    def __init__(self, platform=None, verilog_path="index_generator.v"):
        # Wishbone master bus for index buffer reads
        self.bus = wishbone.Interface(data_width=32, adr_width=30)

        # Stream source (indices)
        self.source_index = stream.Endpoint(index_layout)

        # Control and status
        self._start = CSRStorage(1, description="Start index generation")
        self._ready = CSRStatus(1, description="Index generator ready")

        # Configuration
        self._index_address = CSRStorage(32, description="Index buffer base address")
        self._index_count = CSRStorage(32, description="Number of indices to generate")
        self._index_kind = CSRStorage(
            2, description="Index type: 0=none, 1=u8, 2=u16, 3=u32"
        )

        # # #

        gpu_params = {
            "i_clk": ClockSignal(),
            "i_rst": ResetSignal(),
            "i_start": self._start.storage,
            "o_ready": self._ready.status,
            "i_c_address": self._index_address.storage,
            "i_c_count": self._index_count.storage,
            "i_c_kind": self._index_kind.storage,
            # Wishbone connections
            "o_bus__adr": self.bus.adr,
            "i_bus__dat_r": self.bus.dat_r,
            "o_bus__sel": self.bus.sel,
            "o_bus__cyc": self.bus.cyc,
            "o_bus__stb": self.bus.stb,
            "i_bus__ack": self.bus.ack,
            # Stream output (os_index)
            "o_os_index__valid": self.source_index.valid,
            "i_os_index__ready": self.source_index.ready,
            "o_os_index__payload": self.source_index.payload.data,
        }

        self.specials += Instance("IndexGenerator", **gpu_params)


__all__ = ["LiteXIndexGenerator"]
