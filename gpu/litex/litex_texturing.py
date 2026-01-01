"""LiteX wrapper for Texturing."""

from litex.gen import *
from litex.soc.interconnect import stream
from litex.soc.interconnect.csr import *
from migen import *

from .stream_defs import fragment_layout


class LiteXTexturing(LiteXModule, AutoCSR):
    """LiteX wrapper for the Texturing stage."""

    def __init__(self, platform=None, verilog_path="texturing.v"):
        # Status
        self._ready = CSRStatus(1, description="Texturing ready")

        # Stream sink/source (fragments in/out)
        self.sink_fragment = stream.Endpoint(fragment_layout)
        self.source_fragment = stream.Endpoint(fragment_layout)

        # # #

        gpu_params = {
            # Texturing is currently a pass-through, no config ports yet
            # Stream input (is_fragment)
            "i_is_fragment__valid": self.sink_fragment.valid,
            "o_is_fragment__ready": self.sink_fragment.ready,
            "i_is_fragment__payload": self.sink_fragment.payload.data,
            # Stream output (os_fragment)
            "o_os_fragment__valid": self.source_fragment.valid,
            "i_os_fragment__ready": self.source_fragment.ready,
            "o_os_fragment__payload": self.source_fragment.payload.data,
        }

        self.specials += Instance("Texturing", **gpu_params)


__all__ = ["LiteXTexturing"]
