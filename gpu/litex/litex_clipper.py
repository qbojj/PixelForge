"""LiteX wrapper for PrimitiveClipper."""

from litex.gen import *
from litex.soc.interconnect import stream
from litex.soc.interconnect.csr import *
from migen import *

from .stream_defs import vertex_layout


class LiteXPrimitiveClipper(LiteXModule, AutoCSR):
    """LiteX wrapper for the PrimitiveClipper stage."""

    def __init__(self, platform=None, verilog_path="primitive_clipper.v"):
        # Status
        self._ready = CSRStatus(1, description="Primitive clipper ready")

        # Stream sink/source (vertices in/out)
        self.sink_vertex = stream.Endpoint(vertex_layout)
        self.source_vertex = stream.Endpoint(vertex_layout)

        # Clipping configuration
        self._clip_enable = CSRStorage(1, description="Enable clipping")
        self._clip_near = CSRStorage(32, description="Near clipping plane")
        self._clip_far = CSRStorage(32, description="Far clipping plane")

        # # #

        gpu_params = {
            "i_clk": ClockSignal(),
            "i_rst": ResetSignal(),
            "o_ready": self._ready.status,
            "i_prim_type": 0,  # Clipper uses prim_type, clip config not yet implemented
            # Stream input (is_vertex)
            "i_is_vertex__valid": self.sink_vertex.valid,
            "o_is_vertex__ready": self.sink_vertex.ready,
            "i_is_vertex__payload": self.sink_vertex.payload.data,
            # Stream output (os_vertex)
            "o_os_vertex__valid": self.source_vertex.valid,
            "i_os_vertex__ready": self.source_vertex.ready,
            "o_os_vertex__payload": self.source_vertex.payload.data,
        }

        self.specials += Instance("PrimitiveClipper", **gpu_params)


__all__ = ["LiteXPrimitiveClipper"]
