"""LiteX wrapper for PrimitiveAssembly."""

from litex.gen import *
from litex.soc.interconnect import stream
from litex.soc.interconnect.csr import *
from migen import *

from .stream_defs import primitive_layout, vertex_layout


class LiteXPrimitiveAssembly(LiteXModule, AutoCSR):
    """LiteX wrapper for the PrimitiveAssembly stage."""

    def __init__(self, platform=None, verilog_path="primitive_assembly.v"):
        # Status
        self._ready = CSRStatus(1, description="Primitive assembly ready")

        # Stream sink/source (vertices → primitives)
        self.sink_vertex = stream.Endpoint(vertex_layout)
        self.source_primitive = stream.Endpoint(primitive_layout)

        # Configuration
        self._prim_type = CSRStorage(
            3, description="Primitive type: 0=point, 1=line, 2=triangle, etc"
        )
        self._cull_mode = CSRStorage(
            2, description="Culling mode: 0=none, 1=front, 2=back, 3=both"
        )
        self._front_face = CSRStorage(1, description="Front face: 0=CCW, 1=CW")

        # # #

        gpu_params = {
            "i_clk": ClockSignal(),
            "i_rst": ResetSignal(),
            "o_ready": self._ready.status,
            "i_prim_config": Cat(
                self._prim_type.storage,
                self._cull_mode.storage,
                self._front_face.storage,
            ),
            # Stream input (is_vertex)
            "i_is_vertex__valid": self.sink_vertex.valid,
            "o_is_vertex__ready": self.sink_vertex.ready,
            "i_is_vertex__payload": self.sink_vertex.payload.data,
            # Stream output (os_primitive)
            "o_os_primitive__valid": self.source_primitive.valid,
            "i_os_primitive__ready": self.source_primitive.ready,
            "o_os_primitive__payload": self.source_primitive.payload.data,
        }

        self.specials += Instance("PrimitiveAssembly", **gpu_params)


__all__ = ["LiteXPrimitiveAssembly"]
