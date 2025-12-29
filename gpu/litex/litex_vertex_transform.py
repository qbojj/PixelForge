"""LiteX wrapper for VertexTransform."""

from litex.gen import *
from litex.soc.interconnect import stream
from litex.soc.interconnect.csr import *
from migen import *

from .stream_defs import vertex_layout


class LiteXVertexTransform(LiteXModule, AutoCSR):
    """LiteX wrapper for the VertexTransform stage."""

    def __init__(self, platform=None, verilog_path="vertex_transform.v"):
        # Status
        self._ready = CSRStatus(1, description="Vertex transform ready")

        # Stream sink/source (vertices in/out)
        self.sink_vertex = stream.Endpoint(vertex_layout)
        self.source_vertex = stream.Endpoint(vertex_layout)

        # Transform configuration (enable flags for MVP, normal transforms)
        self._enabled = CSRStorage(4, description="Enable transform stages")

        # Transformation matrices (simplified - 16 entries for 4x4 matrices)
        self._position_mv_0 = CSRStorage(32, description="Position MV matrix [0]")
        self._position_mv_1 = CSRStorage(32, description="Position MV matrix [1]")
        self._position_mv_2 = CSRStorage(32, description="Position MV matrix [2]")
        self._position_mv_3 = CSRStorage(32, description="Position MV matrix [3]")

        self._position_p_0 = CSRStorage(32, description="Position P matrix [0]")
        self._position_p_1 = CSRStorage(32, description="Position P matrix [1]")
        self._position_p_2 = CSRStorage(32, description="Position P matrix [2]")
        self._position_p_3 = CSRStorage(32, description="Position P matrix [3]")

        # # #

        gpu_params = {
            "i_clk": ClockSignal(),
            "i_rst": ResetSignal(),
            "o_ready": self._ready.status,
            "i_enabled": self._enabled.storage,
            "i_position_mv": Cat(
                self._position_mv_0.storage,
                self._position_mv_1.storage,
                self._position_mv_2.storage,
                self._position_mv_3.storage,
            ),
            "i_position_p": Cat(
                self._position_p_0.storage,
                self._position_p_1.storage,
                self._position_p_2.storage,
                self._position_p_3.storage,
            ),
            "i_normal_mv_inv_t": 0,  # Not yet configured
            "i_texture_transforms": 0,  # Not yet configured
            # Stream input (is_vertex)
            "i_is_vertex__valid": self.sink_vertex.valid,
            "o_is_vertex__ready": self.sink_vertex.ready,
            "i_is_vertex__payload": self.sink_vertex.payload.data,
            # Stream output (os_vertex)
            "o_os_vertex__valid": self.source_vertex.valid,
            "i_os_vertex__ready": self.source_vertex.ready,
            "o_os_vertex__payload": self.source_vertex.payload.data,
        }

        self.specials += Instance("VertexTransform", **gpu_params)


__all__ = ["LiteXVertexTransform"]
