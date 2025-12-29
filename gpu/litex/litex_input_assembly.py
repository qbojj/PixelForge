"""LiteX wrapper for InputAssembly."""

from litex.gen import *
from litex.soc.interconnect import stream, wishbone
from litex.soc.interconnect.csr import *
from migen import *

from .stream_defs import index_layout, vertex_layout


class LiteXInputAssembly(LiteXModule, AutoCSR):
    """LiteX wrapper for the InputAssembly stage."""

    def __init__(self, platform=None, verilog_path="input_assembly.v"):
        # Wishbone master bus for vertex buffer reads
        self.bus = wishbone.Interface(data_width=32, adr_width=30)

        # Stream sink/source (indices → vertices)
        self.sink_index = stream.Endpoint(index_layout)
        self.source_vertex = stream.Endpoint(vertex_layout)

        # Status
        self._ready = CSRStatus(1, description="Input assembly ready")

        # Vertex attribute configuration
        self._pos_offset = CSRStorage(
            16, description="Position attribute offset in bytes"
        )
        self._pos_stride = CSRStorage(
            16, description="Position attribute stride in bytes"
        )
        self._pos_format = CSRStorage(4, description="Position attribute format")

        self._norm_offset = CSRStorage(16, description="Normal attribute offset")
        self._norm_stride = CSRStorage(16, description="Normal attribute stride")
        self._norm_format = CSRStorage(4, description="Normal attribute format")

        self._col_offset = CSRStorage(16, description="Color attribute offset")
        self._col_stride = CSRStorage(16, description="Color attribute stride")
        self._col_format = CSRStorage(4, description="Color attribute format")

        # # #

        gpu_params = {
            "i_clk": ClockSignal(),
            "i_rst": ResetSignal(),
            "o_ready": self._ready.status,
            "i_c_pos": Cat(
                self._pos_format.storage,
                self._pos_stride.storage,
                self._pos_offset.storage,
            ),
            "i_c_norm": Cat(
                self._norm_format.storage,
                self._norm_stride.storage,
                self._norm_offset.storage,
            ),
            "i_c_col": Cat(
                self._col_format.storage,
                self._col_stride.storage,
                self._col_offset.storage,
            ),
            # Wishbone connections
            "o_bus__adr": self.bus.adr,
            "i_bus__dat_r": self.bus.dat_r,
            "o_bus__sel": self.bus.sel,
            "o_bus__cyc": self.bus.cyc,
            "o_bus__stb": self.bus.stb,
            "i_bus__ack": self.bus.ack,
            # Stream input (is_index)
            "i_is_index__valid": self.sink_index.valid,
            "o_is_index__ready": self.sink_index.ready,
            "i_is_index__payload": self.sink_index.payload.data,
            # Stream output (os_vertex)
            "o_os_vertex__valid": self.source_vertex.valid,
            "i_os_vertex__ready": self.source_vertex.ready,
            "o_os_vertex__payload": self.source_vertex.payload.data,
        }

        self.specials += Instance("InputAssembly", **gpu_params)


__all__ = ["LiteXInputAssembly"]
