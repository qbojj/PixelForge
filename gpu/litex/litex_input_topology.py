"""LiteX wrapper for InputTopologyProcessor."""

from litex.gen import *
from litex.soc.interconnect import stream
from litex.soc.interconnect.csr import *
from migen import *

from .stream_defs import index_layout


class LiteXInputTopologyProcessor(LiteXModule, AutoCSR):
    """LiteX wrapper for the InputTopologyProcessor stage."""

    def __init__(self, platform=None, verilog_path="input_topology.v"):
        # Stream sink/source (indices)
        self.sink_index = stream.Endpoint(index_layout)
        self.source_index = stream.Endpoint(index_layout)
        # Status
        self._ready = CSRStatus(1, description="Input topology processor ready")

        # Topology configuration
        self._input_topology = CSRStorage(
            3, description="Input topology: 0=point_list, 1=line_list, etc"
        )
        self._primitive_restart_enable = CSRStorage(
            1, description="Enable primitive restart"
        )
        self._primitive_restart_index = CSRStorage(
            32, description="Primitive restart index value"
        )
        self._base_vertex = CSRStorage(32, description="Base vertex offset")

        # # #

        gpu_params = {
            "i_clk": ClockSignal(),
            "i_rst": ResetSignal(),
            "o_ready": self._ready.status,
            "i_c_input_topology": self._input_topology.storage,
            "i_c_primitive_restart_enable": self._primitive_restart_enable.storage,
            "i_c_primitive_restart_index": self._primitive_restart_index.storage,
            "i_c_base_vertex": self._base_vertex.storage,
            # Stream input (is_index)
            "i_is_index__valid": self.sink_index.valid,
            "o_is_index__ready": self.sink_index.ready,
            "i_is_index__payload": self.sink_index.payload.data,
            # Stream output (os_index)
            "o_os_index__valid": self.source_index.valid,
            "i_os_index__ready": self.source_index.ready,
            "o_os_index__payload": self.source_index.payload.data,
        }

        self.specials += Instance("InputTopologyProcessor", **gpu_params)


__all__ = ["LiteXInputTopologyProcessor"]
