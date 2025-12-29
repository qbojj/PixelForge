"""LiteX wrapper for VertexShading."""

from litex.gen import *
from litex.soc.interconnect import stream
from litex.soc.interconnect.csr import *
from migen import *

from .stream_defs import vertex_layout


class LiteXVertexShading(LiteXModule, AutoCSR):
    """LiteX wrapper for the VertexShading stage."""

    def __init__(self, platform=None, verilog_path="vertex_shading.v"):
        # Status
        self._ready = CSRStatus(1, description="Vertex shading ready")

        # Stream sink/source (vertices in/out)
        self.sink_vertex = stream.Endpoint(vertex_layout)
        self.source_vertex = stream.Endpoint(vertex_layout)

        # Material properties
        self._ambient_r = CSRStorage(32, description="Material ambient R")
        self._ambient_g = CSRStorage(32, description="Material ambient G")
        self._ambient_b = CSRStorage(32, description="Material ambient B")
        self._diffuse_r = CSRStorage(32, description="Material diffuse R")
        self._diffuse_g = CSRStorage(32, description="Material diffuse G")
        self._diffuse_b = CSRStorage(32, description="Material diffuse B")
        self._specular_r = CSRStorage(32, description="Material specular R")
        self._specular_g = CSRStorage(32, description="Material specular G")
        self._specular_b = CSRStorage(32, description="Material specular B")
        self._shininess = CSRStorage(32, description="Material shininess")

        # # #

        gpu_params = {
            "i_clk": ClockSignal(),
            "i_rst": ResetSignal(),
            "o_ready": self._ready.status,
            "i_material": Cat(
                self._ambient_r.storage,
                self._ambient_g.storage,
                self._ambient_b.storage,
                self._diffuse_r.storage,
                self._diffuse_g.storage,
                self._diffuse_b.storage,
                self._specular_r.storage,
                self._specular_g.storage,
                self._specular_b.storage,
                self._shininess.storage,
            ),
            # Stream input (is_vertex)
            "i_is_vertex__valid": self.sink_vertex.valid,
            "o_is_vertex__ready": self.sink_vertex.ready,
            "i_is_vertex__payload": self.sink_vertex.payload.data,
            # Stream output (os_vertex)
            "o_os_vertex__valid": self.source_vertex.valid,
            "i_os_vertex__ready": self.source_vertex.ready,
            "o_os_vertex__payload": self.source_vertex.payload.data,
        }

        self.specials += Instance("VertexShading", **gpu_params)


__all__ = ["LiteXVertexShading"]
