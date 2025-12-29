"""LiteX GPU composed from modular pipeline stages.

This module composes the GPU from small LiteX wrapper modules instead of
instantiating a single monolithic Verilog "top". Each stage exposes its own
CSRs and (when applicable) Wishbone master interface. The top-level module
exposes the four main Wishbone masters used by the pipeline.
"""

import os

from litex.gen import LiteXModule
from litex.soc.integration.doc import ModuleDoc
from litex.soc.interconnect.csr import *
from migen import *

from .litex_clipper import LiteXPrimitiveClipper
from .litex_depthstencil import LiteXDepthStencilTest

# Pipeline stage wrappers
from .litex_index_generator import LiteXIndexGenerator
from .litex_input_assembly import LiteXInputAssembly
from .litex_input_topology import LiteXInputTopologyProcessor
from .litex_primitive_assembly import LiteXPrimitiveAssembly
from .litex_rasterizer import LiteXTriangleRasterizer
from .litex_swapchain import LiteXSwapchainOutput
from .litex_texturing import LiteXTexturing
from .litex_vertex_shading import LiteXVertexShading
from .litex_vertex_transform import LiteXVertexTransform


class LiteXGPU(LiteXModule):
    """LiteX-composed GPU using modular pipeline wrappers.

    Exposes four Wishbone master interfaces:
    - bus_index: For reading vertex indices (IndexGenerator)
    - bus_vertex: For reading vertex data (InputAssembly)
    - bus_depthstencil: For depth/stencil buffer access (DepthStencilTest)
    - bus_color: For color framebuffer writes (SwapchainOutput)
    """

    def __init__(self, platform=None):
        self.intro = ModuleDoc("Modular 3D Graphics Pipeline composed of LiteX stages.")

        # Submodules (each with its own CSRs)
        self.submodules.idx = LiteXIndexGenerator(platform)
        self.submodules.topo = LiteXInputTopologyProcessor(platform)
        self.submodules.ia = LiteXInputAssembly(platform)
        self.submodules.vtxxf = LiteXVertexTransform(platform)
        self.submodules.vtxsh = LiteXVertexShading(platform)
        self.submodules.pa = LiteXPrimitiveAssembly(platform)
        self.submodules.clip = LiteXPrimitiveClipper(platform)
        self.submodules.rast = LiteXTriangleRasterizer(platform)
        self.submodules.tex = LiteXTexturing(platform)
        self.submodules.ds = LiteXDepthStencilTest(platform)
        self.submodules.sc = LiteXSwapchainOutput(platform)

        # Expose Wishbone masters by aliasing submodule interfaces
        self.bus_index = self.idx.bus
        self.bus_vertex = self.ia.bus
        self.bus_depthstencil = self.ds.wb_bus
        self.bus_color = self.sc.wb_bus

        # Aggregate pipeline readiness status
        self._ready = CSRStatus(1, description="Pipeline composed stages ready")
        self.comb += self._ready.status.eq(
            self.idx._ready.status
            & self.topo._ready.status
            & self.ia._ready.status
            & self.vtxxf._ready.status
            & self.vtxsh._ready.status
            & self.pa._ready.status
            & self.clip._ready.status
            & self.rast._ready.status
            & self.tex._ready.status
            & self.ds._ready.status
            & self.sc._ready.status
        )

        # Stream wiring between stages using Endpoint.connect()
        self.comb += self.idx.source_index.connect(self.topo.sink_index)
        self.comb += self.topo.source_index.connect(self.ia.sink_index)
        self.comb += self.ia.source_vertex.connect(self.vtxxf.sink_vertex)
        self.comb += self.vtxxf.source_vertex.connect(self.vtxsh.sink_vertex)
        self.comb += self.vtxsh.source_vertex.connect(self.pa.sink_vertex)
        self.comb += self.pa.source_primitive.connect(self.clip.sink_vertex)
        self.comb += self.clip.source_vertex.connect(self.rast.sink_vertex)
        self.comb += self.rast.source_fragment.connect(self.tex.sink_fragment)
        self.comb += self.tex.source_fragment.connect(self.ds.sink_fragment)
        self.comb += self.ds.source_fragment.connect(self.sc.sink_fragment)


def generate_verilog(output_path="graphics_pipeline.v"):
    """Deprecated: use gpu.litex.build.generate_all_verilog instead."""
    from .build import generate_all_verilog

    generated = generate_all_verilog(
        os.path.dirname(output_path) or "build/gpu_verilog"
    )
    return generated


__all__ = ["LiteXGPU", "generate_verilog"]

if __name__ == "__main__":
    generate_verilog()
