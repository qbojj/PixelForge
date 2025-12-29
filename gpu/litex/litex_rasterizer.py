"""LiteX wrapper for TriangleRasterizer."""

from litex.gen import *
from litex.soc.interconnect import stream
from litex.soc.interconnect.csr import *
from migen import *

from .stream_defs import fragment_layout, vertex_layout


class LiteXTriangleRasterizer(LiteXModule, AutoCSR):
    """LiteX wrapper for the TriangleRasterizer stage."""

    def __init__(self, platform=None, verilog_path="triangle_rasterizer.v"):
        # Status
        self._ready = CSRStatus(1, description="Rasterizer ready")

        # Stream sink/source (vertices → fragments)
        self.sink_vertex = stream.Endpoint(vertex_layout)
        self.source_fragment = stream.Endpoint(fragment_layout)

        # Framebuffer configuration
        self._fb_width = CSRStorage(16, description="Framebuffer width")
        self._fb_height = CSRStorage(16, description="Framebuffer height")
        self._fb_color_addr = CSRStorage(32, description="Color buffer base address")
        self._fb_depth_addr = CSRStorage(32, description="Depth buffer base address")

        # Viewport configuration
        self._viewport_x = CSRStorage(16, description="Viewport X offset")
        self._viewport_y = CSRStorage(16, description="Viewport Y offset")
        self._viewport_width = CSRStorage(16, description="Viewport width")
        self._viewport_height = CSRStorage(16, description="Viewport height")

        # # #

        gpu_params = {
            "i_clk": ClockSignal(),
            "i_rst": ResetSignal(),
            "o_ready": self._ready.status,
            "i_fb_info": Cat(
                self._fb_width.storage,
                self._fb_height.storage,
                self._fb_color_addr.storage,
                self._fb_depth_addr.storage,
            ),
            # Stream input (is_vertex)
            "i_is_vertex__valid": self.sink_vertex.valid,
            "o_is_vertex__ready": self.sink_vertex.ready,
            "i_is_vertex__payload": self.sink_vertex.payload.data,
            # Stream output (os_fragment)
            "o_os_fragment__valid": self.source_fragment.valid,
            "i_os_fragment__ready": self.source_fragment.ready,
            "o_os_fragment__payload": self.source_fragment.payload.data,
        }

        self.specials += Instance("TriangleRasterizer", **gpu_params)


__all__ = ["LiteXTriangleRasterizer"]
