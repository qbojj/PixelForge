"""LiteX wrapper for SwapchainOutput."""

from litex.gen import *
from litex.soc.interconnect import stream, wishbone
from litex.soc.interconnect.csr import *
from migen import *

from .stream_defs import fragment_layout


class LiteXSwapchainOutput(LiteXModule, AutoCSR):
    """LiteX wrapper for the SwapchainOutput stage."""

    def __init__(self, platform=None, verilog_path="swapchain_output.v"):
        # Wishbone master bus for color buffer writes
        self.wb_bus = wishbone.Interface(data_width=32, adr_width=30)

        # Status
        self._ready = CSRStatus(1, description="Swapchain output ready")

        # Stream sink (fragments)
        self.sink_fragment = stream.Endpoint(fragment_layout)

        # Framebuffer configuration
        self._fb_width = CSRStorage(16, description="Framebuffer width")
        self._fb_height = CSRStorage(16, description="Framebuffer height")
        self._fb_color_addr = CSRStorage(32, description="Color buffer base address")

        # Blend configuration
        self._conf = CSRStorage(
            12,
            fields=[
                CSRField("blend_enable", size=1, description="Enable blending"),
                CSRField("blend_src_factor", size=4, description="Source blend factor"),
                CSRField(
                    "blend_dst_factor", size=4, description="Destination blend factor"
                ),
                CSRField("blend_op", size=3, description="Blend operation"),
            ],
        )

        # # #

        gpu_params = {
            "i_clk": ClockSignal(),
            "i_rst": ResetSignal(),
            "o_ready": self._ready.status,
            "i_fb_info": Cat(
                self._fb_width.storage,
                self._fb_height.storage,
                self._fb_color_addr.storage,
            ),
            "i_conf": self._conf.storage,
            # Wishbone connections
            "o_wb_bus__adr": self.wb_bus.adr,
            "i_wb_bus__dat_r": self.wb_bus.dat_r,
            "o_wb_bus__dat_w": self.wb_bus.dat_w,
            "o_wb_bus__sel": self.wb_bus.sel,
            "o_wb_bus__cyc": self.wb_bus.cyc,
            "o_wb_bus__stb": self.wb_bus.stb,
            "o_wb_bus__we": self.wb_bus.we,
            "i_wb_bus__ack": self.wb_bus.ack,
            # Stream input (is_fragment)
            "i_is_fragment__valid": self.sink_fragment.valid,
            "o_is_fragment__ready": self.sink_fragment.ready,
            "i_is_fragment__payload": self.sink_fragment.payload.data,
        }

        self.specials += Instance("SwapchainOutput", **gpu_params)


__all__ = ["LiteXSwapchainOutput"]
