"""LiteX wrapper for DepthStencilTest."""

from litex.gen import *
from litex.soc.interconnect import stream, wishbone
from litex.soc.interconnect.csr import *
from migen import *

from .stream_defs import fragment_layout


class LiteXDepthStencilTest(LiteXModule, AutoCSR):
    """LiteX wrapper for the DepthStencilTest stage."""

    def __init__(self, platform=None, verilog_path="depthstencil_test.v"):
        # Wishbone master bus for depth/stencil buffer access
        self.wb_bus = wishbone.Interface(data_width=32, adr_width=30)

        # Status
        self._ready = CSRStatus(1, description="Depth/stencil test ready")

        # Stream sink/source (fragments in/out)
        self.sink_fragment = stream.Endpoint(fragment_layout)
        self.source_fragment = stream.Endpoint(fragment_layout)

        # Framebuffer configuration
        self._fb_width = CSRStorage(16, description="Framebuffer width")
        self._fb_height = CSRStorage(16, description="Framebuffer height")
        self._fb_depth_addr = CSRStorage(32, description="Depth buffer base address")
        self._fb_stencil_addr = CSRStorage(
            32, description="Stencil buffer base address"
        )

        # Depth test configuration
        self._depth_test_enable = CSRStorage(1, description="Enable depth testing")
        self._depth_write_enable = CSRStorage(1, description="Enable depth writes")
        self._depth_func = CSRStorage(3, description="Depth test function")

        # Stencil test configuration (front face)
        self._stencil_test_enable = CSRStorage(1, description="Enable stencil testing")
        self._stencil_func_front = CSRStorage(
            3, description="Front stencil test function"
        )
        self._stencil_op_sfail_front = CSRStorage(
            3, description="Front stencil op on fail"
        )
        self._stencil_op_dpfail_front = CSRStorage(
            3, description="Front stencil op on depth fail"
        )
        self._stencil_op_dppass_front = CSRStorage(
            3, description="Front stencil op on depth pass"
        )

        # # #

        gpu_params = {
            "i_clk": ClockSignal(),
            "i_rst": ResetSignal(),
            "o_ready": self._ready.status,
            "i_fb_info": Cat(
                self._fb_width.storage,
                self._fb_height.storage,
                self._fb_depth_addr.storage,
                self._fb_stencil_addr.storage,
            ),
            "i_depth_conf": Cat(
                self._depth_test_enable.storage,
                self._depth_write_enable.storage,
                self._depth_func.storage,
            ),
            "i_stencil_conf_front": Cat(
                self._stencil_test_enable.storage,
                self._stencil_func_front.storage,
                self._stencil_op_sfail_front.storage,
                self._stencil_op_dpfail_front.storage,
                self._stencil_op_dppass_front.storage,
            ),
            "i_stencil_conf_back": 0,  # Not yet configured
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
            # Stream output (os_fragment)
            "o_os_fragment__valid": self.source_fragment.valid,
            "i_os_fragment__ready": self.source_fragment.ready,
            "o_os_fragment__payload": self.source_fragment.payload.data,
        }

        self.specials += Instance("DepthStencilTest", **gpu_params)


__all__ = ["LiteXDepthStencilTest"]
