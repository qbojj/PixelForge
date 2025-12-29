#!/usr/bin/env python3

#
# LiteX SoC with GPU on Terasic DE1-SoC.
#
# Copyright (c) 2025
# SPDX-License-Identifier: BSD-2-Clause

from litedram.modules import IS42S16320
from litedram.phy import GENSDRPHY
from litex.build.io import DDROutput
from litex.gen import *
from litex.soc.cores.clock import CycloneVPLL
from litex.soc.cores.video import VideoVGAPHY, video_timings
from litex.soc.integration.builder import *
from litex.soc.integration.soc_core import *
from migen import *

# Import GPU module
from gpu.litex import LiteXGPU

# Use custom platform with corrected VGA pins
from gpu.litex.platforms import de1soc_platform

# CRG ----------------------------------------------------------------------------------------------


class _CRG(LiteXModule):
    def __init__(
        self, platform, sys_clk_freq, with_vga=False, vga_timings="640x480@60Hz"
    ):
        self.rst = Signal()
        self.cd_sys = ClockDomain("sys")
        self.cd_sys_ps = ClockDomain("sys_ps")

        if with_vga:
            self.cd_vga = ClockDomain("vga")

        if isinstance(vga_timings, str):
            vga_timings = video_timings[vga_timings]

        # # #

        # Clk / Rst
        clk50 = platform.request("clk50")

        # PLL
        self.pll = pll = CycloneVPLL(speedgrade="-C6")
        self.comb += pll.reset.eq(self.rst)
        pll.register_clkin(clk50, 50e6)
        pll.create_clkout(self.cd_sys, sys_clk_freq)
        pll.create_clkout(self.cd_sys_ps, sys_clk_freq, phase=90)
        if with_vga:
            pll.create_clkout(self.cd_vga, vga_timings["pix_clk"])

        # SDRAM clock
        self.specials += DDROutput(
            1, 0, platform.request("sdram_clock"), ClockSignal("sys_ps")
        )


# BaseSoC ------------------------------------------------------------------------------------------


class BaseSoC(SoCCore):
    def __init__(
        self,
        sys_clk_freq=50e6,
        with_vga=False,
        with_gpu=False,
        vga_timings="640x480@60Hz",
        **kwargs,
    ):
        platform = de1soc_platform.Platform()

        # CRG --------------------------------------------------------------------------------------
        self.crg = _CRG(
            platform,
            sys_clk_freq,
            with_vga=with_vga or with_gpu,
            vga_timings=vga_timings,
        )

        # SoCCore ----------------------------------------------------------------------------------
        SoCCore.__init__(
            self,
            platform,
            sys_clk_freq,
            ident="LiteX SoC on DE1-SoC with GPU",
            **kwargs,
        )

        # SDR SDRAM --------------------------------------------------------------------------------
        if not self.integrated_main_ram_size:
            self.sdrphy = GENSDRPHY(platform.request("sdram"), sys_clk_freq)
            self.add_sdram(
                "sdram",
                phy=self.sdrphy,
                module=IS42S16320(sys_clk_freq, "1:1"),
                l2_cache_size=kwargs.get("l2_size", 8192),
            )

        # GPU --------------------------------------------------------------------------------------
        if with_gpu:
            # Add GPU module
            self.submodules.gpu = LiteXGPU(platform)

            # Register GPU Verilog sources with the platform
            platform.add_source_dir("/home/kuba/Desktop/FPGA/praca/build/gpu_verilog")

            # Connect GPU Wishbone buses to main bus as masters
            self.bus.add_master(name="gpu_index", master=self.gpu.bus_index)
            self.bus.add_master(name="gpu_vertex", master=self.gpu.bus_vertex)
            self.bus.add_master(
                name="gpu_depthstencil", master=self.gpu.bus_depthstencil
            )
            self.bus.add_master(name="gpu_color", master=self.gpu.bus_color)

        # VGA Terminal -----------------------------------------------------------------------------
        if with_vga:
            self.videophy = VideoVGAPHY(platform.request("vga"), clock_domain="vga")
            self.add_video_terminal(
                phy=self.videophy, timings=vga_timings, clock_domain="vga"
            )


# Build --------------------------------------------------------------------------------------------


def main():
    from litex.build.parser import LiteXArgumentParser

    parser = LiteXArgumentParser(
        platform=de1soc_platform.Platform,
        description="LiteX SoC on DE1-SoC with GPU support.",
    )
    parser.add_target_argument(
        "--sys-clk-freq", default=50e6, type=float, help="System clock frequency."
    )
    parser.add_target_argument(
        "--with-vga", action="store_true", help="Enable VGA terminal."
    )
    parser.add_target_argument(
        "--with-gpu", action="store_true", help="Enable GPU (3D graphics pipeline)."
    )
    args = parser.parse_args()

    soc = BaseSoC(
        sys_clk_freq=args.sys_clk_freq,
        with_vga=args.with_vga,
        with_gpu=args.with_gpu,
        **parser.soc_argdict,
    )
    builder = Builder(soc, **parser.builder_argdict)
    if args.build:
        builder.build(**parser.toolchain_argdict)

    if args.load:
        prog = soc.platform.create_programmer()
        prog.load_bitstream(builder.get_bitstream_filename(mode="sram"))


if __name__ == "__main__":
    main()
