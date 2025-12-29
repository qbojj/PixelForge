#!/usr/bin/env python3

from litex.build.generic_platform import IOStandard, Pins, Subsignal
from litex_boards.platforms import terasic_de1soc
from migen import *

# Corrected VGA resource for DE1-SoC
_vga_io = [
    (
        "vga",
        0,
        Subsignal("r", Pins("A13 C13 E13 B12 C12 D12 E12 F13")),
        Subsignal("g", Pins("J9  J10 H12 G10 G11 G12 F11 E11")),
        Subsignal("b", Pins("B13 G13 H13 F14 H14 F15 G15 J14")),
        Subsignal("clk", Pins("A11")),
        Subsignal("be", Pins("F10")),
        Subsignal("sync_n", Pins("C10")),
        Subsignal("hsync_n", Pins("B11")),
        Subsignal("vsync_n", Pins("D11")),
        IOStandard("3.3-V LVTTL"),
    ),
]


class Platform(terasic_de1soc.Platform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace VGA resource with corrected version
        self.add_extension(_vga_io)
