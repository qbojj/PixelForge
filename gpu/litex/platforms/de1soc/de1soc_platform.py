#!/usr/bin/env python3

from litex.build.generic_platform import Inverted, IOStandard, Pins, Subsignal
from litex_boards.platforms import terasic_de1soc

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
        Subsignal("sync", Pins("C10"), Inverted()),
        Subsignal("hsync", Pins("B11"), Inverted()),
        Subsignal("vsync", Pins("D11"), Inverted()),
        IOStandard("3.3-V LVTTL"),
    ),
]

# HPS DDR3 resource with board pinout (DE1-SoC user manual)
_hps_ddr3_io = [
    (
        "hps_ddr3",
        0,
        # Control signals
        Subsignal(
            "a",
            Pins("F26 G30 F28 F30 J25 J27 F29 E28 H27 G26 D29 C30 B30 C29 H25"),
        ),  # Address[14:0]
        Subsignal("ba", Pins("E29 J24 J23")),  # Bank address[2:0]
        Subsignal("cas_n", Pins("E27")),
        Subsignal("cke", Pins("L29")),
        Subsignal("ck_n", Pins("L23")),
        Subsignal("ck_p", Pins("M23")),
        Subsignal("cs_n", Pins("H24")),
        Subsignal("dm", Pins("K28 M28 R28 W30")),  # Data mask[3:0]
        # Bidirectional data signals
        Subsignal(
            "dq",
            Pins(
                "K23 K22 H30 G28 L25 L24 J30 J29 "
                "K26 L26 K29 K27 M26 M27 L28 M30 "
                "U26 T26 N29 N28 P26 P27 N27 R29 "
                "P24 P25 T29 T28 R27 R26 V30 W29"
            ),
        ),  # Data[31:0]
        Subsignal("dqs_n", Pins("M19 N24 R18 R21")),  # Data strobe n[3:0]
        Subsignal("dqs_p", Pins("N18 N25 R19 R22")),  # Data strobe p[3:0]
        # Other controls
        Subsignal("odt", Pins("H28")),
        Subsignal("ras_n", Pins("D30")),
        Subsignal("reset_n", Pins("P30")),
        Subsignal("we_n", Pins("C28")),
        Subsignal("rzq", Pins("D27")),
        IOStandard("SSTL15"),
    ),
]


class Platform(terasic_de1soc.Platform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace VGA resource with corrected version
        self.add_extension(_vga_io)
        # Add HPS DDR3 resource
        self.add_extension(_hps_ddr3_io)
