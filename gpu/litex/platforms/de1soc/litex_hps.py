"""LiteX HPS Bridge Integration for DE1-SoC.

Instantiates the Altera HPS IP and exposes the full and lightweight AXI
masters plus DDR3 pads, reset, and IRQ lines for LiteX integration.
"""

from litex.gen import *
from litex.soc.interconnect import axi
from migen import *


def quartus_axi3_params(prefix, axi_interface, dir="master"):
    """Generates a dictionary of Verilog instance parameters for an AXI3 interface."""
    params = {}

    o = "o" if dir == "master" else "i"
    i = "i" if dir == "master" else "o"

    # Write address channel
    params[f"{o}_{prefix}_awid"] = axi_interface.aw.id
    params[f"{o}_{prefix}_awaddr"] = axi_interface.aw.addr
    params[f"{o}_{prefix}_awlen"] = axi_interface.aw.len
    params[f"{o}_{prefix}_awsize"] = axi_interface.aw.size
    params[f"{o}_{prefix}_awburst"] = axi_interface.aw.burst
    params[f"{o}_{prefix}_awlock"] = axi_interface.aw.lock
    params[f"{o}_{prefix}_awcache"] = axi_interface.aw.cache
    params[f"{o}_{prefix}_awprot"] = axi_interface.aw.prot
    params[f"{o}_{prefix}_awvalid"] = axi_interface.aw.valid
    params[f"{i}_{prefix}_awready"] = axi_interface.aw.ready

    # Write data channel
    params[f"{o}_{prefix}_wid"] = axi_interface.w.id
    params[f"{o}_{prefix}_wdata"] = axi_interface.w.data
    params[f"{o}_{prefix}_wstrb"] = axi_interface.w.strb
    params[f"{o}_{prefix}_wlast"] = axi_interface.w.last
    params[f"{o}_{prefix}_wvalid"] = axi_interface.w.valid
    params[f"{i}_{prefix}_wready"] = axi_interface.w.ready

    # Write response channel
    params[f"{i}_{prefix}_bid"] = axi_interface.b.id
    params[f"{i}_{prefix}_bresp"] = axi_interface.b.resp
    params[f"{i}_{prefix}_bvalid"] = axi_interface.b.valid
    params[f"{o}_{prefix}_bready"] = axi_interface.b.ready

    # Read address channel
    params[f"{o}_{prefix}_arid"] = axi_interface.ar.id
    params[f"{o}_{prefix}_araddr"] = axi_interface.ar.addr
    params[f"{o}_{prefix}_arlen"] = axi_interface.ar.len
    params[f"{o}_{prefix}_arsize"] = axi_interface.ar.size
    params[f"{o}_{prefix}_arburst"] = axi_interface.ar.burst
    params[f"{o}_{prefix}_arlock"] = axi_interface.ar.lock
    params[f"{o}_{prefix}_arcache"] = axi_interface.ar.cache
    params[f"{o}_{prefix}_arprot"] = axi_interface.ar.prot
    params[f"{o}_{prefix}_arvalid"] = axi_interface.ar.valid
    params[f"{i}_{prefix}_arready"] = axi_interface.ar.ready

    # Read data channel
    params[f"{i}_{prefix}_rid"] = axi_interface.r.id
    params[f"{i}_{prefix}_rdata"] = axi_interface.r.data
    params[f"{i}_{prefix}_rresp"] = axi_interface.r.resp
    params[f"{i}_{prefix}_rlast"] = axi_interface.r.last
    params[f"{i}_{prefix}_rvalid"] = axi_interface.r.valid
    params[f"{o}_{prefix}_rready"] = axi_interface.r.ready

    return params


class LiteXHPSBridge(LiteXModule):
    """Integrates the Altera HPS (Hard Processor System) with LiteX."""

    def __init__(self, ddr3_pads):
        # HPS output: reset (active low)
        self.hps_rst_n = Signal()

        # HPS inputs: interrupt requests (32-bit vectors)
        self.irq0 = Signal(32)
        self.irq1 = Signal(32)

        # Full AXI master (HPS as master)
        self.axi = axi.AXIInterface(
            data_width=32,
            address_width=30,
            id_width=12,
            version="axi3",
        )

        # Lightweight AXI master (HPS as master)
        self.axi_l = axi.AXIInterface(
            data_width=32,
            address_width=21,
            id_width=12,
            version="axi3",
        )

        self.axi_sdram = axi.AXIInterface(
            data_width=64,
            address_width=30,
            id_width=12,
            version="axi3",
        )

        hps_params = {
            "o_h2f_rst_reset_n": self.hps_rst_n,
            # Clocks
            "i_h2f_axi_clk_clk": ClockSignal(self.axi.clock_domain),
            "i_h2f_axi_l_clk_clk": ClockSignal(self.axi_l.clock_domain),
            "i_hps_0_f2h_sdram0_clock_clk": ClockSignal(self.axi_sdram.clock_domain),
            # Interrupt inputs
            "i_irq0_irq": self.irq0,
            "i_irq1_irq": self.irq1,
            # AXI master
            **quartus_axi3_params("h2f_axi_master", self.axi, dir="master"),
            # Lightweight AXI master
            **quartus_axi3_params("h2f_axi_l_master", self.axi_l, dir="master"),
            # f2h_sdram0 slave interface
            **quartus_axi3_params("f2h_sdram0_slave", self.axi_sdram, dir="slave"),
            # DDR3 pads
            "o_memory_mem_a": ddr3_pads.a,
            "o_memory_mem_ba": ddr3_pads.ba,
            "o_memory_mem_ck": ddr3_pads.ck_p,
            "o_memory_mem_ck_n": ddr3_pads.ck_n,
            "o_memory_mem_cke": ddr3_pads.cke,
            "o_memory_mem_cs_n": ddr3_pads.cs_n,
            "o_memory_mem_ras_n": ddr3_pads.ras_n,
            "o_memory_mem_cas_n": ddr3_pads.cas_n,
            "o_memory_mem_we_n": ddr3_pads.we_n,
            "o_memory_mem_reset_n": ddr3_pads.reset_n,
            "o_memory_mem_odt": ddr3_pads.odt,
            "o_memory_mem_dm": ddr3_pads.dm,
            "io_memory_mem_dq": ddr3_pads.dq,
            "io_memory_mem_dqs": ddr3_pads.dqs_p,
            "io_memory_mem_dqs_n": ddr3_pads.dqs_n,
            "i_memory_oct_rzqin": ddr3_pads.rzq,
        }

        self.specials += Instance("HPS", **hps_params)


__all__ = ["LiteXHPSBridge"]
