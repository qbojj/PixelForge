import amaranth_soc.wishbone.bus as wishbone
from amaranth import *
from amaranth.lib import wiring
from amaranth.lib.cdc import PulseSynchronizer

from gpu.output.vga import FrameBufferDMA, VGADisplay


class OutputSystem(wiring.Component):
    """
    VGA + dma
    """

    def __init__(self):
        super().__init__(
            {
                "vga": Out(VGAInterface()),
                "fb_base": In(unsigned(32)),
                "fb_pitch": In(unsigned(32)),
                "wb_bus": Out(
                    wishbone.Signature(
                        addr_width=wb_bus_addr_width,
                        data_width=wb_bus_data_width,
                        granularity=8,
                    )
                ),
            }
        )

    def elaborate(self, platform):
        m = Module()

        m.domain.vga = cd_vga = ClockDomain()
        m.d.sync += cd_vga.eq(~cd_vga)  # vga freq ~ 50MHz / 2
        m.d.comb += cd_vga.rst.eq(ResetSignal())

        # Submodules
        m.submodules.dma = dma = FrameBufferDMA(bus_domain="sync", stream_domain="vga")
        m.submodules.vga = vga = DomainRenamer("vga")(VGADisplay())
        m.d.submodules.ps = ps = PulseSynchronizer("vga", "sync")

        m.d.comb += [
            ps.i.eq(vga.start_fetching),
            dma.start_fetching.eq(ps.o),
        ]

        wiring.connect(m, vga.color_stream, dma.color_stream)

        wiring.connect(m, wiring.flipped(self.vga), vga.interface)
        wiring.connect(m, wiring.flipped(self.wb_bus), dma.wb_bus)
        wiring.connect(m, wiring.flipped(self.fb_base), dma.fb_base)
        wiring.connect(m, wiring.flipped(self.fb_pitch), dma.fb_pitch)

        return m
