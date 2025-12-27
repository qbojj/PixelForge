import amaranth_soc.wishbone.bus as wb
from amaranth import *
from amaranth.lib import data, fifo, stream, wiring
from amaranth.lib.wiring import In, Out

from gpu.utils.layouts import wb_bus_addr_width, wb_bus_data_width

# 640x480@60 timings (25.175 MHz nominal pixel clock)
H_VISIBLE = 640
H_FRONT = 16
H_SYNC = 96
H_BACK = 48
H_TOTAL = H_VISIBLE + H_FRONT + H_SYNC + H_BACK

V_VISIBLE = 480
V_FRONT = 10
V_SYNC = 2
V_BACK = 33
V_TOTAL = V_VISIBLE + V_FRONT + V_SYNC + V_BACK


class VGASignature(wiring.Signature):
    """VGA output signals signature"""

    h_sync: Out(1)
    v_sync: Out(1)
    r: Out(8)
    g: Out(8)
    b: Out(8)
    blank: Out(1)
    clk: Out(1)


class VGAInterface(wiring.PureInterface):
    def __init__(self):
        super().__init__(VGASignature())


class RGBColorLayout(data.Struct):
    r: unsigned(8)
    g: unsigned(8)
    b: unsigned(8)


class FrameBufferDMA(wiring.Component):
    """Framebuffer DMA engine for VGA output.

    Issues read cycles over a Wishbone bus to fetch pixel data for VGA
    output. The module assumes one 32-bit pixel per address beat and that
    framebuffer base and pitch are provided in Wishbone address units
    (addr_width refers to bus words, not bytes).
    """

    fb_base: Value
    fb_pitch: Value
    wb_bus: wb.Interface
    start_fetching: Value
    color_stream: stream.Interface
    ready: Value

    width: Value
    height: Value

    def __init__(self, bus_domain="sync", stream_domain="sync"):
        super().__init__(
            {
                "fb_base": In(unsigned(wb_bus_addr_width)),
                "fb_pitch": In(unsigned(wb_bus_addr_width)),
                "wb_bus": Out(
                    wb.Signature(
                        addr_width=wb_bus_addr_width,
                        data_width=wb_bus_data_width,
                        granularity=8,
                    )
                ),
                "start_fetching": In(1),
                "color_stream": Out(stream.Signature(RGBColorLayout)),
                "ready": Out(1),
                "width": In(unsigned(12)),
                "height": In(unsigned(12)),
            }
        )
        self._bus_domain = bus_domain
        self._stream_domain = stream_domain

    def elaborate(self, platform):
        m = Module()

        fifo_soft_reset = Signal()

        m.domains.fifo_bus_domain = fifo_bus_domain = ClockDomain(local=True)
        fifo_bus_domain.clk = ClockSignal(self._bus_domain)
        fifo_bus_domain.rst = ResetSignal(self._bus_domain) | fifo_soft_reset

        data_depth = H_VISIBLE * 2  # enough for 2 lines of pixels

        # Create FIFO with appropriate clock domain crossing
        fifo_width = RGBColorLayout.as_shape()
        if self._bus_domain == self._stream_domain:
            m.submodules.color_fifo = color_fifo = DomainRenamer(fifo_bus_domain)(
                fifo.SyncFIFOBuffered(
                    width=fifo_width,
                    depth=data_depth,
                )
            )
        else:
            m.submodules.color_fifo = color_fifo = fifo.AsyncFIFOBuffered(
                width=fifo_width,
                depth=data_depth,
                write_domain=fifo_bus_domain,
                read_domain=self._stream_domain,
            )

        # DMA state signals
        x = Signal(unsigned(12))  # max 4k
        y = Signal(unsigned(12))
        x_end = Signal()
        y_end = Signal()

        # Extract RGB from 32-bit word and pack into FIFO
        rgb_data = Signal(fifo_width)
        m.d.comb += [
            rgb_data[0:8].eq(self.wb_bus.dat_r[0:8]),
            rgb_data[8:16].eq(self.wb_bus.dat_r[8:16]),
            rgb_data[16:24].eq(self.wb_bus.dat_r[16:24]),
            color_fifo.w_data.eq(rgb_data),
        ]

        # Combinational computation of end conditions
        m.d.comb += [
            x_end.eq(x == self.width - 1),
            y_end.eq(y == self.height - 1),
        ]

        addr_x0 = Signal(wb_bus_addr_width)

        # Bus domain: DMA fetching
        with m.FSM(domain=self._bus_domain):
            with m.State("IDLE"):
                m.d.comb += self.ready.eq(1)

                m.d[self._bus_domain] += [
                    self.wb_bus.cyc.eq(0),
                    self.wb_bus.stb.eq(0),
                ]
                with m.If(self.start_fetching):
                    m.next = "INIT"
            with m.State("INIT"):
                m.d.comb += fifo_soft_reset.eq(1)  # clear FIFO data
                m.d[self._bus_domain] += [
                    x.eq(0),
                    y.eq(0),
                    addr_x0.eq(self.fb_base),
                ]
                m.next = "FETCH_PIXEL"
            with m.State("FETCH_PIXEL"):
                with m.If(color_fifo.w_rdy):
                    m.d[self._bus_domain] += [
                        self.wb_bus.cyc.eq(1),
                        self.wb_bus.stb.eq(1),
                        self.wb_bus.adr.eq(addr_x0 + x),
                        self.wb_bus.sel.eq(0b0111),
                    ]
                    m.next = "WAIT_ACK"
            with m.State("WAIT_ACK"):
                with m.If(self.wb_bus.ack):
                    m.d.comb += color_fifo.w_en.eq(1)

                    m.d[self._bus_domain] += [
                        self.wb_bus.cyc.eq(0),
                        self.wb_bus.stb.eq(0),
                    ]

                    with m.If(x_end & y_end):
                        m.next = "IDLE"
                    with m.Elif(x_end):
                        m.d[self._bus_domain] += [
                            x.eq(0),
                            y.eq(y + 1),
                            addr_x0.eq(addr_x0 + self.fb_pitch),
                        ]
                        m.next = "FETCH_PIXEL"
                    with m.Else():
                        m.d[self._bus_domain] += x.eq(x + 1)
                        m.next = "FETCH_PIXEL"

        # Stream domain: output stream delivery
        m.d.comb += [
            color_fifo.r_en.eq(self.color_stream.ready),
            self.color_stream.p.eq(color_fifo.r_data),
            self.color_stream.valid.eq(color_fifo.r_rdy),
        ]

        return m


class VGADisplay(wiring.Component):
    """VGA timing generator that produces timing signals and pixel fetch strobes.

    Generates VGA sync/blank signals and strobes `start_fetching` at the beginning
    of each visible line. Consumes pixels from an external `color_stream` interface
    during the active area.
    """

    vga: VGAInterface

    start_fetching: Value
    color_stream: stream.Interface

    def __init__(self):
        super().__init__(
            {
                "vga": Out(VGASignature()),
                "start_fetching": Out(1),
                "color_stream": In(stream.Signature(RGBColorLayout)),
                "ok_to_vsync": Out(1),
            }
        )

    def elaborate(self, platform):
        m = Module()

        # Timing counters
        x_cnt = Signal(range(H_TOTAL))
        y_cnt = Signal(range(V_TOTAL))

        x_end = Signal()
        y_end = Signal()

        missed_pixel = Signal()

        m.d.comb += [
            x_end.eq(x_cnt == H_TOTAL - 1),
            y_end.eq(y_cnt == V_TOTAL - 1),
        ]

        # Horizontal/vertical counters
        with m.If(x_end & y_end):
            m.d.sync += x_cnt.eq(0)
            m.d.sync += y_cnt.eq(0)
            m.d.sync += [
                Assert(~missed_pixel, "Missed pixel during frame!"),
                missed_pixel.eq(0),
            ]
        with m.Elif(x_end):
            m.d.sync += x_cnt.eq(0)
            m.d.sync += y_cnt.eq(y_cnt + 1)
        with m.Else():
            m.d.sync += x_cnt.eq(x_cnt + 1)

        # Sync pulses (active low)
        m.d.sync += self.vga.h_sync.eq(
            ~((x_cnt >= H_VISIBLE + H_FRONT) & (x_cnt < H_VISIBLE + H_FRONT + H_SYNC))
        )
        m.d.sync += self.vga.v_sync.eq(
            ~((y_cnt >= V_VISIBLE + V_FRONT) & (y_cnt < V_VISIBLE + V_FRONT + V_SYNC))
        )
        m.d.comb += self.vga.clk.eq(ClockSignal())
        m.d.sync += self.vga.blank.eq((x_cnt < H_VISIBLE) & (y_cnt < V_VISIBLE))

        with m.If(self.vga.blank):
            # During visible area, output pixel data from stream
            with m.If(self.color_stream.valid & self.color_stream.ready):
                m.d.sync += [
                    self.vga.r.eq(self.color_stream.p.r),
                    self.vga.g.eq(self.color_stream.p.g),
                    self.vga.b.eq(self.color_stream.p.b),
                ]
            with m.Else():
                # No pixel data available, output black
                m.d.sync += [
                    self.vga.r.eq(0),
                    self.vga.g.eq(0),
                    self.vga.b.eq(0),
                ]
                m.d.sync += missed_pixel.eq(1)

        # start fetching data right after v_sync
        m.d.comb += self.start_fetching.eq(
            (x_cnt == 0) & (y_cnt == V_VISIBLE + V_FRONT + V_SYNC)
        )

        return m
