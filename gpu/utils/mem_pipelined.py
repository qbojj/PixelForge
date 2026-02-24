import amaranth_soc.wishbone.bus as wb
from amaranth import *
from amaranth.lib import data, fifo, stream, wiring
from amaranth.lib.wiring import In, Out


def mem_read_request_layout(addr_width, data_width, tag_width):
    sel_width = data_width // 8
    return data.StructLayout(
        {
            "addr": unsigned(addr_width),
            "data": unsigned(data_width),
            "sel": unsigned(sel_width),
            "tag": unsigned(tag_width),
        }
    )


def mem_write_request_layout(addr_width, data_width, tag_width):
    sel_width = data_width // 8
    return data.StructLayout(
        {
            "addr": unsigned(addr_width),
            "sel": unsigned(sel_width),
            "tag": unsigned(tag_width),
        }
    )


def mem_read_response_layout(data_width, tag_width):
    return data.StructLayout(
        {
            "data": unsigned(data_width),
            "tag": unsigned(tag_width),
        }
    )


def mem_write_response_layout(tag_width):
    return data.StructLayout({"tag": unsigned(tag_width)})


class WishbonePipelinedMaster(wiring.Component):
    """Pipelined Wishbone master with request/response streams.

    Assumes in-order responses for outstanding requests.
    """

    def __init__(
        self,
        addr_width: int,
        data_width: int,
        tag_width: int = 16,
        read_req_depth: int = 8,
        write_req_depth: int = 8,
        inflight_depth: int = 8,
    ):
        self._addr_width = addr_width
        self._data_width = data_width
        self._tag_width = tag_width
        self._read_req_depth = read_req_depth
        self._write_req_depth = write_req_depth
        self._inflight_depth = inflight_depth

        self._read_req_layout = mem_read_request_layout(
            addr_width, data_width, tag_width
        )
        self._write_req_layout = mem_write_request_layout(
            addr_width, data_width, tag_width
        )

        self._read_resp_layout = mem_read_response_layout(data_width, tag_width)
        self._write_resp_layout = mem_write_response_layout(tag_width)

        super().__init__(
            {
                "read_req": In(stream.Signature(self._read_req_layout)),
                "write_req": In(stream.Signature(self._write_req_layout)),
                "read_resp": Out(stream.Signature(self._read_resp_layout)),
                "write_resp": Out(stream.Signature(self._write_resp_layout)),
                "wb_bus": Out(
                    wb.Signature(
                        addr_width=addr_width,
                        data_width=data_width,
                        features={wb.Feature.STALL},
                    )
                ),
            }
        )

    def elaborate(self, platform):
        m = Module()

        m.submodules.pending_fifo = pending_fifo = fifo.SyncFIFOBuffered(
            width=Shape.cast(self._req_layout).width,
            depth=self._req_depth,
        )
        issued_layout = data.StructLayout(
            {"we": unsigned(1), "tag": unsigned(self._tag_width)}
        )
        m.submodules.issued_fifo = issued_fifo = fifo.SyncFIFO(
            width=Shape.cast(issued_layout).width,
            depth=self._inflight_depth,
        )
        m.submodules.read_resp_fifo = read_resp_fifo = fifo.SyncFIFOBuffered(
            width=Shape.cast(self._read_resp_layout).width,
            depth=self._inflight_depth,
        )
        m.submodules.write_resp_fifo = write_resp_fifo = fifo.SyncFIFOBuffered(
            width=Shape.cast(self._write_resp_layout).width,
            depth=self._inflight_depth,
        )

        req_in = Signal(self._req_layout)
        m.d.comb += req_in.eq(self.req.payload)

        m.d.comb += self.req.ready.eq(pending_fifo.w_rdy)
        m.d.comb += pending_fifo.w_en.eq(self.req.valid & self.req.ready)
        m.d.comb += pending_fifo.w_data.eq(req_in)

        req_out = Signal(self._req_layout)
        m.d.comb += req_out.eq(pending_fifo.r_data)

        issued_out = Signal(issued_layout)
        m.d.comb += issued_out.eq(issued_fifo.r_data)

        # Track the number of inflight/not consumed responses to make sure
        # we will have space in the response FIFOs when we will get the ack.
        pending_reads = Signal(range(2 * self._inflight_depth + 1))
        pending_writes = Signal(range(2 * self._inflight_depth + 1))

        can_fit_next_request = Signal()
        m.d.comb += can_fit_next_request.eq(
            Mux(req_out.we, pending_writes, pending_reads) < self._inflight_depth
        )

        try_issue = pending_fifo.r_rdy & can_fit_next_request
        issue = try_issue & ~getattr(self.wb_bus, "stall", Const(0))

        issued_in = Signal(issued_layout)
        m.d.comb += [
            issued_in.we.eq(req_out.we),
            issued_in.tag.eq(req_out.tag),
            pending_fifo.r_en.eq(issue),
            issued_fifo.w_en.eq(issue),
            issued_fifo.w_data.eq(issued_in),
        ]

        bus_active = try_issue | issued_fifo.r_rdy
        m.d.comb += [
            self.wb_bus.cyc.eq(bus_active),
            self.wb_bus.stb.eq(try_issue),
            self.wb_bus.adr.eq(req_out.addr),
            self.wb_bus.we.eq(req_out.we),
            self.wb_bus.sel.eq(req_out.sel),
            self.wb_bus.dat_w.eq(req_out.data),
        ]

        ack = self.wb_bus.cyc & self.wb_bus.ack

        with m.If(ack):
            m.d.sync += Assert(issued_fifo.r_rdy)

        m.d.comb += issued_fifo.r_en.eq(ack)

        read_ack = ack & ~issued_out.we
        write_ack = ack & issued_out.we

        read_resp_in = Signal(self._read_resp_layout)
        write_resp_in = Signal(self._write_resp_layout)
        m.d.comb += [
            read_resp_in.data.eq(self.wb_bus.dat_r),
            read_resp_in.tag.eq(issued_out.tag),
            write_resp_in.tag.eq(issued_out.tag),
            read_resp_fifo.w_en.eq(read_ack),
            read_resp_fifo.w_data.eq(read_resp_in),
            write_resp_fifo.w_en.eq(write_ack),
            write_resp_fifo.w_data.eq(write_resp_in),
        ]

        with m.If(read_ack):
            m.d.comb += Assert(read_resp_fifo.w_rdy)
        with m.If(write_ack):
            m.d.comb += Assert(write_resp_fifo.w_rdy)

        m.d.comb += [
            self.read_resp.valid.eq(read_resp_fifo.r_rdy),
            self.read_resp.payload.eq(read_resp_fifo.r_data),
            read_resp_fifo.r_en.eq(self.read_resp.valid & self.read_resp.ready),
        ]

        m.d.comb += [
            self.write_resp.valid.eq(write_resp_fifo.r_rdy),
            self.write_resp.payload.eq(write_resp_fifo.r_data),
            write_resp_fifo.r_en.eq(self.write_resp.valid & self.write_resp.ready),
        ]

        m.d.sync += pending_reads.eq(
            pending_reads + (issue & ~req_out.we) - read_resp_fifo.r_en
        )
        m.d.sync += pending_writes.eq(
            pending_writes + (issue & req_out.we) - write_resp_fifo.r_en
        )

        return m
