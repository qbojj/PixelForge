import amaranth_soc.wishbone.bus as wb
from amaranth import *
from amaranth.lib import data, fifo, stream, wiring
from amaranth.lib.wiring import In, Out
from amaranth.utils import log2_int

from .layouts import axi_addr_width, axi_data_width, axi_id_width

__all__ = [
    "Signature",
    "Interface",
    "AxiPipelinedMaster",
    "mem_read_request_layout",
    "mem_write_request_layout",
    "mem_read_response_layout",
    "mem_write_response_layout",
]


def mem_read_request_layout(addr_width, data_width, tag_width):
    sel_width = data_width // 8
    return data.StructLayout(
        {
            "addr": unsigned(addr_width),
            "sel": unsigned(sel_width),
            "tag": unsigned(tag_width),
        }
    )


def mem_write_request_layout(addr_width, data_width, tag_width):
    sel_width = data_width // 8
    return data.StructLayout(
        {
            "addr": unsigned(addr_width),
            "data": unsigned(data_width),
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


class Signature(wiring.Signature):
    """AXI4 master-facing signature (single-beat only)."""

    def __init__(
        self,
        *,
        addr_width: int,
        data_width: int,
        id_width: int,
    ) -> None:
        if not isinstance(addr_width, int) or addr_width <= 0:
            raise TypeError(
                f"Address width must be a positive integer, not {addr_width!r}"
            )
        if not isinstance(data_width, int) or data_width <= 0:
            raise TypeError(
                f"Data width must be a positive integer, not {data_width!r}"
            )
        if data_width % 8 != 0:
            raise ValueError("AXI data_width must be a multiple of 8")
        if not isinstance(id_width, int) or id_width <= 0:
            raise TypeError(f"ID width must be a positive integer, not {id_width!r}")

        self._addr_width = addr_width
        self._data_width = data_width
        self._id_width = id_width

        strb_width = data_width // 8

        members: dict[str, wiring.Direction] = {
            # Write address channel
            "aw_valid": Out(1),
            "aw_ready": In(1),
            "aw_addr": Out(unsigned(addr_width)),
            "aw_id": Out(unsigned(id_width)),
            "aw_len": Out(unsigned(8)),
            "aw_size": Out(unsigned(3)),
            "aw_burst": Out(unsigned(2)),
            # Write data channel
            "w_valid": Out(1),
            "w_ready": In(1),
            "w_data": Out(unsigned(data_width)),
            "w_strb": Out(unsigned(strb_width)),
            "w_last": Out(1),
            # Write response channel
            "b_valid": In(1),
            "b_ready": Out(1),
            "b_id": In(unsigned(id_width)),
            "b_resp": In(unsigned(2)),
            # Read address channel
            "ar_valid": Out(1),
            "ar_ready": In(1),
            "ar_addr": Out(unsigned(addr_width)),
            "ar_id": Out(unsigned(id_width)),
            "ar_len": Out(unsigned(8)),
            "ar_size": Out(unsigned(3)),
            "ar_burst": Out(unsigned(2)),
            # Read data channel
            "r_valid": In(1),
            "r_ready": Out(1),
            "r_id": In(unsigned(id_width)),
            "r_data": In(unsigned(data_width)),
            "r_resp": In(unsigned(2)),
            "r_last": In(1),
        }

        super().__init__(members)

    @property
    def addr_width(self) -> int:
        return self._addr_width

    @property
    def data_width(self) -> int:
        return self._data_width

    @property
    def id_width(self) -> int:
        return self._id_width

    def create(self, *, path=None, src_loc_at: int = 0):
        return Interface(
            addr_width=self.addr_width,
            data_width=self.data_width,
            id_width=self.id_width,
            path=path,
            src_loc_at=1 + src_loc_at,
        )

    def __eq__(self, other):
        return (
            isinstance(other, AxiSignature)
            and self.addr_width == other.addr_width
            and self.data_width == other.data_width
            and self.id_width == other.id_width
        )

    def __repr__(self):
        return f"axi.Signature({self.members!r})"


class Interface(wiring.PureInterface):
    """AXI4 interface using the AXI signature."""

    def __init__(
        self,
        *,
        addr_width: int,
        data_width: int,
        id_width: int,
        path=None,
        src_loc_at: int = 0,
    ) -> None:
        super().__init__(
            Signature(
                addr_width=addr_width,
                data_width=data_width,
                id_width=id_width,
            ),
            path=path,
            src_loc_at=1 + src_loc_at,
        )

    @property
    def addr_width(self) -> int:
        return self.signature.addr_width

    @property
    def data_width(self) -> int:
        return self.signature.data_width

    @property
    def id_width(self) -> int:
        return self.signature.id_width

    def __repr__(self):
        return f"axi.Interface({self.signature!r})"


class AxiPipelinedMaster(wiring.Component):
    """Pipelined AXI4 master with request/response streams.

    Single-beat transactions only. Read and write channels are independent.
    Write data is issued in the same order as accepted write addresses.
    """

    def __init__(
        self,
        addr_width: int,
        data_width: int,
        id_width: int = 4,
        tag_width: int = 16,
        read_req_depth: int = 8,
        write_req_depth: int = 8,
        inflight_depth: int = 8,
    ):
        self._addr_width = addr_width
        self._data_width = data_width
        self._id_width = id_width
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
                "axi": Out(
                    AxiSignature(
                        addr_width=addr_width, data_width=data_width, id_width=id_width
                    )
                ),
            }
        )

    def elaborate(self, platform):
        m = Module()

        bytes_per_beat = self._data_width // 8
        size_code = int(log2_int(bytes_per_beat))

        m.submodules.read_tag_fifo = read_tag_fifo = fifo.SyncFIFO(
            width=self._tag_width,
            depth=self._inflight_depth,
        )
        m.submodules.write_tag_fifo = write_tag_fifo = fifo.SyncFIFO(
            width=self._tag_width,
            depth=self._inflight_depth,
        )
        w_data_layout = data.StructLayout(
            {
                "data": unsigned(self._data_width),
                "strb": unsigned(bytes_per_beat),
            }
        )
        m.submodules.write_data_fifo = write_data_fifo = fifo.SyncFIFOBuffered(
            width=Shape.cast(w_data_layout).width,
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

        # Read address channel
        ar_fire = self.read_req.valid & self.read_req.ready
        m.d.comb += [
            self.read_req.ready.eq(self.axi.ar_ready & read_tag_fifo.w_rdy),
            self.axi.ar_valid.eq(self.read_req.valid),
            self.axi.ar_addr.eq(self.read_req.payload.addr),
            self.axi.ar_id.eq(0),
            self.axi.ar_len.eq(0),
            self.axi.ar_size.eq(size_code),
            self.axi.ar_burst.eq(1),
            read_tag_fifo.w_en.eq(ar_fire),
            read_tag_fifo.w_data.eq(self.read_req.payload.tag),
        ]

        # Read data channel
        r_fire = self.axi.r_valid & self.axi.r_ready
        read_resp_in = Signal(self._read_resp_layout)
        m.d.comb += [
            self.axi.r_ready.eq(read_resp_fifo.w_rdy & read_tag_fifo.r_rdy),
            read_resp_in.data.eq(self.axi.r_data),
            read_resp_in.tag.eq(read_tag_fifo.r_data),
            read_resp_fifo.w_en.eq(r_fire),
            read_resp_fifo.w_data.eq(read_resp_in),
            read_tag_fifo.r_en.eq(r_fire),
        ]

        # Write address channel
        aw_fire = self.write_req.valid & self.write_req.ready
        write_data_in = Signal(w_data_layout)
        m.d.comb += [
            self.write_req.ready.eq(
                self.axi.aw_ready & write_tag_fifo.w_rdy & write_data_fifo.w_rdy
            ),
            self.axi.aw_valid.eq(self.write_req.valid),
            self.axi.aw_addr.eq(self.write_req.payload.addr),
            self.axi.aw_id.eq(0),
            self.axi.aw_len.eq(0),
            self.axi.aw_size.eq(size_code),
            self.axi.aw_burst.eq(1),
            write_tag_fifo.w_en.eq(aw_fire),
            write_tag_fifo.w_data.eq(self.write_req.payload.tag),
            write_data_in.data.eq(self.write_req.payload.data),
            write_data_in.strb.eq(self.write_req.payload.sel),
            write_data_fifo.w_en.eq(aw_fire),
            write_data_fifo.w_data.eq(write_data_in),
        ]

        # Write data channel (ordered with respect to accepted AW queue)
        m.d.comb += [
            self.axi.w_valid.eq(write_data_fifo.r_rdy),
            self.axi.w_data.eq(write_data_fifo.r_data.data),
            self.axi.w_strb.eq(write_data_fifo.r_data.strb),
            self.axi.w_last.eq(1),
            write_data_fifo.r_en.eq(self.axi.w_valid & self.axi.w_ready),
        ]

        # Write response channel
        b_fire = self.axi.b_valid & self.axi.b_ready
        write_resp_in = Signal(self._write_resp_layout)
        m.d.comb += [
            self.axi.b_ready.eq(write_resp_fifo.w_rdy & write_tag_fifo.r_rdy),
            write_resp_in.tag.eq(write_tag_fifo.r_data),
            write_resp_fifo.w_en.eq(b_fire),
            write_resp_fifo.w_data.eq(write_resp_in),
            write_tag_fifo.r_en.eq(b_fire),
        ]

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

        return m


class AxiToWishboneBridge(wiring.Component):
    """Simple AXI4 master to wishbone bridge.

    Supports only one transaction at a time.
    """

    def __init__(self, *, addr_width=axi_addr_width, data_width=axi_data_width):
        super().__init__(
            {
                "axi": In(
                    Signature(
                        addr_width=addr_width,
                        data_width=data_width,
                        id_width=axi_id_width,
                    )
                ),
                "wb": Out(
                    wb.Signature(addr_width=addr_width - 2, data_width=data_width)
                ),
            }
        )
        self._addr_width = addr_width
        self._data_width = data_width

    def elaborate(self, platform):
        m = Module()

        bytes_per_beat = self._data_width // 8

        # Write path state
        aw_pending = Signal()
        aw_addr = Signal(unsigned(self._addr_width))
        aw_id = Signal(unsigned(axi_id_width))
        w_pending = Signal()
        w_data = Signal(unsigned(self._data_width))
        w_strb = Signal(unsigned(bytes_per_beat))
        b_valid = Signal()
        b_id = Signal(unsigned(axi_id_width))

        # Read path state
        r_valid = Signal()
        r_id = Signal(unsigned(axi_id_width))
        r_data = Signal(unsigned(self._data_width))

        wb_busy = Signal()

        # Serialize read vs write to keep the bridge simple and deterministic.
        can_accept_write = ~aw_pending & ~w_pending & ~b_valid & ~r_valid & ~wb_busy
        can_accept_read = ~r_valid & ~aw_pending & ~w_pending & ~b_valid & ~wb_busy
        m.d.comb += [
            self.axi.aw_ready.eq(can_accept_write),
            self.axi.w_ready.eq(can_accept_write),
            self.axi.ar_ready.eq(can_accept_read),
        ]
        with m.If(self.axi.aw_valid & self.axi.aw_ready):
            m.d.sync += [
                aw_pending.eq(1),
                aw_addr.eq(self.axi.aw_addr),
                aw_id.eq(self.axi.aw_id),
            ]
        with m.If(self.axi.w_valid & self.axi.w_ready):
            m.d.sync += [
                w_pending.eq(1),
                w_data.eq(self.axi.w_data),
                w_strb.eq(self.axi.w_strb),
            ]

        # Accept read address when no response pending and bus idle
        with m.If(self.axi.ar_valid & self.axi.ar_ready):
            m.d.sync += [
                wb_busy.eq(1),
                r_id.eq(self.axi.ar_id),
                self.wb.cyc.eq(1),
                self.wb.stb.eq(1),
                self.wb.we.eq(0),
                self.wb.adr.eq(self.axi.ar_addr >> 2),
                self.wb.sel.eq(~0),
            ]

        # Launch write when address+data collected and bus idle
        with m.If(~wb_busy & aw_pending & w_pending & ~b_valid):
            m.d.sync += [
                wb_busy.eq(1),
                self.wb.cyc.eq(1),
                self.wb.stb.eq(1),
                self.wb.we.eq(1),
                self.wb.adr.eq(aw_addr >> 2),
                self.wb.dat_w.eq(w_data),
                self.wb.sel.eq(w_strb),
            ]

        # Complete Wishbone cycle
        with m.If(wb_busy & self.wb.ack):
            m.d.sync += [
                self.wb.cyc.eq(0),
                self.wb.stb.eq(0),
                wb_busy.eq(0),
            ]
            with m.If(self.wb.we):
                m.d.sync += [
                    b_valid.eq(1),
                    b_id.eq(aw_id),
                    aw_pending.eq(0),
                    w_pending.eq(0),
                ]
            with m.Else():
                m.d.sync += [
                    r_valid.eq(1),
                    r_data.eq(self.wb.dat_r),
                ]

        # Write response channel
        m.d.comb += [
            self.axi.b_valid.eq(b_valid),
            self.axi.b_id.eq(b_id),
            self.axi.b_resp.eq(0),
        ]
        with m.If(b_valid & self.axi.b_ready):
            m.d.sync += b_valid.eq(0)

        # Read response channel
        m.d.comb += [
            self.axi.r_valid.eq(r_valid),
            self.axi.r_id.eq(r_id),
            self.axi.r_data.eq(r_data),
            self.axi.r_resp.eq(0),
            self.axi.r_last.eq(1),
        ]
        with m.If(r_valid & self.axi.r_ready):
            m.d.sync += r_valid.eq(0)

        return m
