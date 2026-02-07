import amaranth_soc.wishbone.bus as wb
from amaranth import *
from amaranth.lib import wiring
from amaranth.lib.wiring import (
    Component,
    In,
    Out,
)
from amaranth.lib.wiring import Signature as WiringSignature
from amaranth.utils import exact_log2
from amaranth_soc.memory import MemoryMap

__all__ = [
    "Signature",
    "Interface",
    "WishboneMasterToAvalonBridge",
    "WishboneSlaveToAvalonBridge",
]


class Signature(WiringSignature):
    """Avalon-MM master-facing signature.

    Shaped to mirror amaranth-soc's Wishbone signature style: validates widths, stores
    constructor parameters, and provides a ``create`` helper that returns the matching Interface.
    """

    def __init__(
        self,
        *,
        addr_width: int,
        data_width: int,
        burst_count_width: int | None = None,
        has_byte_enable: bool = True,
        has_readdatavalid: bool = False,
        pipelined: bool = False,
    ) -> None:
        if not isinstance(addr_width, int) or addr_width < 0:
            raise TypeError(
                f"Address width must be a non-negative integer, not {addr_width!r}"
            )
        if not isinstance(data_width, int) or data_width <= 0:
            raise TypeError(
                f"Data width must be a positive integer, not {data_width!r}"
            )
        if data_width % 8 != 0:
            raise ValueError("Avalon-MM data_width must be a multiple of 8")
        if burst_count_width is not None and (
            not isinstance(burst_count_width, int) or burst_count_width <= 0
        ):
            raise TypeError(
                f"Burst count width must be a positive integer or None, not {burst_count_width!r}"
            )

        self._addr_width = addr_width
        self._data_width = data_width
        self._burst_count_width = burst_count_width
        self._has_byte_enable = bool(has_byte_enable)
        self._has_readdatavalid = bool(has_readdatavalid)
        self._pipelined = bool(pipelined)

        members: dict[str, wiring.Direction] = {
            "address": Out(unsigned(addr_width)),
            "write": Out(1),
            "read": Out(1),
            "writedata": Out(unsigned(data_width)),
            "readdata": In(unsigned(data_width)),
            "waitrequest": In(1),
        }

        if self._has_byte_enable:
            members["byteenable"] = Out(unsigned(data_width // 8))
        if self._burst_count_width is not None:
            members["burstcount"] = Out(unsigned(self._burst_count_width))

        if self._has_readdatavalid:
            members["readdatavalid"] = In(1)

        if self._pipelined:
            members["readdatavalid"] = In(1)
            members["writeresponsevalid"] = In(1)

        super().__init__(members)

    @property
    def addr_width(self) -> int:
        return self._addr_width

    @property
    def data_width(self) -> int:
        return self._data_width

    @property
    def burst_count_width(self) -> int | None:
        return self._burst_count_width

    @property
    def has_byte_enable(self) -> bool:
        return self._has_byte_enable

    @property
    def has_readdatavalid(self) -> bool:
        return self._has_readdatavalid

    @property
    def pipelined(self) -> bool:
        return self._pipelined

    def create(self, *, path=None, src_loc_at: int = 0):
        """Create a compatible Avalon-MM interface."""

        return Interface(
            addr_width=self.addr_width,
            data_width=self.data_width,
            burst_count_width=self.burst_count_width,
            has_byte_enable=self.has_byte_enable,
            has_readdatavalid=self.has_readdatavalid,
            pipelined=self.pipelined,
            path=path,
            src_loc_at=1 + src_loc_at,
        )

    def __eq__(self, other):
        return (
            isinstance(other, Signature)
            and self.addr_width == other.addr_width
            and self.data_width == other.data_width
            and self.burst_count_width == other.burst_count_width
            and self.has_byte_enable == other.has_byte_enable
            and self.has_readdatavalid == other.has_readdatavalid
            and self.pipelined == other.pipelined
        )

    def __repr__(self):
        return f"avalon.Signature({self.members!r})"


class Interface(wiring.PureInterface):
    """Avalon-MM interface using the Avalon signature."""

    def __init__(
        self,
        *,
        addr_width: int,
        data_width: int,
        burst_count_width: int | None = None,
        has_byte_enable: bool = True,
        has_readdatavalid: bool = True,
        pipelined: bool = False,
        path=None,
        src_loc_at: int = 0,
    ) -> None:
        super().__init__(
            Signature(
                addr_width=addr_width,
                data_width=data_width,
                burst_count_width=burst_count_width,
                has_byte_enable=has_byte_enable,
                has_readdatavalid=has_readdatavalid,
                pipelined=pipelined,
            ),
            path=path,
            src_loc_at=1 + src_loc_at,
        )
        self._memory_map = None

    @property
    def addr_width(self) -> int:
        return self.signature.addr_width

    @property
    def data_width(self) -> int:
        return self.signature.data_width

    @property
    def burst_count_width(self) -> int | None:
        return self.signature.burst_count_width

    @property
    def has_byte_enable(self) -> bool:
        return self.signature.has_byte_enable

    @property
    def has_readdatavalid(self) -> bool:
        return self.signature.has_readdatavalid

    @property
    def pipelined(self) -> bool:
        return self.signature.pipelined

    @property
    def memory_map(self):
        if self._memory_map is None:
            raise AttributeError(f"{self!r} does not have a memory map")
        return self._memory_map

    @memory_map.setter
    def memory_map(self, memory_map):
        if not isinstance(memory_map, MemoryMap):
            raise TypeError(
                f"Memory map must be an instance of MemoryMap, not {memory_map!r}"
            )

        # If byteenable is present, granularity is 8 bits; otherwise granularity equals data width.
        granularity = 8 if self.has_byte_enable else self.data_width
        if memory_map.data_width != granularity:
            raise ValueError(
                f"Memory map has data width {memory_map.data_width}, which is not the same as bus granularity {granularity}"
            )

        granularity_bits = exact_log2(self.data_width // granularity)
        effective_addr_width = self.addr_width + granularity_bits
        effective_addr_width = max(1, effective_addr_width)

        if memory_map.addr_width != effective_addr_width:
            raise ValueError(
                f"Memory map has address width {memory_map.addr_width}, which is not the same as "
                f"bus effective address width {effective_addr_width} (= {self.addr_width} address bits + "
                f"{granularity_bits} granularity bits)"
            )

        self._memory_map = memory_map

    def __repr__(self):
        return f"avalon.Interface({self.signature!r})"


class WishboneMasterToAvalonBridge(Component):
    """Wishbone initiator to Avalon-MM master bridge."""

    def __init__(self, bus: wb.Interface):
        if isinstance(bus, wiring.FlippedInterface):
            unflipped_bus = wiring.flipped(bus)
        else:
            unflipped_bus = bus

        if not isinstance(unflipped_bus, wb.Interface):
            raise TypeError(f"bus must be a Wishbone Interface, not {unflipped_bus!r}")

        if not unflipped_bus.features.issubset({wb.Feature.STALL}):
            raise ValueError(
                "Wishbone features other than STALL are not supported by Avalon bridge",
                str(list(unflipped_bus.features)),
            )

        self._addr_width = unflipped_bus.signature.addr_width
        self._data_width = unflipped_bus.signature.data_width
        self._granularity = unflipped_bus.signature.granularity
        self._shift_bits = exact_log2(unflipped_bus.signature.data_width // 8)
        self._has_byte_enable = (
            self._granularity < self._data_width and self._granularity == 8
        )

        if self._granularity != 8 and self._granularity != self._data_width:
            raise ValueError(
                f"Unsupported Wishbone granularity for Avalon bridge: data_width={self._data_width}, granularity={self._granularity}"
            )

        self._pipelined = wb.Feature.STALL in unflipped_bus.features

        avl_signature = Signature(
            addr_width=self._addr_width + self._shift_bits,
            data_width=self._data_width,
            has_byte_enable=self._has_byte_enable,
            pipelined=self._pipelined,
        )

        super().__init__({"avl_bus": Out(avl_signature)})
        self._bus = bus

    def elaborate(self, platform):
        m = Module()

        wb_bus = self._bus
        avl = self.avl_bus

        op_send = Signal()
        m.d.comb += op_send.eq(wb_bus.cyc & wb_bus.stb)

        m.d.comb += avl.address.eq(Cat(Const(0, self._shift_bits), wb_bus.adr))
        m.d.comb += avl.writedata.eq(wb_bus.dat_w)
        m.d.comb += wb_bus.dat_r.eq(avl.readdata)
        m.d.comb += avl.write.eq(op_send & wb_bus.we)
        m.d.comb += avl.read.eq(op_send & ~wb_bus.we)

        if self._has_byte_enable:
            m.d.comb += avl.byteenable.eq(wb_bus.sel)

        if not self._pipelined:
            m.d.comb += wb_bus.ack.eq(~avl.waitrequest)
        else:
            m.d.comb += wb_bus.ack.eq(avl.readdatavalid | avl.writeresponsevalid)
            m.d.comb += wb_bus.stall.eq(avl.waitrequest)

        return m


class WishboneSlaveToAvalonBridge(Component):
    """Bridge a Wishbone target (slave) to an Avalon-MM target (slave).

    Accepts Avalon transactions and issues Wishbone responses with ack backpressure.
    Avalon is word-addressed for slaves

    Parameters
    ----------
    wb_bus : wb.Interface
        Wishbone target-side interface (word-addressed). The bridge derives the Avalon
        signature from this interface's signature.
    """

    def __init__(self, wb_bus: wb.Interface):
        # Handle flipped interfaces
        if isinstance(wb_bus, wiring.FlippedInterface):
            unflipped_bus = wiring.flipped(wb_bus)
        else:
            unflipped_bus = wb_bus

        if not isinstance(unflipped_bus, wb.Interface):
            raise TypeError(
                f"wb_bus must be a Wishbone Interface, not {unflipped_bus!r}"
            )

        if not unflipped_bus.features.issubset({wb.Feature.STALL}):
            raise ValueError(
                "Wishbone features other than STALL are not supported by Avalon bridge"
            )

        if unflipped_bus.data_width == unflipped_bus.granularity:
            has_byte_enable = False
        elif unflipped_bus.granularity == 8:
            has_byte_enable = True
        else:
            raise ValueError(
                "Unsupported Wishbone granularity for Avalon bridge: "
                f"data_width={unflipped_bus.data_width}, granularity={unflipped_bus.granularity}"
            )

        self._pipelined = wb.Feature.STALL in unflipped_bus.features

        # Create Avalon signature with same address and data widths (word-addressed on both sides)
        # Derive byteenable from Wishbone granularity
        avl_sig = Signature(
            addr_width=unflipped_bus.addr_width,
            data_width=unflipped_bus.data_width,
            has_byte_enable=has_byte_enable,
            pipelined=self._pipelined,
        )

        super().__init__({"avl_bus": In(avl_sig)})
        self._wb_bus = wb_bus

    def elaborate(self, platform):
        m = Module()

        avl = self.avl_bus
        wb_bus = self._wb_bus

        send_op = Signal()
        m.d.comb += send_op.eq(avl.read | avl.write)

        m.d.comb += wb_bus.dat_w.eq(avl.writedata)
        m.d.comb += avl.readdata.eq(wb_bus.dat_r)
        m.d.comb += wb_bus.adr.eq(avl.address)
        m.d.comb += wb_bus.we.eq(avl.write)

        if not self._pipelined:
            m.d.comb += wb_bus.cyc.eq(send_op)
            m.d.comb += wb_bus.stb.eq(send_op)
            m.d.comb += avl.waitrequest.eq(~wb_bus.ack)
        else:
            max_transactions = 8
            was_write = Signal(max_transactions)

            submit_idx = Signal(range(max_transactions + 1))
            complete_idx = Signal.like(submit_idx)

            can_accept = Signal()
            m.d.comb += can_accept.eq(submit_idx + 1 != complete_idx)

            m.d.comb += wb_bus.cyc.eq(send_op | (submit_idx != complete_idx))
            m.d.comb += wb_bus.stb.eq(send_op & can_accept)
            m.d.comb += avl.waitrequest.eq(~can_accept)

            with m.If(wb_bus.stb):
                m.d.sync += was_write[submit_idx].eq(wb_bus.we)
                m.d.sync += submit_idx.eq(submit_idx + 1)

            with m.If(wb_bus.ack):
                with m.If(was_write[complete_idx]):
                    m.d.comb += avl.writeresponsevalid.eq(1)
                with m.Else():
                    m.d.comb += avl.readdatavalid.eq(1)
                m.d.sync += complete_idx.eq(complete_idx + 1)

        if self.avl_bus.has_byte_enable:
            m.d.comb += wb_bus.sel.eq(avl.byteenable)
        else:
            m.d.comb += wb_bus.sel.eq(~0)

        return m
