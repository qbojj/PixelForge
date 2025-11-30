from amaranth import *
from amaranth.lib import wiring
from amaranth_soc import csr
from amaranth_soc.csr.wishbone import WishboneCSRBridge
from amaranth_soc.memory import MemoryMap
from amaranth_soc.wishbone.bus import Arbiter, Decoder
from amaranth_soc.wishbone.sram import WishboneSRAM

from gpu.utils.layouts import wb_bus_addr_width, wb_bus_data_width, wb_bus_granularity
from tests.utils.memory import DebugAccess, get_memory_resource


def div_ceil(a: int, b: int) -> int:
    return (a + b - 1) // b


class SimpleTestbench:
    def __init__(
        self,
        dut: Module,
        addr_width: int = wb_bus_addr_width,
        data_width: int = wb_bus_data_width,
        granularity: int = wb_bus_granularity,
        features: frozenset = frozenset(),
        mem_size: int = 1024 * 4,
        mem_addr: int = 0x80000000,
    ):
        self.m = m = Module()

        m.submodules.dut = self.dut = dut

        self.data_width = data_width
        self.addr_width = addr_width
        self.granularity = granularity
        self.mem_size = mem_size
        self.mem_addr = mem_addr

        m.submodules.arbiter = self.arbiter = Arbiter(
            addr_width=addr_width,
            data_width=data_width,
            granularity=granularity,
            features=features,
        )

        m.submodules.decoder = self.decoder = Decoder(
            addr_width=addr_width,
            data_width=data_width,
            granularity=granularity,
            features=features,
        )

        self.dbg_access = dbg_access = DebugAccess(
            addr_width=addr_width, data_width=data_width, granularity=granularity
        )

        m.submodules.csr_decoder = self.csr_decoder = csr.Decoder(
            addr_width=20, data_width=granularity, alignment=3
        )

        m.submodules.mem = self.mem = WishboneSRAM(
            size=mem_size, data_width=data_width, granularity=granularity, writable=True
        )

        self.arbiter.add(dbg_access)
        self.csrs = []

    def set_csrs(
        self,
        csr_bus: csr.Interface,
        data: list[tuple[tuple, bytes | Const]],
        name: str | None = None,
    ):
        if name is None:
            name = f"csr_window_{len(self.csrs)}"

        prepared_data = []
        for path, value in data:
            if not isinstance(value, bytes):
                value = Const.cast(value)
                value = value.value.to_bytes(div_ceil(len(value), 8), "little")
            prepared_data.append((path, value))

        self.csr_decoder.add(csr_bus, name=name)
        self.csrs.append((name, prepared_data))

    def make(self):
        m = self.m
        m.submodules.csr_bridge = self.csr_bridge = csr_bridge = WishboneCSRBridge(
            self.csr_decoder.bus, data_width=self.data_width
        )
        self.decoder.add(csr_bridge.wb_bus)
        self.decoder.add(self.mem.wb_bus, addr=self.mem_addr)

        wiring.connect(self.m, self.arbiter.bus, self.decoder.bus)
        self.arbiter.bus.memory_map = self.decoder.bus.memory_map

    async def initialize_csrs(self, ctx):
        mmap: MemoryMap = self.decoder.bus.memory_map

        for name, data in self.csrs:
            for path, value in data:
                res = get_memory_resource(mmap, (name,) + path)
                await self.dbg_access.write_bytes(ctx, res.start, value)

    async def initialize_memory(self, ctx, addr: int, data: bytes):
        await self.dbg_access.write_bytes(ctx, addr, data)
