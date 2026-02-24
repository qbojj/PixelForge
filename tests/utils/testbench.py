from amaranth import *
from amaranth.lib import wiring
from amaranth_soc import csr
from amaranth_soc.csr.wishbone import WishboneCSRBridge
from amaranth_soc.memory import MemoryMap
from amaranth_soc.wishbone.bus import Arbiter, Decoder
from amaranth_soc.wishbone.sram import WishboneSRAM

from gpu.utils.axi import AxiToWishboneBridge
from gpu.utils.layouts import wb_bus_addr_width, wb_bus_data_width, wb_bus_granularity
from tests.utils.memory import DebugAccess, get_memory_resource


def div_ceil(a: int, b: int) -> int:
    return (a + b - 1) // b


class SimpleTestbench(Elaboratable):
    def __init__(
        self,
        dut: Elaboratable | None = None,
        addr_width: int = wb_bus_addr_width,
        data_width: int = wb_bus_data_width,
        granularity: int = wb_bus_granularity,
        features: frozenset = frozenset(),
        mem_size: int = 1024 * 4,
        mem_addr: int = 0x80000000,
    ):
        self.dut = dut
        self.data_width = data_width
        self.addr_width = addr_width
        self.granularity = granularity
        self.mem_size = mem_size
        self.mem_addr = mem_addr

        self.arbiter = Arbiter(
            addr_width=addr_width,
            data_width=data_width,
            granularity=granularity,
            features=features,
        )

        self.decoder = Decoder(
            addr_width=addr_width,
            data_width=data_width,
            granularity=granularity,
            features=features,
        )

        self.dbg_access = DebugAccess(
            addr_width=addr_width, data_width=data_width, granularity=granularity
        )

        self.csr_decoder = csr.Decoder(
            addr_width=20, data_width=granularity, alignment=3
        )

        self.mem = WishboneSRAM(
            size=mem_size, data_width=data_width, granularity=granularity, writable=True
        )

        self.arbiter.add(self.dbg_access)
        self.csrs = []
        self.axi_bridges: list[tuple[wiring.PureInterface, AxiToWishboneBridge]] = []

    def add_axi_master(self, axi_bus: wiring.PureInterface):
        bridge = AxiToWishboneBridge(
            addr_width=self.addr_width + 2,
            data_width=self.data_width,
        )
        self.axi_bridges.append((axi_bus, bridge))
        self.arbiter.add(bridge.wb)

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

    def elaborate(self, platform) -> Module:
        m = Module()
        m.submodules.arbiter = self.arbiter
        m.submodules.decoder = self.decoder
        m.submodules.csr_decoder = self.csr_decoder
        m.submodules.mem = self.mem

        m.submodules.csr_bridge = self.csr_bridge = csr_bridge = WishboneCSRBridge(
            self.csr_decoder.bus, data_width=self.data_width
        )

        self.decoder.add(self.mem.wb_bus, addr=self.mem_addr)
        self.decoder.add(csr_bridge.wb_bus)

        wiring.connect(m, self.arbiter.bus, self.decoder.bus)
        self.arbiter.bus.memory_map = self.decoder.bus.memory_map

        if self.dut is not None:
            m.submodules.dut = self.dut

        for idx, (axi_bus, bridge) in enumerate(self.axi_bridges):
            m.submodules[f"axi_bridge_{idx}"] = bridge
            wiring.connect(m, axi_bus, bridge.axi)

        return m

    async def initialize_csrs(self, ctx):
        mmap: MemoryMap = self.decoder.bus.memory_map

        for name, data in self.csrs:
            for path, value in data:
                res = get_memory_resource(mmap, (name,) + path)
                await self.dbg_access.write_bytes(ctx, res.start, value)

    async def initialize_memory(self, ctx, addr: int, data: bytes):
        await self.dbg_access.write_bytes(ctx, addr, data)
