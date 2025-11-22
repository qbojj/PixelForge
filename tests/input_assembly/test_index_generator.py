from amaranth import *
from amaranth.lib import wiring
from amaranth.sim import Simulator
from amaranth_soc import csr
from amaranth_soc.csr.wishbone import WishboneCSRBridge
from amaranth_soc.memory import MemoryMap
from amaranth_soc.wishbone.bus import Arbiter, Decoder
from amaranth_soc.wishbone.sram import WishboneSRAM

from gpu.input_assembly.cores import IndexGenerator
from gpu.utils.types import IndexKind

from ..utils.memory import DebugAccess, get_memory_resource
from ..utils.streams import stream_get


def make_test_index_generator(
    addr: int,
    count: int,
    kind: IndexKind,
    memory_data: list[int],
    expected: list[int],
):
    m = Module()

    m.submodules.dut = dut = IndexGenerator()
    m.submodules.csr_decoder = csr_decoder = csr.Decoder(addr_width=16, data_width=8)
    csr_decoder.add(dut.csr_bus, name="index_gen")

    m.submodules.csr_bridge = csr_bridge = WishboneCSRBridge(
        csr_decoder.bus, data_width=32
    )

    m.submodules.mem = mem = WishboneSRAM(
        size=1024, data_width=32, granularity=8, writable=True
    )
    m.submodules.decoder = decoder = Decoder(
        addr_width=32, data_width=32, granularity=8
    )
    m.submodules.arbiter = arbiter = Arbiter(
        addr_width=32, data_width=32, granularity=8
    )
    dbg_access = DebugAccess(addr_width=32, data_width=32, granularity=8)

    decoder.add(mem.wb_bus, addr=0x80000000)
    decoder.add(csr_bridge.wb_bus, addr=0x00000000)

    arbiter.add(dut.bus)
    arbiter.add(dbg_access)

    wiring.connect(m, arbiter.bus, decoder.bus)
    arbiter.bus.memory_map = decoder.bus.memory_map

    mmap: MemoryMap = decoder.bus.memory_map

    async def tb(ctx):
        # setup memory
        print()
        print(f"Loading memory at {addr:#010x} with data: {memory_data}")
        await dbg_access.write_bytes(ctx, addr, memory_data)

        # Configure DUT
        print()
        print("Configuring DUT...")
        address_addr = get_memory_resource(mmap, "index_gen.address").start
        count_addr = get_memory_resource(mmap, "index_gen.count").start
        kind_addr = get_memory_resource(mmap, "index_gen.kind").start
        start_addr = get_memory_resource(mmap, "index_gen.start").start

        await ctx.tick().until(dut.ready)

        await dbg_access.write(ctx, address_addr, [addr])
        await dbg_access.write(ctx, count_addr, [count])
        await dbg_access.write(ctx, kind_addr, [kind])
        await dbg_access.write(ctx, start_addr, [1])

        await ctx.tick()

        # pull indices until ready
        indices = await stream_get(ctx, dut.os_index, dut.ready)

        print(f"Generated indices: {indices}, expected: {expected}")
        assert indices == expected, f"Expected indices {expected}, got {indices}"

    sim = Simulator(m)
    sim.add_clock(1e-9)
    sim.add_testbench(tb)

    try:
        sim.run()
    except Exception:
        sim.reset()

        with sim.write_vcd(
            "test_index_generator.vcd", "test_index_generator.gtkw", traces=dut
        ):
            sim.run()


def test_not_indexed():
    make_test_index_generator(
        addr=0x80000000,
        count=10,
        kind=IndexKind.NOT_INDEXED,
        memory_data=b"",
        expected=list(range(10)),
    )


def test_indexed_u32():
    make_test_index_generator(
        addr=0x80000000,
        count=5,
        kind=IndexKind.U32,
        memory_data=b"".join((i.to_bytes(4, "little") for i in [0, 2, 4, 1, 5])),
        expected=[0, 2, 4, 1, 5],
    )


def test_indexed_u16():
    make_test_index_generator(
        addr=0x80000000,
        count=6,
        kind=IndexKind.U16,
        memory_data=b"".join((i.to_bytes(2, "little") for i in [2, 3, 4, 5, 1, 0])),
        expected=[2, 3, 4, 5, 1, 0],
    )


def test_indexed_u8():
    make_test_index_generator(
        addr=0x80000000,
        count=8,
        kind=IndexKind.U8,
        memory_data=b"".join(
            (i.to_bytes(1, "little") for i in [1, 2, 3, 4, 5, 6, 7, 8])
        ),
        expected=[1, 2, 3, 4, 5, 6, 7, 8],
    )


def test_indexed_u8_unaligned():
    make_test_index_generator(
        addr=0x80000002,
        count=8,
        kind=IndexKind.U8,
        memory_data=b"".join(
            (i.to_bytes(1, "little") for i in [1, 2, 3, 4, 5, 6, 7, 8])
        ),
        expected=[1, 2, 3, 4, 5, 6, 7, 8],
    )
