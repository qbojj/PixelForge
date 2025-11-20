import inspect

from amaranth import *
from amaranth.lib import wiring
from amaranth.sim import Simulator
from amaranth_soc import csr
from amaranth_soc.csr.wishbone import WishboneCSRBridge
from amaranth_soc.memory import MemoryMap
from amaranth_soc.wishbone.bus import Arbiter, Decoder

from gpu.input_assembly.cores import InputTopologyProcessor
from gpu.utils.types import InputTopology
from tests.utils.streams import stream_get, stream_put

from ..utils.memory import DebugAccess, get_memory_resource


def make_test_input_topology_processor(
    input_topology: InputTopology,
    restart_index: int | None,
    base_vertex: int,
    input: list[int],
    expected: list[int],
):
    m = Module()

    m.submodules.dut = dut = InputTopologyProcessor()
    m.submodules.csr_decoder = csr_decoder = csr.Decoder(addr_width=16, data_width=8)
    csr_decoder.add(dut.csr_bus, name="input_topology_processor")

    m.submodules.csr_bridge = csr_bridge = WishboneCSRBridge(
        csr_decoder.bus, data_width=32
    )

    m.submodules.decoder = decoder = Decoder(
        addr_width=32, data_width=32, granularity=8
    )
    m.submodules.arbiter = arbiter = Arbiter(
        addr_width=32, data_width=32, granularity=8
    )
    m.submodules.dbg_access = dbg_access = DebugAccess(
        addr_width=32, data_width=32, granularity=8
    )

    decoder.add(csr_bridge.wb_bus, addr=0x00000000)

    arbiter.add(dbg_access.wb_bus)

    wiring.connect(m, arbiter.bus, decoder.bus)
    arbiter.bus.memory_map = decoder.bus.memory_map

    mmap: MemoryMap = decoder.bus.memory_map

    index_pusher_done = Signal()
    can_send = Signal()
    can_end = Signal()

    m.d.sync += can_end.eq(index_pusher_done & dut.ready)

    async def index_pusher(ctx):
        with ctx.critical():
            await ctx.tick().until(can_send)
            await stream_put(ctx, dut.is_index, input)
            ctx.set(index_pusher_done, 1)

    async def tb(ctx):
        # Configure DUT
        print()
        print("Configuring DUT...")
        input_topology_reg = get_memory_resource(
            mmap, "input_topology_processor.input_topology"
        )
        restart_enable_reg = get_memory_resource(
            mmap, "input_topology_processor.primitive_restart_enable"
        )
        restart_index_reg = get_memory_resource(
            mmap, "input_topology_processor.primitive_restart_index"
        )
        base_vertex_reg = get_memory_resource(
            mmap, "input_topology_processor.base_vertex"
        )

        await dbg_access.write(ctx, input_topology_reg.start, [input_topology])
        if restart_index is not None:
            await dbg_access.write(ctx, restart_enable_reg.start, [1])
            await dbg_access.write(ctx, restart_index_reg.start, [restart_index])
        else:
            await dbg_access.write(ctx, restart_enable_reg.start, [0])
        await dbg_access.write(ctx, base_vertex_reg.start, [base_vertex])

        ctx.set(can_send, 1)

        print("Waiting for DUT to be ready...")
        indices = await stream_get(ctx, dut.os_index, can_end)

        print(f"Generated indices: {indices}, expected: {expected}")
        assert indices == expected, f"Expected indices {expected}, got {indices}"

    sim = Simulator(m)
    sim.add_clock(1e-9)
    sim.add_process(index_pusher)
    sim.add_testbench(tb)

    prev_func_name = None
    for frame in reversed(inspect.stack()):
        if frame.function.startswith("test_"):
            prev_func_name = frame.function
            break

    try:
        sim.run()
    except Exception:
        sim.reset()

        with sim.write_vcd(
            f"{prev_func_name}.vcd", f"{prev_func_name}.gtkw", traces=dut
        ):
            sim.run()


def test_triangle_list():
    make_test_input_topology_processor(
        input_topology=InputTopology.TRIANGLE_LIST,
        restart_index=None,
        base_vertex=0,
        input=[0, 1, 2, 3, 4, 5],
        expected=[0, 1, 2, 3, 4, 5],
    )


def test_base_vertex():
    make_test_input_topology_processor(
        input_topology=InputTopology.TRIANGLE_LIST,
        restart_index=None,
        base_vertex=10,
        input=[0, 1, 2, 3, 4, 5],
        expected=[10, 11, 12, 13, 14, 15],
    )


def test_triangle_list_with_restart():
    make_test_input_topology_processor(
        input_topology=InputTopology.TRIANGLE_LIST,
        restart_index=0xFFFF,
        base_vertex=0,
        input=[0, 1, 2, 0xFFFF, 3, 4, 5, 6, 0xFFFF, 7, 8, 9],
        expected=[0, 1, 2, 3, 4, 5, 7, 8, 9],
    )


def test_triangle_strip():
    make_test_input_topology_processor(
        input_topology=InputTopology.TRIANGLE_STRIP,
        restart_index=None,
        base_vertex=0,
        input=[0, 1, 2, 3, 4],
        expected=[0, 1, 2, 2, 1, 3, 3, 2, 4],
    )


def test_triangle_fan():
    make_test_input_topology_processor(
        input_topology=InputTopology.TRIANGLE_FAN,
        restart_index=None,
        base_vertex=0,
        input=[0, 1, 2, 3, 4],
        expected=[0, 1, 2, 0, 2, 3, 0, 3, 4],
    )
