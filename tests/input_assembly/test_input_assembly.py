import pytest
from amaranth import *
from amaranth.lib import wiring
from amaranth.sim import Simulator
from amaranth_soc import csr
from amaranth_soc.csr.wishbone import WishboneCSRBridge
from amaranth_soc.memory import MemoryMap
from amaranth_soc.wishbone.bus import Arbiter, Decoder
from amaranth_soc.wishbone.sram import WishboneSRAM

from gpu.input_assembly.cores import InputAssembly
from gpu.input_assembly.layouts import InputData, InputMode
from gpu.utils.types import FixedPoint, FixedPoint_mem, Vector3, Vector4, Vector4_mem

from ..utils.memory import DebugAccess, get_memory_resource
from ..utils.streams import stream_get, stream_put

vec0001_mem = [
    FixedPoint_mem.from_float_const(0.0),
    FixedPoint_mem.from_float_const(0.0),
    FixedPoint_mem.from_float_const(0.0),
    FixedPoint_mem.from_float_const(1.0),
]

vec0001 = [
    FixedPoint.from_float_const(0.0),
    FixedPoint.from_float_const(0.0),
    FixedPoint.from_float_const(0.0),
    FixedPoint.from_float_const(1.0),
]

vec000 = [
    FixedPoint.from_float_const(0.0),
    FixedPoint.from_float_const(0.0),
    FixedPoint.from_float_const(0.0),
]


default_data = InputData.const({"constant_value": vec0001_mem})


def make_test_input_assembly(
    test_name: str,
    addr: int,
    input_idx: list[int],
    memory_data: bytes,
    expected: list,
    pos_mode: InputMode = InputMode.CONSTANT,
    pos_data: InputData = default_data,
    norm_mode: InputMode = InputMode.CONSTANT,
    norm_data: InputData = default_data,
    tex0_mode: InputMode = InputMode.CONSTANT,
    tex0_data: InputData = default_data,
    tex1_mode: InputMode = InputMode.CONSTANT,
    tex1_data: InputData = default_data,
    color_mode: InputMode = InputMode.CONSTANT,
    color_data: InputData = default_data,
):
    m = Module()

    m.submodules.dut = dut = InputAssembly()
    m.submodules.csr_decoder = csr_decoder = csr.Decoder(addr_width=16, data_width=8)
    csr_decoder.add(dut.csr_bus, name="input_assembly")

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

    decoder.add(mem.wb_bus, addr=addr)
    decoder.add(csr_bridge.wb_bus, addr=0x00000000)

    arbiter.add(dut.bus)
    arbiter.add(dbg_access)

    wiring.connect(m, arbiter.bus, decoder.bus)
    arbiter.bus.memory_map = decoder.bus.memory_map

    mmap: MemoryMap = decoder.bus.memory_map

    can_send = Signal()
    can_stop = Signal()

    finished = Signal()
    m.d.sync += finished.eq(can_stop & dut.ready)

    async def sender(ctx):
        with ctx.critical():
            await ctx.tick().until(can_send)
            await stream_put(ctx, dut.is_index, input_idx)
            ctx.set(can_stop, 1)

    async def tb(ctx):
        # setup memory
        print()
        print(f"Loading memory at {addr:#010x} with data: {memory_data}")
        await dbg_access.write_bytes(ctx, addr, memory_data)

        # Configure DUT
        print()
        print("Configuring DUT...")

        res_pos_mode = get_memory_resource(mmap, "input_assembly.position,mode")
        res_pos_data = get_memory_resource(mmap, "input_assembly.position,data")
        res_norm_mode = get_memory_resource(mmap, "input_assembly.normal,mode")
        res_norm_data = get_memory_resource(mmap, "input_assembly.normal,data")
        res_tex0_mode = get_memory_resource(mmap, "input_assembly.texcoords,0,mode")
        res_tex0_data = get_memory_resource(mmap, "input_assembly.texcoords,0,data")
        res_tex1_mode = get_memory_resource(mmap, "input_assembly.texcoords,1,mode")
        res_tex1_data = get_memory_resource(mmap, "input_assembly.texcoords,1,data")
        res_color_mode = get_memory_resource(mmap, "input_assembly.color,mode")
        res_color_data = get_memory_resource(mmap, "input_assembly.color,data")

        v_pos_mode = pos_mode.value.to_bytes(1, "little")
        v_norm_mode = norm_mode.value.to_bytes(1, "little")
        v_tex0_mode = tex0_mode.value.to_bytes(1, "little")
        v_tex1_mode = tex1_mode.value.to_bytes(1, "little")
        v_color_mode = color_mode.value.to_bytes(1, "little")

        v_pos_data = pos_data.as_bits().to_bytes(16, "little")
        v_norm_data = norm_data.as_bits().to_bytes(16, "little")
        v_tex0_data = tex0_data.as_bits().to_bytes(16, "little")
        v_tex1_data = tex1_data.as_bits().to_bytes(16, "little")
        v_color_data = color_data.as_bits().to_bytes(16, "little")

        await ctx.tick().until(dut.ready)
        await dbg_access.write_bytes(ctx, res_pos_mode.start, v_pos_mode)
        await dbg_access.write_bytes(ctx, res_pos_data.start, v_pos_data)
        await dbg_access.write_bytes(ctx, res_norm_mode.start, v_norm_mode)
        await dbg_access.write_bytes(ctx, res_norm_data.start, v_norm_data)
        await dbg_access.write_bytes(ctx, res_tex0_mode.start, v_tex0_mode)
        await dbg_access.write_bytes(ctx, res_tex0_data.start, v_tex0_data)
        await dbg_access.write_bytes(ctx, res_tex1_mode.start, v_tex1_mode)
        await dbg_access.write_bytes(ctx, res_tex1_data.start, v_tex1_data)
        await dbg_access.write_bytes(ctx, res_color_mode.start, v_color_mode)
        await dbg_access.write_bytes(ctx, res_color_data.start, v_color_data)
        await ctx.tick()

        ctx.set(can_send, 1)
        await ctx.tick()

        vertices = await stream_get(ctx, dut.os_vertex, finished)

        assert vertices == expected, f"Expected vertices {expected}, got {vertices}"

    sim = Simulator(m)
    sim.add_clock(1e-9)
    sim.add_process(sender)
    sim.add_testbench(tb)

    try:
        sim.run()
    except Exception:
        sim.reset()

        with sim.write_vcd(f"{test_name}.vcd", f"{test_name}.gtkw", traces=dut):
            sim.run()


vec1234_mem = Vector4_mem.const(
    [
        FixedPoint_mem.from_float_const(1.0),
        FixedPoint_mem.from_float_const(2.0),
        FixedPoint_mem.from_float_const(3.0),
        FixedPoint_mem.from_float_const(4.0),
    ]
)

vec1234 = Vector4.const(
    [
        FixedPoint.from_float_const(1.0),
        FixedPoint.from_float_const(2.0),
        FixedPoint.from_float_const(3.0),
        FixedPoint.from_float_const(4.0),
    ]
)

vec5678_mem = Vector4_mem.const(
    [
        FixedPoint_mem.from_float_const(5.0),
        FixedPoint_mem.from_float_const(6.0),
        FixedPoint_mem.from_float_const(7.0),
        FixedPoint_mem.from_float_const(8.0),
    ]
)

vec5678 = Vector4.const(
    [
        FixedPoint.from_float_const(5.0),
        FixedPoint.from_float_const(6.0),
        FixedPoint.from_float_const(7.0),
        FixedPoint.from_float_const(8.0),
    ]
)


def test_input_assembly_constant_only():
    make_test_input_assembly(
        test_name="test_input_assembly_constant_only",
        addr=0x80000000,
        memory_data=b"",
        input_idx=[0, 1, 2, 3, 4],
        expected=[
            {
                "position": vec0001,
                "normal": vec000,
                "texcoords": [vec0001, vec0001],
                "color": vec0001,
            }
            for _ in range(5)
        ],
    )


@pytest.mark.parametrize(
    ["test_name", "comp", "comp_in", "separation"],
    [
        ("test_input_assembly_continous_pos", "position", "pos", 0),
        ("test_input_assembly_continous_norm", "normal", "norm", 0),
        ("test_input_assembly_continous_tex0", "texcoords[0]", "tex0", 0),
        ("test_input_assembly_continous_tex1", "texcoords[1]", "tex1", 0),
        ("test_input_assembly_continous_col", "color", "color", 0),
        ("test_input_assembly_strided_4_pos", "position", "pos", 4),
        ("test_input_assembly_strided_8_norm", "normal", "norm", 8),
        ("test_input_assembly_strided_12_tex0", "texcoords[0]", "tex0", 12),
        ("test_input_assembly_strided_16_tex1", "texcoords[1]", "tex1", 16),
        ("test_input_assembly_strided_20_col", "color", "color", 20),
    ],
)
def test_input_assembly_single_component(test_name, comp, comp_in, separation):
    expected = [
        {
            "position": vec0001,
            "normal": vec000,
            "texcoords": [vec0001, vec0001],
            "color": vec0001,
        },
        {
            "position": vec0001,
            "normal": vec000,
            "texcoords": [vec0001, vec0001],
            "color": vec0001,
        },
    ]

    if comp == "texcoords[0]":
        expected[0]["texcoords"][0] = vec1234
        expected[1]["texcoords"][0] = vec5678
    elif comp == "texcoords[1]":
        expected[0]["texcoords"][1] = vec1234
        expected[1]["texcoords"][1] = vec5678
    elif comp == "normal":
        expected[0][comp] = Vector3.const(vec1234[:3])
        expected[1][comp] = Vector3.const(vec5678[:3])
    else:
        expected[0][comp] = vec1234
        expected[1][comp] = vec5678

    v = {
        f"{comp_in}_mode": InputMode.PER_VERTEX,
        f"{comp_in}_data": InputData.const(
            {
                "per_vertex": {
                    "address": 0x80000000,
                    "stride": (Vector4_mem.size // 8) + separation,
                }
            }
        ),
    }

    make_test_input_assembly(
        test_name=test_name,
        addr=0x80000000,
        memory_data=b"".join(
            (
                vec1234_mem.as_bits().to_bytes(16, "little"),
                b"\x00" * separation,
                vec5678_mem.as_bits().to_bytes(16, "little"),
            )
        ),
        input_idx=[0, 1],
        expected=expected,
        **v,
    )
