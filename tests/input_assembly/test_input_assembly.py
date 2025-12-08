import pytest
from amaranth import *
from amaranth.sim import Simulator

from gpu.input_assembly.cores import InputAssembly
from gpu.input_assembly.layouts import InputData, InputMode
from gpu.utils.types import Vector4_mem

from ..utils.streams import stream_testbench
from ..utils.testbench import SimpleTestbench

default_data = InputData.const({"constant_value": [0.0, 0.0, 0.0, 1.0]})


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
    dut = InputAssembly()
    t = SimpleTestbench(mem_addr=addr, mem_size=1024)

    t.set_csrs(
        dut.csr_bus,
        [
            ((("position", "mode"),), pos_mode),
            ((("position", "data"),), pos_data),
            ((("normal", "mode"),), norm_mode),
            ((("normal", "data"),), norm_data),
            ((("texcoords", "0", "mode"),), tex0_mode),
            ((("texcoords", "0", "data"),), tex0_data),
            ((("texcoords", "1", "mode"),), tex1_mode),
            ((("texcoords", "1", "data"),), tex1_data),
            ((("color", "mode"),), color_mode),
            ((("color", "data"),), color_data),
        ],
        "input_assembly",
    )

    t.arbiter.add(dut.bus)

    async def tb(ctx):
        await t.initialize_memory(ctx, addr, memory_data)
        await t.initialize_csrs(ctx)

    sim = Simulator(t.make(dut))
    sim.add_clock(1e-9)
    stream_testbench(
        sim,
        init_process=tb,
        input_stream=dut.is_index,
        input_data=input_idx,
        output_stream=dut.os_vertex,
        expected_output_data=expected,
        is_finished=dut.ready,
    )

    try:
        sim.run()
    except Exception:
        sim.reset()

        with sim.write_vcd(f"{test_name}.vcd", f"{test_name}.gtkw", traces=t.dut):
            sim.run()


def test_input_assembly_constant_only():
    make_test_input_assembly(
        test_name="test_input_assembly_constant_only",
        addr=0x80000000,
        memory_data=b"",
        input_idx=[0, 1, 2, 3, 4],
        expected=[
            {
                "position": [0.0, 0.0, 0.0, 1.0],
                "normal": [0.0, 0.0, 0.0],
                "texcoords": [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
                "color": [0.0, 0.0, 0.0, 1.0],
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
            "position": [0.0, 0.0, 0.0, 1.0],
            "normal": [0.0, 0.0, 0.0],
            "texcoords": [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
            "color": [0.0, 0.0, 0.0, 1.0],
        },
        {
            "position": [0.0, 0.0, 0.0, 1.0],
            "normal": [0.0, 0.0, 0.0],
            "texcoords": [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
            "color": [0.0, 0.0, 0.0, 1.0],
        },
    ]

    if comp == "texcoords[0]":
        expected[0]["texcoords"][0] = [1.0, 2.0, 3.0, 4.0]
        expected[1]["texcoords"][0] = [5.0, 6.0, 7.0, 8.0]
    elif comp == "texcoords[1]":
        expected[0]["texcoords"][1] = [1.0, 2.0, 3.0, 4.0]
        expected[1]["texcoords"][1] = [5.0, 6.0, 7.0, 8.0]
    elif comp == "normal":
        expected[0][comp] = [1.0, 2.0, 3.0]
        expected[1][comp] = [5.0, 6.0, 7.0]
    else:
        expected[0][comp] = [1.0, 2.0, 3.0, 4.0]
        expected[1][comp] = [5.0, 6.0, 7.0, 8.0]

    v = {
        f"{comp_in}_mode": InputMode.PER_VERTEX,
        f"{comp_in}_data": InputData.const(
            {
                "per_vertex": {
                    "address": 0x80000000,
                    "stride": 16 + separation,
                }
            }
        ),
    }

    vec1234_mem = Vector4_mem.const([1.0, 2.0, 3.0, 4.0])
    vec5678_mem = Vector4_mem.const([5.0, 6.0, 7.0, 8.0])

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
