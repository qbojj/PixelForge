from amaranth import *
from amaranth.sim import Simulator

from gpu.input_assembly.cores import InputTopologyProcessor
from gpu.utils.types import InputTopology

from ..utils.streams import stream_testbench
from ..utils.testbench import SimpleTestbench


def make_test_input_topology_processor(
    test_name: str,
    input_topology: InputTopology,
    input: list[int],
    expected: list[int],
    restart_index: int | None = None,
    base_vertex: int = 0,
):
    dut = InputTopologyProcessor()
    t = SimpleTestbench(dut)

    async def init_tb(ctx):
        ctx.set(dut.c_input_topology, input_topology)
        ctx.set(dut.c_primitive_restart_enable, 1 if restart_index is not None else 0)
        ctx.set(dut.c_primitive_restart_index, restart_index or 0)
        ctx.set(dut.c_base_vertex, base_vertex)

    sim = Simulator(t)
    sim.add_clock(1e-9)
    stream_testbench(
        sim,
        init_process=init_tb,
        input_stream=dut.is_index,
        input_data=input,
        output_stream=dut.os_index,
        expected_output_data=expected,
        is_finished=dut.ready,
    )

    try:
        sim.run()
    except Exception:
        sim.reset()

        with sim.write_vcd(f"{test_name}.vcd", f"{test_name}.gtkw", traces=t.dut):
            sim.run()


def test_triangle_list():
    make_test_input_topology_processor(
        test_name="test_triangle_list",
        input_topology=InputTopology.TRIANGLE_LIST,
        input=[0, 1, 2, 3, 4, 5, 999, 134],
        expected=[0, 1, 2, 3, 4, 5],
    )


def test_base_vertex():
    make_test_input_topology_processor(
        test_name="test_base_vertex",
        input_topology=InputTopology.TRIANGLE_LIST,
        input=[0, 1, 2, 3, 4, 5],
        expected=[10, 11, 12, 13, 14, 15],
        base_vertex=10,
    )


def test_triangle_list_with_restart():
    make_test_input_topology_processor(
        test_name="test_triangle_list_with_restart",
        input_topology=InputTopology.TRIANGLE_LIST,
        input=[0, 1, 2, 0xFFFE, 3, 4, 5, 6, 0xFFFE, 7, 8, 9],
        expected=[0, 1, 2, 3, 4, 5, 7, 8, 9],
        restart_index=0xFFFE,
    )


def test_triangle_strip():
    make_test_input_topology_processor(
        test_name="test_triangle_strip",
        input_topology=InputTopology.TRIANGLE_STRIP,
        input=[0, 1, 2, 3, 4],
        expected=[0, 1, 2, 2, 1, 3, 3, 2, 4],
    )


def test_triangle_fan():
    make_test_input_topology_processor(
        test_name="test_triangle_fan",
        input_topology=InputTopology.TRIANGLE_FAN,
        input=[0, 1, 2, 3, 4],
        expected=[0, 1, 2, 0, 2, 3, 0, 3, 4],
    )
