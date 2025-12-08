from amaranth import *
from amaranth.sim import Simulator

from gpu.input_assembly.cores import IndexGenerator
from gpu.utils.types import IndexKind
from tests.utils.streams import stream_testbench
from tests.utils.testbench import SimpleTestbench


def make_test_index_generator(
    addr: int,
    count: int,
    kind: IndexKind,
    memory_data: bytes,
    expected: list[int],
):
    dut = IndexGenerator()
    t = SimpleTestbench(mem_addr=0x80000000, mem_size=1024)

    t.set_csrs(
        dut.csr_bus,
        [
            (("address",), C(addr, 32)),
            (("count",), C(count, 32)),
            (("kind",), kind),
            (("start",), 1),
        ],
        "index_gen",
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
        output_stream=dut.os_index,
        expected_output_data=expected,
        is_finished=dut.ready,
    )

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


test_indexed_u32()


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
        addr=0x80000001,
        count=8,
        kind=IndexKind.U8,
        memory_data=b"".join(
            (i.to_bytes(1, "little") for i in [1, 2, 3, 4, 5, 6, 7, 8])
        ),
        expected=[1, 2, 3, 4, 5, 6, 7, 8],
    )


def test_indexed_u16_unaligned():
    make_test_index_generator(
        addr=0x80000002,
        count=6,
        kind=IndexKind.U16,
        memory_data=b"".join((i.to_bytes(2, "little") for i in [2, 3, 4, 5, 1, 0])),
        expected=[2, 3, 4, 5, 1, 0],
    )
