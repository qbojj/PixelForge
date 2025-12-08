import pytest
from amaranth import *
from amaranth.sim import Simulator

from gpu.utils.math import FixedPointInv, FixedPointVecNormalize
from gpu.utils.types import FixedPoint, Vector3

from .streams import stream_testbench


@pytest.mark.parametrize(
    "data",
    [
        [[1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0]],
        [[3.0, 4.0, 0.0]],
        [[-3.0, -4.0, 0.0]],
        [[-3.0, 4.0, 0.0]],
        [[0.2, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [2.0, 0.0, 0.0]],
    ]
    + [[[float(i), 0.0, 0.0]] for i in range(1, 10)],
)
def test_normalize(data: list[list[float]]):
    expected = [[v / sum(comp**2 for comp in vec) ** 0.5 for v in vec] for vec in data]
    dut = FixedPointVecNormalize(Vector3)

    async def output_checker(ctx, results):
        results = [[v.as_float() for v in r] for r in results]
        print("Input data:", data)
        print("Checking output:", results)
        print("Expected data:", expected)
        print()

        def vec_dist(a, b):
            assert len(a) == len(b)
            return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

        assert len(results) == len(expected)
        assert all(vec_dist(r, e) < 1e-3 for r, e in zip(results, expected))

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    stream_testbench(
        sim,
        input_stream=dut.i,
        input_data=data,
        output_stream=dut.o,
        output_data_checker=output_checker,
        idle_for=30,
    )

    try:
        sim.run()
    except Exception:
        sim.reset()
        with sim.write_vcd(
            "test_fixed_point_vec_normalize.vcd",
            "test_fixed_point_vec_normalize.gtkw",
            traces=dut,
        ):
            sim.run()


@pytest.mark.parametrize(
    "data",
    [
        [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
        [-4.0, -2.0, -1.0, -0.5, -0.25],
        [0.1 * i for i in range(1, 11)],
        [10.0, 0.1],
        [f for f in range(-10, 11) if f != 0.0],
    ],
)
def test_inverse(data: list[float]):
    expected = [1.0 / v for v in data]

    dut = FixedPointInv(FixedPoint, steps=4)

    async def output_checker(ctx, results):
        results = [v.as_float() for v in results]
        print("Input data:", data)
        print("Checking output:", results)
        print("Expected data:", expected)
        print()

        def val_dist(a, b):
            return abs(a - b) / abs(b) < 1e-3

        assert len(results) == len(expected)
        assert all(val_dist(r, e) for r, e in zip(results, expected))

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    stream_testbench(
        sim,
        input_stream=dut.i,
        input_data=data,
        output_stream=dut.o,
        output_data_checker=output_checker,
        idle_for=300,
    )

    try:
        sim.run()
    except Exception:
        sim.reset()
        with sim.write_vcd(
            "test_fixed_point_inv.vcd",
            "test_fixed_point_inv.gtkw",
            traces=dut,
        ):
            sim.run()
