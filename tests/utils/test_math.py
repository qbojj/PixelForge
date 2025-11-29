import pytest
from amaranth import *
from amaranth.sim import Simulator

from gpu.utils.math import FixedPointVecNormalize
from gpu.utils.types import Vector3

from .streams import stream_testbench


@pytest.mark.parametrize(
    "data, expected",
    [
        ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        ([0.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]),
        ([3.0, 4.0, 0.0], [0.6, 0.8, 0.0]),
        ([-3.0, -4.0, 0.0], [-0.6, -0.8, 0.0]),
        ([-3.0, 4.0, 0.0], [-0.6, 0.8, 0.0]),
        ([0.2, 0.0, 0.0], [1.0, 0.0, 0.0]),
    ]
    + [([float(i), 0.0, 0.0], [1.0, 0.0, 0.0]) for i in range(1, 10)],
)
def test_normalize(data: list[float], expected: list[float]):
    dut = FixedPointVecNormalize(Vector3, inv_sqrt_steps=6)

    async def output_checker(ctx, results):
        results = [[v.as_float() for v in r] for r in results]
        print("Input data:", data)
        print("Checking output:", results)
        print("Expected data:", [expected])
        print()

        assert len(results) == 1
        assert all(abs(r - e) < 1e-3 for r, e in zip(results[0], expected))

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    stream_testbench(
        sim,
        input_stream=dut.i,
        input_data=[Vector3.const(data)],
        output_stream=dut.o,
        output_data_checker=output_checker,
        idle_for=100,
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
