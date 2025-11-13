from amaranth import *
from amaranth.sim import Simulator

from gpu.utils.math import FixedPointVecNormalize
from gpu.utils.types import FixedPoint, Vector3

rst = Signal(1)
dut = ResetInserter({"sync": rst})(FixedPointVecNormalize(Vector3))


async def tb_operation(data, expected, ctx):
    ctx.set(rst, 1)
    ctx.set(dut.start, 0)
    await ctx.tick()
    ctx.set(rst, 0)
    await ctx.tick()

    print()
    print(f"Testing with input: {data}")

    ctx.set(dut.value, [FixedPoint.from_float_const(v) for v in data])
    ctx.set(dut.start, 1)

    await ctx.tick()
    ctx.set(dut.start, 0)

    print("Waiting for ready signal...")

    cycles = 0
    while not ctx.get(dut.ready):
        await ctx.tick()
        cycles += 1

    result = ctx.get(dut.result)
    result = [ctx.get(r.data) / (1 << FixedPoint.lo_bits) for r in result]
    print(f"got {result}; expected {expected} ({cycles=})")

    if not all(abs(r - e) < 1e-1 for r, e in zip(result, expected)):
        raise AssertionError("Test failed!")


async def tb_normalize(ctx):
    # Test cases: (input_vector, expected_normalized_vector)
    test_cases = []

    # Unit vectors are already normalized
    test_cases.append(([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]))

    test_cases.append(([0.0, 1.0, 0.0], [0.0, 1.0, 0.0]))

    test_cases.append(([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]))

    # 3-4-5 triangle
    test_cases.append(([3.0, 4.0, 0.0], [0.6, 0.8, 0.0]))

    # multiple single value vectors
    for i in range(1, 10):
        test_cases.append(([float(i), 0.0, 0.0], [1.0, 0.0, 0.0]))

    for data, expected in test_cases:
        await tb_operation(data, expected, ctx)
        await ctx.tick().repeat(5)


def test_normalize():
    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(tb_normalize)
    sim.run()
