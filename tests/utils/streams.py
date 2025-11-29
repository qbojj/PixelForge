from typing import Callable

from amaranth import *
from amaranth.lib import stream
from amaranth.sim import Simulator, SimulatorContext


async def stream_get(
    ctx: SimulatorContext, stream: stream.Interface, finish: Value
) -> list:
    results = []

    while not ctx.get(finish):
        await ctx.tick().until(stream.valid | finish)

        while ctx.get(stream.valid):
            results.append(ctx.get(stream.payload))
            ctx.set(stream.ready, 1)
            await ctx.tick()
            ctx.set(stream.ready, 0)

    return results


async def stream_put(
    ctx: SimulatorContext, stream: stream.Interface, data: list
) -> None:
    for item in data:
        ctx.set(stream.payload, item)
        ctx.set(stream.valid, 1)
        await ctx.tick().until(stream.ready)
        ctx.set(stream.valid, 0)


async def idle_cycles(ctx: SimulatorContext, cycles: int, event: Value) -> None:
    idle_count = 0
    while idle_count < cycles:
        await ctx.tick()
        if ctx.get(event):
            idle_count = 0
        else:
            idle_count += 1


def data_checker(expected):
    async def fn(ctx, results):
        print("Checking output:", results)
        print("Expected data:", expected)
        assert ctx.get(results == expected), "Output data does not match expected data"

    return fn


def stream_testbench(
    sim: Simulator,
    input_stream: stream.Interface | None = None,
    input_data: list | None = None,
    output_stream: stream.Interface | None = None,
    expected_output_data: list | None = None,
    output_data_checker: Callable | None = None,
    is_finished: Value = C(1),
    init_process: Callable | None = None,
    wait_after_supposed_finish: int | None = None,
    idle_for: int | None = None,
) -> None:
    if input_data is not None or input_stream is not None:
        assert (
            input_data is not None and input_stream is not None
        ), "Both input_stream and input_data must be provided to send data."

    if output_stream is not None:
        if expected_output_data is None and output_data_checker is None:
            raise ValueError(
                "Either expected_output_data or output_data_checker must be provided "
                "to verify output data."
            )

        # only one of the two can be provided
        if expected_output_data is not None and output_data_checker is not None:
            raise ValueError(
                "Only one of expected_output_data or output_data_checker can be provided."
            )

        if expected_output_data is not None:
            output_data_checker = data_checker(expected_output_data)

    is_initialized = Signal(1) if init_process is not None else C(1)
    all_data_sent = Signal(1) if input_data is not None else C(1)
    waited = Signal(1) if wait_after_supposed_finish is not None else C(1)
    idled = Signal(1) if idle_for is not None else C(1)

    stop_reading = is_finished & all_data_sent & waited & idled

    async def wait_tb(ctx: SimulatorContext):
        await ctx.tick().until(is_finished & all_data_sent)
        await ctx.tick().repeat(wait_after_supposed_finish)
        ctx.set(waited, 1)

    async def idle_tb(ctx: SimulatorContext):
        await idle_cycles(ctx, idle_for, output_stream.valid)
        ctx.set(idled, 1)

    async def input_tb(ctx: SimulatorContext):
        await ctx.tick().until(is_initialized)
        await stream_put(ctx, input_stream, input_data)
        ctx.set(all_data_sent, 1)

    async def output_tb(ctx: SimulatorContext):
        await ctx.tick().until(is_initialized)
        results = await stream_get(ctx, output_stream, stop_reading)
        await output_data_checker(ctx, results)

    async def init_tb(ctx: SimulatorContext):
        await init_process(ctx)
        ctx.set(is_initialized, 1)

    if input_data is not None:
        sim.add_testbench(input_tb)

    if output_stream is not None:
        sim.add_testbench(output_tb)

    if init_process is not None:
        sim.add_testbench(init_tb)

    if wait_after_supposed_finish is not None:
        sim.add_testbench(wait_tb)

    if idle_for is not None and output_stream is not None:
        sim.add_testbench(idle_tb)
