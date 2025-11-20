from amaranth import *
from amaranth.lib import stream
from amaranth.sim import SimulatorContext


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
