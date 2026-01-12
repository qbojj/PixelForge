"""End-to-end per-fragment pipeline integration tests."""

import pytest
from amaranth import Elaboratable, Module
from amaranth.lib import stream, wiring
from amaranth.sim import Simulator

from gpu.pixel_shading import (
    BlendFactor,
    BlendOp,
    DepthStencilTest,
    StencilOp,
    SwapchainOutput,
)
from gpu.utils.layouts import FragmentLayout, num_textures
from gpu.utils.types import CompareOp

from ..utils.streams import stream_testbench
from ..utils.testbench import SimpleTestbench


def make_fragment(x, y, depth, color, front_facing=1):
    return {
        "depth": depth,
        "texcoords": [[0.0, 0.0, 0.0, 1.0] for _ in range(num_textures)],
        "color": color,
        "coord_pos": [x, y],
        "front_facing": front_facing,
    }


BGRA_SWIZZLE = [2, 1, 0, 3]


def make_fb_info(base_addr=0):
    width = 4
    height = 4
    color_pitch = width * 4  # BGRA8
    depthstencil_pitch = width * 4  # D16_X8_S8 combined buffer

    return {
        "width": width,
        "height": height,
        "viewport_x": 0.0,
        "viewport_y": 0.0,
        "viewport_width": float(width),
        "viewport_height": float(height),
        "viewport_min_depth": 0.0,
        "viewport_max_depth": 1.0,
        "scissor_offset_x": 0,
        "scissor_offset_y": 0,
        "scissor_width": width,
        "scissor_height": height,
        "color_address": base_addr,
        "color_pitch": color_pitch,
        "depthstencil_address": base_addr + 0x100,
        "depthstencil_pitch": depthstencil_pitch,
    }


class PerPixelPipeline(Elaboratable):
    """Depth+stencil test followed by swapchain output."""

    def __init__(self):
        self.dst = DepthStencilTest()
        self.swp = SwapchainOutput()

        self.is_fragment = stream.Signature(FragmentLayout).flip().create()

        self.fb_info = self.dst.fb_info
        self.stencil_conf_front = self.dst.stencil_conf_front
        self.stencil_conf_back = self.dst.stencil_conf_back
        self.depth_conf = self.dst.depth_conf
        self.blend_conf = self.swp.conf

        self.wb_bus_depth = self.dst.wb_bus
        self.wb_bus_swap = self.swp.wb_bus

    def elaborate(self, platform):
        m = Module()
        m.submodules.dst = self.dst
        m.submodules.swp = self.swp

        wiring.connect(m, wiring.flipped(self.is_fragment), self.dst.is_fragment)
        wiring.connect(m, self.dst.os_fragment, self.swp.is_fragment)
        m.d.comb += self.swp.fb_info.eq(self.fb_info)

        return m


def test_per_pixel_pipeline_end_to_end():
    pipeline = PerPixelPipeline()
    t = SimpleTestbench(pipeline, mem_size=4096, mem_addr=0)
    t.arbiter.add(pipeline.wb_bus_depth)
    t.arbiter.add(pipeline.wb_bus_swap)

    fb_info = make_fb_info()

    stencil_conf = {
        "compare_op": CompareOp.ALWAYS,
        "pass_op": StencilOp.INCR_WRAP,
        "fail_op": StencilOp.KEEP,
        "depth_fail_op": StencilOp.DECR_WRAP,
        "reference": 0,
        "mask": 0xFF,
        "write_mask": 0xFF,
    }

    depth_conf = {
        "test_enabled": 1,
        "write_enabled": 1,
        "compare_op": CompareOp.GREATER_OR_EQUAL,
    }

    blend_conf = {
        "src_factor": BlendFactor.ONE,
        "dst_factor": BlendFactor.ZERO,
        "src_a_factor": BlendFactor.ONE,
        "dst_a_factor": BlendFactor.ZERO,
        "enabled": 0,
        "blend_op": BlendOp.ADD,
        "blend_a_op": BlendOp.ADD,
        "color_write_mask": 0xF,
    }

    frag1 = make_fragment(0, 0, 0.4, [1.0, 0.5, 0.25, 1.0])
    frag2 = make_fragment(0, 0, 0.3, [0.2, 0.8, 0.2, 0.5])
    fragments = [frag1, frag2]

    expected_color = [1.0, 0.5, 0.25, 1.0]
    expected_depth = 0.4

    sim = Simulator(t)
    sim.add_clock(1e-6)

    async def init_proc(ctx):
        await t.initialize_memory(ctx, fb_info["color_address"], b"\x00" * 64)
        # Initialize combined depth/stencil buffer
        await t.initialize_memory(ctx, fb_info["depthstencil_address"], b"\x00" * 64)

        ctx.set(pipeline.fb_info, fb_info)
        ctx.set(pipeline.stencil_conf_front, stencil_conf)
        ctx.set(pipeline.stencil_conf_back, stencil_conf)
        ctx.set(pipeline.depth_conf, depth_conf)
        ctx.set(pipeline.blend_conf, blend_conf)

    async def final_checker(ctx):
        color_bytes = await t.dbg_access.read_bytes(ctx, fb_info["color_address"], 4)
        # Convert BGRA bytes back to RGBA for comparison
        bgra_values = [c / 255.0 for c in color_bytes]
        color_values = [bgra_values[2], bgra_values[1], bgra_values[0], bgra_values[3]]
        assert color_values == pytest.approx(expected_color, abs=1 / 255)
        # Depth is lower 16 bits of combined 32-bit word
        depth_bytes = await t.dbg_access.read_bytes(
            ctx, fb_info["depthstencil_address"], 2
        )
        depth_value = (int.from_bytes(depth_bytes, "little") / 65535.0) * 2.0 - 1.0
        assert depth_value == pytest.approx(expected_depth, abs=1 / 65535)
        # Stencil is the upper byte of combined word (offset +3)
        stencil_bytes = await t.dbg_access.read_bytes(
            ctx, fb_info["depthstencil_address"] + 3, 1
        )
        assert stencil_bytes[0] == 0  # increment then decrement

    stream_testbench(
        sim,
        input_stream=pipeline.is_fragment,
        input_data=fragments,
        init_process=init_proc,
        final_checker=final_checker,
        wait_after_supposed_finish=200,
    )

    sim.run()
