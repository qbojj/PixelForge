"""Unit tests for the pixel shading building blocks."""

import pytest
from amaranth.sim import Simulator

from gpu.pixel_shading import (
    BlendFactor,
    BlendOp,
    DepthStencilTest,
    StencilOp,
    SwapchainOutput,
)
from gpu.utils.types import CompareOp

from ..utils.streams import stream_testbench
from ..utils.testbench import SimpleTestbench


def make_fragment(x, y, depth, color, front_facing=1):
    return {
        "depth": depth,
        "texcoords": [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        "color": color,
        "coord_pos": [x, y],
        "front_facing": front_facing,
    }


def make_fb_info(base_addr=0):
    width = 8
    height = 8
    color_pitch = width * 4
    depth_pitch = width * 2
    stencil_pitch = width

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
        "depth_address": base_addr + 0x100,
        "depth_pitch": depth_pitch,
        "stencil_address": base_addr + 0x200,
        "stencil_pitch": stencil_pitch,
    }


def test_depth_stencil_pass_and_depth_write():
    dut = DepthStencilTest()
    t = SimpleTestbench(dut, mem_size=4096, mem_addr=0)
    t.arbiter.add(dut.wb_bus)

    fb_info = make_fb_info()

    stencil_conf = {
        "compare_op": CompareOp.ALWAYS,
        "pass_op": StencilOp.INCR,
        "fail_op": StencilOp.KEEP,
        "depth_fail_op": StencilOp.KEEP,
        "reference": 0,
        "mask": 0xFF,
        "write_mask": 0xFF,
    }

    depth_conf = {
        "test_enabled": 1,
        "write_enabled": 1,
        "compare_op": CompareOp.GREATER_OR_EQUAL,
    }

    fragments = [make_fragment(0, 0, 0.75, [0.2, 0.2, 0.2, 1.0])]
    expected_depth = int((1.0 + 0.75) / 2.0 * ((1 << 16) - 1))

    sim = Simulator(t)
    sim.add_clock(1e-6)

    async def init_proc(ctx):
        ctx.set(t.dut.fb_info, fb_info)
        ctx.set(t.dut.stencil_conf_front, stencil_conf)
        ctx.set(t.dut.stencil_conf_back, stencil_conf)
        ctx.set(t.dut.depth_conf, depth_conf)

    async def check_output(ctx, results):
        assert len(results) == len(fragments)
        depth_bytes = await t.dbg_access.read_bytes(ctx, fb_info["depth_address"], 2)
        stencil_byte = await t.dbg_access.read_bytes(ctx, fb_info["stencil_address"], 1)
        stored_depth = int.from_bytes(depth_bytes, "little")
        stored_stencil = stencil_byte[0]

        assert stored_stencil == 1  # Stencil incremented
        assert stored_depth == expected_depth

    stream_testbench(
        sim,
        input_stream=dut.is_fragment,
        input_data=fragments,
        output_stream=dut.os_fragment,
        output_data_checker=check_output,
        init_process=init_proc,
        idle_for=1000,
    )

    sim.run()


def test_depth_stencil_depth_fail():
    """Test that fragments fail depth test and don't update depth buffer."""
    dut = DepthStencilTest()
    t = SimpleTestbench(dut, mem_size=4096, mem_addr=0)
    t.arbiter.add(dut.wb_bus)

    fb_info = make_fb_info()

    # Pre-fill depth buffer with 0.5
    initial_depth = round((0.5 + 1.0) / 2.0 * ((1 << 16) - 1))

    stencil_conf = {
        "compare_op": CompareOp.ALWAYS,
        "pass_op": StencilOp.KEEP,
        "fail_op": StencilOp.KEEP,
        "depth_fail_op": StencilOp.DECR,  # Should execute on depth fail
        "reference": 0,
        "mask": 0xFF,
        "write_mask": 0xFF,
    }

    depth_conf = {
        "test_enabled": 1,
        "write_enabled": 1,
        "compare_op": CompareOp.GREATER_OR_EQUAL,  # Fragment must be >= stored depth
    }

    # Fragment with depth 0.3, which is less than stored 0.5, should fail
    fragments = [make_fragment(0, 0, 0.3, [0.2, 0.2, 0.2, 1.0])]

    sim = Simulator(t)
    sim.add_clock(1e-6)

    async def init_proc(ctx):
        # Pre-fill depth buffer
        await t.initialize_memory(
            ctx, fb_info["depth_address"], initial_depth.to_bytes(2, "little")
        )
        # Pre-fill stencil with 5
        await t.initialize_memory(ctx, fb_info["stencil_address"], b"\x05")

        ctx.set(t.dut.fb_info, fb_info)
        ctx.set(t.dut.stencil_conf_front, stencil_conf)
        ctx.set(t.dut.stencil_conf_back, stencil_conf)
        ctx.set(t.dut.depth_conf, depth_conf)

    async def check_output(ctx, results):
        # Fragment should be rejected, no output
        assert len(results) == 0

        # Depth should remain unchanged
        depth_bytes_read = await t.dbg_access.read_bytes(
            ctx, fb_info["depth_address"], 2
        )
        stored_depth = int.from_bytes(depth_bytes_read, "little")
        assert stored_depth == initial_depth

        # Stencil should be decremented (depth_fail_op)
        stencil_byte = await t.dbg_access.read_bytes(ctx, fb_info["stencil_address"], 1)
        assert stencil_byte[0] == 4

    stream_testbench(
        sim,
        input_stream=dut.is_fragment,
        input_data=fragments,
        output_stream=dut.os_fragment,
        output_data_checker=check_output,
        init_process=init_proc,
        idle_for=1000,
    )

    sim.run()


def test_depth_stencil_stencil_fail():
    """Test stencil test failure."""
    dut = DepthStencilTest()
    t = SimpleTestbench(dut, mem_size=4096, mem_addr=0)
    t.arbiter.add(dut.wb_bus)

    fb_info = make_fb_info()

    stencil_conf = {
        "compare_op": CompareOp.EQUAL,  # Will fail: stored is 0, reference is 5
        "pass_op": StencilOp.KEEP,
        "fail_op": StencilOp.INCR,
        "depth_fail_op": StencilOp.KEEP,
        "reference": 5,
        "mask": 0xFF,
        "write_mask": 0xFF,
    }

    depth_conf = {
        "test_enabled": 1,
        "write_enabled": 0,
        "compare_op": CompareOp.ALWAYS,
    }

    fragments = [make_fragment(0, 0, 0.5, [0.5, 0.5, 0.5, 1.0])]

    sim = Simulator(t)
    sim.add_clock(1e-6)

    async def init_proc(ctx):
        ctx.set(t.dut.fb_info, fb_info)
        ctx.set(t.dut.stencil_conf_front, stencil_conf)
        ctx.set(t.dut.stencil_conf_back, stencil_conf)
        ctx.set(t.dut.depth_conf, depth_conf)

    async def check_output(ctx, results):
        # Fragment should fail stencil test
        assert len(results) == 0

        # Stencil should be incremented (fail_op)
        stencil_byte = await t.dbg_access.read_bytes(ctx, fb_info["stencil_address"], 1)
        assert stencil_byte[0] == 1

    stream_testbench(
        sim,
        input_stream=dut.is_fragment,
        input_data=fragments,
        output_stream=dut.os_fragment,
        output_data_checker=check_output,
        init_process=init_proc,
        idle_for=1000,
    )

    sim.run()


def test_depth_stencil_depth_never():
    """Test NEVER depth compare operation."""
    dut = DepthStencilTest()
    t = SimpleTestbench(dut, mem_size=4096, mem_addr=0)
    t.arbiter.add(dut.wb_bus)

    fb_info = make_fb_info()

    stencil_conf = {
        "compare_op": CompareOp.ALWAYS,
        "pass_op": StencilOp.KEEP,
        "fail_op": StencilOp.KEEP,
        "depth_fail_op": StencilOp.KEEP,
        "reference": 0,
        "mask": 0xFF,
        "write_mask": 0,
    }

    depth_conf = {
        "test_enabled": 1,
        "write_enabled": 0,
        "compare_op": CompareOp.NEVER,  # Always fails
    }

    fragments = [make_fragment(0, 0, 0.9, [0.9, 0.9, 0.9, 1.0])]

    sim = Simulator(t)
    sim.add_clock(1e-6)

    async def init_proc(ctx):
        ctx.set(t.dut.fb_info, fb_info)
        ctx.set(t.dut.stencil_conf_front, stencil_conf)
        ctx.set(t.dut.stencil_conf_back, stencil_conf)
        ctx.set(t.dut.depth_conf, depth_conf)

    async def check_output(ctx, results):
        # Fragment should always fail with NEVER
        assert len(results) == 0

    stream_testbench(
        sim,
        input_stream=dut.is_fragment,
        input_data=fragments,
        output_stream=dut.os_fragment,
        output_data_checker=check_output,
        init_process=init_proc,
        idle_for=1000,
    )

    sim.run()


def test_depth_stencil_stencil_replace():
    """Test stencil REPLACE operation on pass."""
    dut = DepthStencilTest()
    t = SimpleTestbench(dut, mem_size=4096, mem_addr=0)
    t.arbiter.add(dut.wb_bus)

    fb_info = make_fb_info()

    stencil_conf = {
        "compare_op": CompareOp.ALWAYS,
        "pass_op": StencilOp.REPLACE,
        "fail_op": StencilOp.KEEP,
        "depth_fail_op": StencilOp.KEEP,
        "reference": 42,
        "mask": 0xFF,
        "write_mask": 0xFF,
    }

    depth_conf = {
        "test_enabled": 0,
        "write_enabled": 0,
        "compare_op": CompareOp.ALWAYS,
    }

    fragments = [make_fragment(0, 0, 0.5, [0.5, 0.5, 0.5, 1.0])]

    sim = Simulator(t)
    sim.add_clock(1e-6)

    async def init_proc(ctx):
        # Pre-fill stencil with 0xFF
        await t.initialize_memory(ctx, fb_info["stencil_address"], b"\xff")

        ctx.set(t.dut.fb_info, fb_info)
        ctx.set(t.dut.stencil_conf_front, stencil_conf)
        ctx.set(t.dut.stencil_conf_back, stencil_conf)
        ctx.set(t.dut.depth_conf, depth_conf)

    async def check_output(ctx, results):
        assert len(results) == 1

        # Stencil should be replaced with reference value
        stencil_byte = await t.dbg_access.read_bytes(ctx, fb_info["stencil_address"], 1)
        assert stencil_byte[0] == 42

    stream_testbench(
        sim,
        input_stream=dut.is_fragment,
        input_data=fragments,
        output_stream=dut.os_fragment,
        output_data_checker=check_output,
        init_process=init_proc,
        idle_for=1000,
    )

    sim.run()


def test_depth_stencil_stencil_write_mask():
    """Test that stencil write mask correctly masks write bits."""
    dut = DepthStencilTest()
    t = SimpleTestbench(dut, mem_size=4096, mem_addr=0)
    t.arbiter.add(dut.wb_bus)

    fb_info = make_fb_info()

    stencil_conf = {
        "compare_op": CompareOp.ALWAYS,
        "pass_op": StencilOp.REPLACE,
        "fail_op": StencilOp.KEEP,
        "depth_fail_op": StencilOp.KEEP,
        "reference": 0xFF,  # Try to set all bits
        "mask": 0xFF,
        "write_mask": 0x0F,  # Only write lower 4 bits
    }

    depth_conf = {
        "test_enabled": 1,
        "write_enabled": 0,
        "compare_op": CompareOp.ALWAYS,
    }

    fragments = [make_fragment(0, 0, 0.5, [0.5, 0.5, 0.5, 1.0])]

    sim = Simulator(t)
    sim.add_clock(1e-6)

    async def init_proc(ctx):
        # Pre-fill stencil with 0xF0
        await t.initialize_memory(ctx, fb_info["stencil_address"], b"\xf0")

        ctx.set(t.dut.fb_info, fb_info)
        ctx.set(t.dut.stencil_conf_front, stencil_conf)
        ctx.set(t.dut.stencil_conf_back, stencil_conf)
        ctx.set(t.dut.depth_conf, depth_conf)

    async def check_output(ctx, results):
        assert len(results) == 1

        # Should have written 0x0F into lower bits, keeping upper bits as 0xF0
        stencil_byte = await t.dbg_access.read_bytes(ctx, fb_info["stencil_address"], 1)
        assert stencil_byte[0] == 0xFF  # 0xF0 | 0x0F = 0xFF

    stream_testbench(
        sim,
        input_stream=dut.is_fragment,
        input_data=fragments,
        output_stream=dut.os_fragment,
        output_data_checker=check_output,
        init_process=init_proc,
        idle_for=1000,
    )


# Test cases for SwapchainOutput blending operations
SWAPCHAIN_TEST_CASES = [
    pytest.param(
        {
            "src_factor": BlendFactor.ONE,
            "dst_factor": BlendFactor.ZERO,
            "src_a_factor": BlendFactor.ONE,
            "dst_a_factor": BlendFactor.ZERO,
            "enabled": 0,
            "blend_op": BlendOp.ADD,
            "blend_a_op": BlendOp.ADD,
            "color_write_mask": 0xF,
        },
        [make_fragment(0, 0, 0.2, [1.0, 0.5, 0.25, 1.0])],
        [1.0, 0.5, 0.25, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        id="unblended_color",
    ),
    pytest.param(
        {
            "src_factor": BlendFactor.SRC_ALPHA,
            "dst_factor": BlendFactor.ONE_MINUS_SRC_ALPHA,
            "src_a_factor": BlendFactor.ONE,
            "dst_a_factor": BlendFactor.ZERO,
            "enabled": 1,
            "blend_op": BlendOp.ADD,
            "blend_a_op": BlendOp.ADD,
            "color_write_mask": 0xF,
        },
        [make_fragment(0, 0, 0.5, [1.0, 0.5, 0.6, 0.75])],
        [1.0, 0.375, 0.7, 0.75],
        [1.0, 0.0, 1.0, 0.0],
        id="alpha_blending",
    ),
    pytest.param(
        {
            "src_factor": BlendFactor.ONE,
            "dst_factor": BlendFactor.ONE,
            "src_a_factor": BlendFactor.ONE,
            "dst_a_factor": BlendFactor.ONE,
            "enabled": 1,
            "blend_op": BlendOp.ADD,
            "blend_a_op": BlendOp.ADD,
            "color_write_mask": 0xF,
        },
        [make_fragment(0, 0, 0.5, [0.25, 0.25, 0.25, 0.5])],
        [0.75, 0.75, 0.75, 1.0],
        [0.5, 0.5, 0.5, 0.5],
        id="additive_blending",
    ),
    pytest.param(
        {
            "src_factor": BlendFactor.DST_ALPHA,
            "dst_factor": BlendFactor.ZERO,
            "src_a_factor": BlendFactor.ONE,
            "dst_a_factor": BlendFactor.ZERO,
            "enabled": 1,
            "blend_op": BlendOp.ADD,
            "blend_a_op": BlendOp.ADD,
            "color_write_mask": 0xF,
        },
        [make_fragment(0, 0, 0.5, [0.5, 1.0, 0.5, 1.0])],
        [0.4, 0.8, 0.4, 1.0],
        [0.2, 0.4, 0.6, 0.8],
        id="multiply_blending",
    ),
    pytest.param(
        {
            "src_factor": BlendFactor.ONE,
            "dst_factor": BlendFactor.ONE,
            "src_a_factor": BlendFactor.ONE,
            "dst_a_factor": BlendFactor.ZERO,
            "enabled": 1,
            "blend_op": BlendOp.SUBTRACT,
            "blend_a_op": BlendOp.ADD,
            "color_write_mask": 0xF,
        },
        [make_fragment(0, 0, 0.5, [0.8, 0.6, 0.4, 1.0])],
        [0.2, 0.4, 0.4, 1.0],
        [0.6, 0.2, 0.0, 0.0],
        id="subtract_blending",
    ),
    pytest.param(
        {
            "src_factor": BlendFactor.ONE,
            "dst_factor": BlendFactor.ONE,
            "src_a_factor": BlendFactor.ONE,
            "dst_a_factor": BlendFactor.ONE,
            "enabled": 1,
            "blend_op": BlendOp.MIN,
            "blend_a_op": BlendOp.MIN,
            "color_write_mask": 0xF,
        },
        [make_fragment(0, 0, 0.5, [0.8, 0.3, 0.6, 0.9])],
        [0.6, 0.3, 0.4, 0.8],
        [0.6, 0.7, 0.4, 0.8],
        id="min_blending",
    ),
    pytest.param(
        {
            "src_factor": BlendFactor.ONE,
            "dst_factor": BlendFactor.ONE,
            "src_a_factor": BlendFactor.ONE,
            "dst_a_factor": BlendFactor.ONE,
            "enabled": 1,
            "blend_op": BlendOp.MAX,
            "blend_a_op": BlendOp.MAX,
            "color_write_mask": 0xF,
        },
        [make_fragment(0, 0, 0.5, [0.3, 0.7, 0.5, 0.4])],
        [0.6, 0.7, 0.5, 0.8],
        [0.6, 0.2, 0.1, 0.8],
        id="max_blending",
    ),
]


@pytest.mark.parametrize(
    "blend_conf,fragments,expected_color,dst_color",
    SWAPCHAIN_TEST_CASES,
)
def test_swapchain_output(blend_conf, fragments, expected_color, dst_color):
    """Test SwapchainOutput with various blending operations."""
    dut = SwapchainOutput()
    t = SimpleTestbench(dut, mem_size=4096, mem_addr=0)
    t.arbiter.add(dut.wb_bus)
    fb_info = make_fb_info()

    sim = Simulator(t)
    sim.add_clock(1e-6)

    async def init_proc(ctx):
        color_bytes = bytes([int(c * 255) for c in dst_color])
        await t.initialize_memory(ctx, fb_info["color_address"], color_bytes)
        ctx.set(t.dut.fb_info, fb_info)
        ctx.set(t.dut.conf, blend_conf)

    async def verify_memory(ctx):
        color_bytes = await t.dbg_access.read_bytes(ctx, fb_info["color_address"], 4)

        color_values = [int(b) / 255.0 for b in color_bytes]
        expected_values = expected_color

        assert color_values == pytest.approx(
            expected_values, abs=1 / 255
        ), f"Final color mismatch: got {color_values}, expected {expected_values}"

    stream_testbench(
        sim,
        input_stream=dut.is_fragment,
        input_data=fragments,
        init_process=init_proc,
        final_checker=verify_memory,
        wait_after_supposed_finish=1000,
    )

    sim.run()
