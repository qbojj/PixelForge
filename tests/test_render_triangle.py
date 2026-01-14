"""
Integration test: Render a single RGB triangle using GraphicsPipeline.

This test mimics the --render-triangle functionality from pixelforge_demo.c,
driving the GraphicsPipeline component directly in simulation.
"""

import struct

import pytest
from amaranth import *
from amaranth.sim import Simulator

from gpu.input_assembly.layouts import InputData, InputMode
from gpu.pipeline import GraphicsPipeline
from gpu.pixel_shading.cores import BlendFactor, BlendOp, StencilOp
from gpu.utils import fixed
from gpu.utils.types import (
    CompareOp,
    CullFace,
    FrontFace,
    IndexKind,
    InputTopology,
    PrimitiveType,
)
from tests.utils.testbench import SimpleTestbench


def fp16_16(v: float) -> int:
    """Convert float to signed 16.16 fixed-point integer."""
    return int(v * 65536)


def setup_triangle_geometry(vb_mem_addr: int) -> tuple:
    """
    Create triangle geometry in memory format.

    Returns:
        (idx_addr, idx_count, pos_addr, norm_addr, col_addr, stride, memory_data)
    """
    # Triangle vertices: red (bottom-left), green (bottom-right), blue (top)
    vertices = [
        {  # Vertex 0
            "pos": [-0.7, -0.7, 0.2, 1.0],
            "norm": [0.0, 0.0, 1.0],
            "col": [1.0, 0.0, 0.0, 1.0],  # red
        },
        {  # Vertex 1
            "pos": [0.7, -0.7, 0.2, 1.0],
            "norm": [0.0, 0.0, 1.0],
            "col": [0.0, 1.0, 0.0, 1.0],  # green
        },
        {  # Vertex 2
            "pos": [0.0, 0.7, 0.2, 1.0],
            "norm": [0.0, 0.0, 1.0],
            "col": [0.0, 0.0, 1.0, 1.0],  # blue
        },
    ]

    # Pack vertices: position (4x4 bytes), normal (3x4 bytes), color (4x4 bytes)
    # Total stride = 44 bytes per vertex
    stride = 4 * 4 + 3 * 4 + 4 * 4  # 44 bytes

    pos_offset = 0
    norm_offset = 16
    col_offset = 28

    vb_data = bytearray()
    for v in vertices:
        # Position
        for val in v["pos"]:
            vb_data.extend(struct.pack("<i", int(val * 65536)))
        # Normal
        for val in v["norm"]:
            vb_data.extend(struct.pack("<i", int(val * 65536)))
        # Color
        for val in v["col"]:
            vb_data.extend(struct.pack("<i", int(val * 65536)))

    # Index buffer (3 indices for triangle, u16 format)
    idx_data = struct.pack("<HHH", 0, 1, 2)

    # Place index buffer after vertex data
    idx_offset = len(vb_data)

    # Combine into single memory blob
    memory_data = vb_data + idx_data

    pos_addr = vb_mem_addr + pos_offset
    norm_addr = vb_mem_addr + norm_offset
    col_addr = vb_mem_addr + col_offset
    idx_addr = vb_mem_addr + idx_offset
    idx_count = 3

    return (idx_addr, idx_count, pos_addr, norm_addr, col_addr, stride, memory_data)


def save_ppm_image(filename: str, width: int, height: int, data: bytes):
    """Save image data as PPM format."""
    with open(filename, "wb") as f:
        # PPM header
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        # change BGR to RGB
        for i in range(0, len(data), 3):
            b, g, r = data[i : i + 3]
            f.write(bytes([r, g, b]))
        # Write pixel data (RGB)


def save_depth_image(filename: str, width: int, height: int, depth_data: bytes):
    """Save depth buffer as grayscale PPM image."""
    with open(filename, "wb") as f:
        # PPM header for grayscale
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        # Convert depth to grayscale RGB
        for i in range(0, len(depth_data), 4):
            # Read 32-bit depth value (D16_X8_S8 format: depth in bits 16-31)
            pixel = struct.unpack("<I", depth_data[i : i + 4])[0]
            depth16 = (pixel >> 16) & 0xFFFF
            # Convert to 8-bit grayscale
            gray = int((depth16 / 65535.0) * 255)
            # Write as RGB (all channels same)
            f.write(bytes([gray, gray, gray]))


def save_stencil_image(filename: str, width: int, height: int, depth_data: bytes):
    """Save stencil buffer as grayscale PPM image."""
    with open(filename, "wb") as f:
        # PPM header for grayscale
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        # Convert stencil to grayscale RGB
        for i in range(0, len(depth_data), 4):
            # Read 32-bit value (D16_X8_S8 format: stencil in bits 0-7)
            pixel = struct.unpack("<I", depth_data[i : i + 4])[0]
            stencil8 = pixel & 0xFF
            # Stencil is already 8-bit
            # Write as RGB (all channels same)
            f.write(bytes([stencil8, stencil8, stencil8]))


VB_MEM_ADDR = 0x80000000
VB_SIZE = 32 * 1024 * 1024  # 32MB vertex buffer
FB_WIDTH = 256
FB_HEIGHT = 256

VERTEX_BUFFER = VB_MEM_ADDR + 0x0000000
COLOR_BUFFER = VB_MEM_ADDR + 0x00400000
DEPTHSTENCIL_BUFFER = VB_MEM_ADDR + 0x00600000


@pytest.mark.slow
def test_render_triangle():
    """
    Test rendering a single RGB triangle through the full pipeline.
    """
    # Memory setup
    idx_addr, idx_count, pos_addr, norm_addr, col_addr, stride, vb_data = (
        setup_triangle_geometry(VERTEX_BUFFER)
    )

    # Create DUT with testbench infrastructure
    dut = GraphicsPipeline()
    t = SimpleTestbench(dut, mem_addr=VB_MEM_ADDR, mem_size=VB_SIZE)

    # Add all Wishbone buses to arbiter
    t.arbiter.add(dut.wb_index)
    t.arbiter.add(dut.wb_vertex)
    t.arbiter.add(dut.wb_depthstencil)
    t.arbiter.add(dut.wb_color)

    async def testbench(ctx):
        # Initialize vertex buffer memory
        await t.initialize_memory(ctx, VERTEX_BUFFER, vb_data)

        clear_color = struct.pack("<I", 0xFFFF0000)  # opaque red
        clear_depthstencil = struct.pack("<I", 0x0000FFFF)  # max depth, stencil=0
        # Clear color buffer
        await t.initialize_memory(
            ctx, COLOR_BUFFER, clear_color * (FB_WIDTH * FB_HEIGHT)
        )
        # Clear depth/stencil buffer
        await t.initialize_memory(
            ctx, DEPTHSTENCIL_BUFFER, clear_depthstencil * (FB_WIDTH * FB_HEIGHT)
        )

        print(
            f"Triangle setup: idx_addr=0x{idx_addr:08x}, count={idx_count}, stride={stride}"
        )
        print(
            f"  pos_addr=0x{pos_addr:08x}, norm_addr=0x{norm_addr:08x}, col_addr=0x{col_addr:08x}"
        )

        # Configure index generator
        ctx.set(dut.c_index_address, idx_addr)
        ctx.set(dut.c_index_count, idx_count)
        ctx.set(dut.c_index_kind, IndexKind.U16)

        # Configure topology
        ctx.set(dut.c_input_topology, InputTopology.TRIANGLE_LIST)
        ctx.set(dut.c_primitive_restart_enable, 0)
        ctx.set(dut.c_primitive_restart_index, 0)
        ctx.set(dut.c_base_vertex, 0)

        # Configure input assembly (per-vertex attributes)
        # Position
        ctx.set(dut.c_pos.mode, InputMode.PER_VERTEX)
        ctx.set(
            dut.c_pos.info,
            InputData.const(
                {
                    "per_vertex": {
                        "address": pos_addr,
                        "stride": stride,
                    }
                }
            ),
        )

        # Normal
        ctx.set(dut.c_norm.mode, InputMode.PER_VERTEX)
        ctx.set(
            dut.c_norm.info,
            InputData.const(
                {
                    "per_vertex": {
                        "address": norm_addr,
                        "stride": stride,
                    }
                }
            ),
        )

        # Color
        ctx.set(dut.c_col.mode, InputMode.PER_VERTEX)
        ctx.set(
            dut.c_col.info,
            InputData.const(
                {
                    "per_vertex": {
                        "address": col_addr,
                        "stride": stride,
                    }
                }
            ),
        )

        # Vertex transform: identity matrices
        ctx.set(dut.vt_enabled.normal, 1)

        # Identity 4x4 for position_mv
        identity_4x4 = [1.0 if i % 5 == 0 else 0.0 for i in range(16)]
        ctx.set(dut.position_mv, identity_4x4)

        # Identity 4x4 for position_p
        ctx.set(dut.position_p, identity_4x4)

        # Identity 3x3 for normal_mv_inv_t
        identity_3x3 = [1.0 if i % 4 == 0 else 0.0 for i in range(9)]
        ctx.set(dut.normal_mv_inv_t, identity_3x3)

        # Material: ambient=0.2, diffuse=0.8, specular=0.2, shininess=1.0
        ctx.set(dut.material.ambient, [0.2] * 3)
        ctx.set(dut.material.diffuse, [0.8] * 3)
        ctx.set(dut.material.specular, [0.2] * 3)
        ctx.set(dut.material.shininess, fixed.Const(1.0))

        # Light 0: position=(0,0,1,1), ambient=0.2, diffuse=0.8, specular=0.2
        ctx.set(dut.lights[0].position, [0.0, 0.0, 1.0, 1.0])
        ctx.set(dut.lights[0].ambient, [0.2] * 3)
        ctx.set(dut.lights[0].diffuse, [0.8] * 3)
        ctx.set(dut.lights[0].specular, [0.2] * 3)

        # Primitive assembly: TRIANGLES, no culling, CCW
        ctx.set(dut.pa_conf.type, PrimitiveType.TRIANGLES)
        ctx.set(dut.pa_conf.cull, CullFace.NONE)
        ctx.set(dut.pa_conf.winding, FrontFace.CCW)

        # Framebuffer: 640x480
        ctx.set(dut.fb_info.width, FB_WIDTH)
        ctx.set(dut.fb_info.height, FB_HEIGHT)
        ctx.set(dut.fb_info.viewport_x, 0.0)
        ctx.set(dut.fb_info.viewport_y, 0.0)
        ctx.set(dut.fb_info.viewport_width, float(FB_WIDTH))
        ctx.set(dut.fb_info.viewport_height, float(FB_HEIGHT))
        ctx.set(dut.fb_info.viewport_min_depth, 0.0)
        ctx.set(dut.fb_info.viewport_max_depth, 1.0)
        ctx.set(dut.fb_info.scissor_offset_x, 0)
        ctx.set(dut.fb_info.scissor_offset_y, 0)
        ctx.set(dut.fb_info.scissor_width, FB_WIDTH)
        ctx.set(dut.fb_info.scissor_height, FB_HEIGHT)
        ctx.set(dut.fb_info.color_address, COLOR_BUFFER)
        ctx.set(dut.fb_info.color_pitch, FB_WIDTH * 4)
        ctx.set(dut.fb_info.depthstencil_address, DEPTHSTENCIL_BUFFER)
        ctx.set(dut.fb_info.depthstencil_pitch, FB_WIDTH * 4)

        # Depth/stencil: disabled
        ctx.set(dut.depth_conf.test_enabled, 0)
        ctx.set(dut.depth_conf.write_enabled, 0)
        ctx.set(dut.depth_conf.compare_op, CompareOp.ALWAYS)

        # Stencil: compare=ALWAYS, masks=0xFF, ops=KEEP
        for stencil_conf in [dut.stencil_conf_front, dut.stencil_conf_back]:
            ctx.set(stencil_conf.compare_op, CompareOp.ALWAYS)
            ctx.set(stencil_conf.reference, 0x00)
            ctx.set(stencil_conf.mask, 0xFF)
            ctx.set(stencil_conf.write_mask, 0xFF)
            ctx.set(stencil_conf.pass_op, StencilOp.KEEP)
            ctx.set(stencil_conf.fail_op, StencilOp.KEEP)
            ctx.set(stencil_conf.depth_fail_op, StencilOp.KEEP)

        # Blending: disabled (src=ONE, dst=ZERO)
        ctx.set(dut.blend_conf.enabled, 0)
        ctx.set(dut.blend_conf.src_factor, BlendFactor.ONE)
        ctx.set(dut.blend_conf.dst_factor, BlendFactor.ZERO)
        ctx.set(dut.blend_conf.src_a_factor, BlendFactor.ONE)
        ctx.set(dut.blend_conf.dst_a_factor, BlendFactor.ZERO)
        ctx.set(dut.blend_conf.blend_op, BlendOp.ADD)
        ctx.set(dut.blend_conf.blend_a_op, BlendOp.ADD)
        ctx.set(dut.blend_conf.color_write_mask, 0xF)

        print("Configuration done, starting test...")
        # Wait a few cycles for config to settle
        await ctx.tick().repeat(2)
        print("✓ Configuration settled")

        # Check initial ready status
        ready = ctx.get(dut.ready)
        ready_comp = ctx.get(dut.ready_components)
        ready_vec = ctx.get(dut.ready_vec)
        print(
            f"Initial ready: {ready}, components: 0x{ready_comp:x}, vec: 0x{ready_vec:08x}"
        )

        # Start rendering
        print("Starting GPU...")
        ctx.set(dut.start, 1)
        await ctx.tick()
        ctx.set(dut.start, 0)
        await ctx.tick()

        await ctx.tick().until(dut.ready)
        print("✓ Triangle rendering completed successfully")

        # Read back framebuffer and depth/stencil data
        print("Reading framebuffer data...")
        color_data = await t.dbg_access.read_bytes(
            ctx, COLOR_BUFFER, FB_WIDTH * FB_HEIGHT * 4
        )

        print("Reading depth/stencil data...")
        depthstencil_data = await t.dbg_access.read_bytes(
            ctx, DEPTHSTENCIL_BUFFER, FB_WIDTH * FB_HEIGHT * 4
        )

        # Convert RGBA to RGB for PPM
        rgb_data = bytearray()
        for i in range(0, len(color_data), 4):
            # RGBA8888 format, extract RGB
            rgb_data.extend(color_data[i : i + 3])

        # Save images
        save_ppm_image(
            "test_render_triangle_color.ppm", FB_WIDTH, FB_HEIGHT, bytes(rgb_data)
        )
        save_depth_image(
            "test_render_triangle_depth.ppm", FB_WIDTH, FB_HEIGHT, depthstencil_data
        )
        save_stencil_image(
            "test_render_triangle_stencil.ppm", FB_WIDTH, FB_HEIGHT, depthstencil_data
        )
        print("✓ Saved output images")

    # Simulation
    sim = Simulator(t)
    sim.add_clock(1e-6)  # 1 MHz clock
    sim.add_testbench(testbench)

    try:
        sim.run()
    except Exception:
        sim.reset()
        with sim.write_vcd(
            "test_render_triangle.vcd", "test_render_triangle.gtkw", traces=dut
        ):
            sim.run()


@pytest.mark.slow
def test_render_triangle_strip_two():
    """
    Render a simple TRIANGLE_STRIP with two triangles (4 indices) and verify it completes.
    This guards against "every second triangle" being dropped in strip handling.
    """
    # Build a strip with 4 vertices: a quad split into two triangles
    vertices = [
        {  # v0 bottom-left (red)
            "pos": [-0.8, -0.5, 0.2, 1.0],
            "norm": [0.0, 0.0, 1.0],
            "col": [1.0, 0.0, 0.0, 1.0],
        },
        {  # v1 bottom-right (green)
            "pos": [0.0, -0.5, 0.2, 1.0],
            "norm": [0.0, 0.0, 1.0],
            "col": [0.0, 1.0, 0.0, 1.0],
        },
        {  # v2 top-left (blue)
            "pos": [-0.8, 0.5, 0.2, 1.0],
            "norm": [0.0, 0.0, 1.0],
            "col": [0.0, 0.0, 1.0, 1.0],
        },
        {  # v3 top-right (white)
            "pos": [0.0, 0.5, 0.2, 1.0],
            "norm": [0.0, 0.0, 1.0],
            "col": [1.0, 1.0, 1.0, 1.0],
        },
    ]

    stride = 4 * 4 + 3 * 4 + 4 * 4
    pos_offset = 0
    norm_offset = 16
    col_offset = 28

    vb_data = bytearray()
    for v in vertices:
        for val in v["pos"]:
            vb_data.extend(struct.pack("<i", int(val * 65536)))
        for val in v["norm"]:
            vb_data.extend(struct.pack("<i", int(val * 65536)))
        for val in v["col"]:
            vb_data.extend(struct.pack("<i", int(val * 65536)))

    indices = struct.pack("<HHHH", 0, 1, 2, 3)
    idx_count = 4

    vb_mem_addr = VERTEX_BUFFER
    pos_addr = vb_mem_addr + pos_offset
    norm_addr = vb_mem_addr + norm_offset
    col_addr = vb_mem_addr + col_offset
    idx_addr = vb_mem_addr + len(vb_data)
    memory_data = vb_data + indices

    dut = GraphicsPipeline()
    t = SimpleTestbench(dut, mem_addr=VB_MEM_ADDR, mem_size=VB_SIZE)
    t.arbiter.add(dut.wb_index)
    t.arbiter.add(dut.wb_vertex)
    t.arbiter.add(dut.wb_depthstencil)
    t.arbiter.add(dut.wb_color)

    async def testbench(ctx):
        await t.initialize_memory(ctx, VERTEX_BUFFER, memory_data)

        clear_color = struct.pack("<I", 0x00000000)
        clear_depthstencil = struct.pack("<I", 0x0000FFFF)
        await t.initialize_memory(
            ctx, COLOR_BUFFER, clear_color * (FB_WIDTH * FB_HEIGHT)
        )
        await t.initialize_memory(
            ctx, DEPTHSTENCIL_BUFFER, clear_depthstencil * (FB_WIDTH * FB_HEIGHT)
        )

        # Index + topology (triangle strip)
        ctx.set(dut.c_index_address, idx_addr)
        ctx.set(dut.c_index_count, idx_count)
        ctx.set(dut.c_index_kind, IndexKind.U16)
        ctx.set(dut.c_input_topology, InputTopology.TRIANGLE_STRIP)
        ctx.set(dut.c_primitive_restart_enable, 0)
        ctx.set(dut.c_primitive_restart_index, 0)
        ctx.set(dut.c_base_vertex, 0)

        # Attributes
        ctx.set(dut.c_pos.mode, InputMode.PER_VERTEX)
        ctx.set(
            dut.c_pos.info,
            InputData.const({"per_vertex": {"address": pos_addr, "stride": stride}}),
        )
        ctx.set(dut.c_norm.mode, InputMode.PER_VERTEX)
        ctx.set(
            dut.c_norm.info,
            InputData.const({"per_vertex": {"address": norm_addr, "stride": stride}}),
        )
        ctx.set(dut.c_col.mode, InputMode.PER_VERTEX)
        ctx.set(
            dut.c_col.info,
            InputData.const({"per_vertex": {"address": col_addr, "stride": stride}}),
        )

        # Identity transforms and simple material/light as in single-triangle test
        ctx.set(dut.vt_enabled.normal, 1)
        identity_4x4 = [1.0 if i % 5 == 0 else 0.0 for i in range(16)]
        ctx.set(dut.position_mv, identity_4x4)
        ctx.set(dut.position_p, identity_4x4)
        identity_3x3 = [1.0 if i % 4 == 0 else 0.0 for i in range(9)]
        ctx.set(dut.normal_mv_inv_t, identity_3x3)
        ctx.set(dut.material.ambient, [0.2] * 3)
        ctx.set(dut.material.diffuse, [0.8] * 3)
        ctx.set(dut.material.specular, [0.2] * 3)
        ctx.set(dut.material.shininess, fixed.Const(1.0))
        ctx.set(dut.lights[0].position, [0.0, 0.0, 1.0, 1.0])
        ctx.set(dut.lights[0].ambient, [0.2] * 3)
        ctx.set(dut.lights[0].diffuse, [0.8] * 3)
        ctx.set(dut.lights[0].specular, [0.2] * 3)

        # Primitive config: triangles, no cull, CCW
        ctx.set(dut.pa_conf.type, PrimitiveType.TRIANGLES)
        ctx.set(dut.pa_conf.cull, CullFace.NONE)
        ctx.set(dut.pa_conf.winding, FrontFace.CCW)

        # Framebuffer config
        ctx.set(dut.fb_info.width, FB_WIDTH)
        ctx.set(dut.fb_info.height, FB_HEIGHT)
        ctx.set(dut.fb_info.viewport_x, 0.0)
        ctx.set(dut.fb_info.viewport_y, 0.0)
        ctx.set(dut.fb_info.viewport_width, float(FB_WIDTH))
        ctx.set(dut.fb_info.viewport_height, float(FB_HEIGHT))
        ctx.set(dut.fb_info.viewport_min_depth, 0.0)
        ctx.set(dut.fb_info.viewport_max_depth, 1.0)
        ctx.set(dut.fb_info.scissor_offset_x, 0)
        ctx.set(dut.fb_info.scissor_offset_y, 0)
        ctx.set(dut.fb_info.scissor_width, FB_WIDTH)
        ctx.set(dut.fb_info.scissor_height, FB_HEIGHT)
        ctx.set(dut.fb_info.color_address, COLOR_BUFFER)
        ctx.set(dut.fb_info.color_pitch, FB_WIDTH * 4)
        ctx.set(dut.fb_info.depthstencil_address, DEPTHSTENCIL_BUFFER)
        ctx.set(dut.fb_info.depthstencil_pitch, FB_WIDTH * 4)

        # Disable depth/stencil and blending
        ctx.set(dut.depth_conf.test_enabled, 0)
        ctx.set(dut.depth_conf.write_enabled, 0)
        ctx.set(dut.depth_conf.compare_op, CompareOp.ALWAYS)
        for stencil_conf in [dut.stencil_conf_front, dut.stencil_conf_back]:
            ctx.set(stencil_conf.compare_op, CompareOp.ALWAYS)
            ctx.set(stencil_conf.reference, 0x00)
            ctx.set(stencil_conf.mask, 0xFF)
            ctx.set(stencil_conf.write_mask, 0xFF)
            ctx.set(stencil_conf.pass_op, StencilOp.KEEP)
            ctx.set(stencil_conf.fail_op, StencilOp.KEEP)
            ctx.set(stencil_conf.depth_fail_op, StencilOp.KEEP)
        ctx.set(dut.blend_conf.enabled, 0)
        ctx.set(dut.blend_conf.src_factor, BlendFactor.ONE)
        ctx.set(dut.blend_conf.dst_factor, BlendFactor.ZERO)
        ctx.set(dut.blend_conf.src_a_factor, BlendFactor.ONE)
        ctx.set(dut.blend_conf.dst_a_factor, BlendFactor.ZERO)
        ctx.set(dut.blend_conf.blend_op, BlendOp.ADD)
        ctx.set(dut.blend_conf.blend_a_op, BlendOp.ADD)
        ctx.set(dut.blend_conf.color_write_mask, 0xF)

        await ctx.tick().repeat(2)

        # Start and wait for completion
        ctx.set(dut.start, 1)
        await ctx.tick()
        ctx.set(dut.start, 0)
        await ctx.tick()
        await ctx.tick().until(dut.ready)

        # Read back color to ensure something was drawn (count non-zero pixels)
        color_data = await t.dbg_access.read_bytes(
            ctx, COLOR_BUFFER, FB_WIDTH * FB_HEIGHT * 4
        )
        nonzero = 0
        for i in range(0, len(color_data), 4):
            if (
                color_data[i]
                | color_data[i + 1]
                | color_data[i + 2]
                | color_data[i + 3]
            ):
                nonzero += 1
        # At least some pixels should be non-zero for two triangles
        assert nonzero > 0

    sim = Simulator(t)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    sim.run()
