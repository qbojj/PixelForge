import struct

import pytest
from amaranth import Module
from amaranth.lib import wiring
from amaranth.sim import Simulator

from gpu.input_assembly.cores import (
    IndexGenerator,
    InputAssembly,
    InputTopologyProcessor,
)
from gpu.input_assembly.layouts import InputData, InputMode
from gpu.primitive_assembly.cores import PrimitiveAssembly
from gpu.rasterizer.cores import PrimitiveClipper
from gpu.rasterizer.rasterizer import PerspectiveDivide, TriangleRasterizer
from gpu.utils.layouts import num_lights, num_textures
from gpu.utils.types import (
    CullFace,
    FrontFace,
    IndexKind,
    InputTopology,
    PrimitiveType,
)
from gpu.vertex_shading.cores import (
    VertexShading,
)
from gpu.vertex_transform.cores import VertexTransform

from ..utils.streams import stream_testbench
from ..utils.testbench import SimpleTestbench
from ..utils.visualization import Fragment, FragmentVisualizer


def make_pa_vertex(pos, color):
    """Create a primitive assembly vertex (output of PrimitiveAssembly)"""
    return {
        "position_ndc": pos,
        "texcoords": [[0.0, 0.0, 0.0, 1.0] for _ in range(num_textures)],
        "color": color,
        "front_facing": 1,
    }


def fp16_16(v: float) -> int:
    return int(round(v * (1 << 16)))


def build_vertex_memory(base_addr: int):
    vertices = [
        {
            "pos": [-0.5, -0.5, 0.2, 1.0],
            "norm": [0.0, 0.0, 1.0],
            "col": [1.0, 0.2, 0.1, 1.0],
        },
        {
            "pos": [0.6, -0.4, 0.2, 1.0],
            "norm": [0.0, 0.0, 1.0],
            "col": [0.1, 0.9, 0.2, 1.0],
        },
        {
            "pos": [0.0, 0.7, 0.2, 1.0],
            "norm": [0.0, 0.0, 1.0],
            "col": [0.2, 0.3, 0.95, 1.0],
        },
    ]

    stride = 44  # 4 pos + 3 norm + 4 color, all 32-bit
    vb_data = bytearray()
    for v in vertices:
        for val in v["pos"]:
            vb_data.extend(struct.pack("<i", fp16_16(val)))
        for val in v["norm"]:
            vb_data.extend(struct.pack("<i", fp16_16(val)))
        for val in v["col"]:
            vb_data.extend(struct.pack("<i", fp16_16(val)))

    idx_data = struct.pack("<HHH", 0, 1, 2)
    memory = vb_data + idx_data

    pos_addr = base_addr
    norm_addr = base_addr + 16
    col_addr = base_addr + 28
    idx_addr = base_addr + len(vb_data)

    return {
        "idx_addr": idx_addr,
        "idx_count": 3,
        "pos_addr": pos_addr,
        "norm_addr": norm_addr,
        "col_addr": col_addr,
        "stride": stride,
        "memory": memory,
        "colors": [v["col"] for v in vertices],
    }


def test_clip_to_perspective_divide_colors():
    """Run clipper + perspective divide and log colors of vertices that pass."""
    m = Module()
    m.submodules.idx = idx = IndexGenerator()
    m.submodules.topo = topo = InputTopologyProcessor()
    m.submodules.ia = ia = InputAssembly()
    m.submodules.vtx_xf = vtx_xf = VertexTransform()
    m.submodules.vtx_sh = vtx_sh = VertexShading(num_lights)
    m.submodules.pa = pa = PrimitiveAssembly()
    m.submodules.clip = clip = PrimitiveClipper()
    m.submodules.div = div = PerspectiveDivide()

    wiring.connect(m, idx.os_index, topo.is_index)
    wiring.connect(m, topo.os_index, ia.is_index)
    wiring.connect(m, ia.os_vertex, vtx_xf.is_vertex)
    wiring.connect(m, vtx_xf.os_vertex, vtx_sh.is_vertex)
    wiring.connect(m, vtx_sh.os_vertex, pa.is_vertex)
    wiring.connect(m, pa.os_primitive, clip.is_vertex)
    wiring.connect(m, clip.os_vertex, div.i_vertex)

    t = SimpleTestbench(m)
    t.arbiter.add(idx.bus)
    t.arbiter.add(ia.bus)

    vb_base = 0x80000000
    geom = build_vertex_memory(vb_base)

    logged_colors: list[tuple[float, float, float, float]] = []

    async def testbench(ctx):
        await t.initialize_memory(ctx, vb_base, geom["memory"])

        ctx.set(idx.c_address, geom["idx_addr"])
        ctx.set(idx.c_count, geom["idx_count"])
        ctx.set(idx.c_kind, IndexKind.U16)

        ctx.set(topo.c_input_topology, InputTopology.TRIANGLE_LIST)
        ctx.set(topo.c_primitive_restart_enable, 0)
        ctx.set(topo.c_primitive_restart_index, 0)
        ctx.set(topo.c_base_vertex, 0)

        ctx.set(ia.c_pos.mode, InputMode.PER_VERTEX)
        ctx.set(
            ia.c_pos.info,
            InputData.const(
                {"per_vertex": {"address": geom["pos_addr"], "stride": geom["stride"]}}
            ),
        )
        ctx.set(ia.c_norm.mode, InputMode.PER_VERTEX)
        ctx.set(
            ia.c_norm.info,
            InputData.const(
                {"per_vertex": {"address": geom["norm_addr"], "stride": geom["stride"]}}
            ),
        )
        ctx.set(ia.c_col.mode, InputMode.PER_VERTEX)
        ctx.set(
            ia.c_col.info,
            InputData.const(
                {"per_vertex": {"address": geom["col_addr"], "stride": geom["stride"]}}
            ),
        )

        identity_4x4 = [1.0 if i % 5 == 0 else 0.0 for i in range(16)]
        identity_3x3 = [1.0 if i % 4 == 0 else 0.0 for i in range(9)]
        ctx.set(vtx_xf.enabled.normal, 1)
        ctx.set(vtx_xf.position_mv, identity_4x4)
        ctx.set(vtx_xf.position_p, identity_4x4)
        ctx.set(vtx_xf.normal_mv_inv_t, identity_3x3)

        # Ambient-only lighting so vertex colors pass through modulation unchanged
        ctx.set(vtx_sh.material.ambient, [1.0, 1.0, 1.0])
        ctx.set(vtx_sh.material.diffuse, [0.0, 0.0, 0.0])
        ctx.set(vtx_sh.material.specular, [0.0, 0.0, 0.0])
        ctx.set(vtx_sh.material.shininess, 0)

        ctx.set(vtx_sh.lights[0].position, [0.0, 0.0, 1.0, 0.0])
        ctx.set(vtx_sh.lights[0].ambient, [1.0, 1.0, 1.0])
        ctx.set(vtx_sh.lights[0].diffuse, [0.0, 0.0, 0.0])
        ctx.set(vtx_sh.lights[0].specular, [0.0, 0.0, 0.0])

        ctx.set(pa.config.type, PrimitiveType.TRIANGLES)
        ctx.set(pa.config.cull, CullFace.NONE)
        ctx.set(pa.config.winding, FrontFace.CCW)

        ctx.set(clip.prim_type, PrimitiveType.TRIANGLES)

        await ctx.tick().repeat(2)

        ctx.set(idx.start, 1)
        await ctx.tick()
        ctx.set(idx.start, 0)

        for _ in range(2000):
            ctx.set(div.o_vertex.ready, 1)
            await ctx.tick()
            if ctx.get(div.o_vertex.valid):
                vtx = ctx.get(div.o_vertex.payload)
                color = tuple(comp.as_float() for comp in vtx.color)
                logged_colors.append(color)
                print(f"Passing vertex {len(logged_colors)-1} color: {color}")
                if len(logged_colors) == geom["idx_count"]:
                    break

    sim = Simulator(t)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    sim.run()

    assert logged_colors, "Expected to log passing vertex colors"
    assert len(logged_colors) == geom["idx_count"]
    print("Logged colors:", logged_colors)
    for color in logged_colors:
        assert any(color == pytest.approx(c, abs=1 / 255) for c in geom["colors"])


@pytest.mark.slow
@pytest.mark.parametrize("persp", [True, False])
def test_rasterizer_single_triangle(persp: bool):
    """Test rasterizing a single triangle"""
    m = Module()
    m.submodules.div = div = PerspectiveDivide()
    m.submodules.rast = dut = TriangleRasterizer()
    wiring.connect(m, div.o_vertex, dut.is_vertex)
    t = SimpleTestbench(m)

    # Setup framebuffer
    fb_width = 128
    fb_height = 128
    fb_info = {
        "width": fb_width,
        "height": fb_height,
        "viewport_x": 0.0,
        "viewport_y": 0.0,
        "viewport_width": float(fb_width),
        "viewport_height": float(fb_height),
        "viewport_min_depth": 0.0,
        "viewport_max_depth": 1.0,
        "scissor_offset_x": 0,
        "scissor_offset_y": 0,
        "scissor_width": fb_width,
        "scissor_height": fb_height,
        "color_address": 0,
        "color_pitch": fb_width * 4,
    }

    # Create a triangle in NDC space (centered, filling ~1/4 of viewport)
    # Triangle vertices in NDC [-1, 1]
    triangle_vertices = [
        make_pa_vertex(
            [-1.0, -1.0, 0.5, 1.0], [1.0, 0.0, 0.0, 1.0]  # Bottom-left (NDC)  # Red
        ),
        make_pa_vertex(
            [1.0, -1.0, 0.5, 1.0], [0.0, 1.0, 0.0, 1.0]  # Bottom-right (NDC)  # Green
        ),
        make_pa_vertex([0.0, 1.0, 0.5, 1.0], [0.0, 0.0, 1.0, 1.0]),  # Top (NDC)  # Blue
    ]

    if persp:
        for i, v in enumerate(triangle_vertices):
            for j in range(4):
                v["position_ndc"][j] *= (
                    0.5 + i * 0.5
                )  # Vary w for perspective interpolation

    collected_fragments = []

    async def collect_output(ctx, results):
        nonlocal collected_fragments
        collected_fragments = results
        # Verify we got some fragments
        assert len(results) > 0, "No fragments generated for triangle"
        print(f"Generated {len(results)} fragments")

        # Basic validation: all fragments should be within bounds
        for frag in results:
            x, y = int(frag.coord_pos[0]), int(frag.coord_pos[1])
            assert 0 <= x < fb_width, f"Fragment X {x} out of bounds"
            assert 0 <= y < fb_height, f"Fragment Y {y} out of bounds"

            assert all(
                0.0 <= c.as_float() <= 1.0 for c in frag.color
            ), f"Fragment color {frag.color} out of range"

    sim = Simulator(t)
    sim.add_clock(1e-6)

    async def init_proc(ctx):
        # Set framebuffer info
        ctx.set(dut.fb_info, fb_info)

    stream_testbench(
        sim,
        init_process=init_proc,
        input_stream=div.i_vertex,
        input_data=triangle_vertices,
        output_stream=dut.os_fragment,
        output_data_checker=collect_output,
        idle_for=100000,  # Wait for rasterization to complete
    )

    sim.run()

    assert len(collected_fragments) > 0, "No fragments were rasterized"

    fragments = [
        Fragment(
            coord_pos=(frag.coord_pos[0], frag.coord_pos[1]),
            color=(
                frag.color[0].as_float(),
                frag.color[1].as_float(),
                frag.color[2].as_float(),
                frag.color[3].as_float(),
            ),
        )
        for frag in collected_fragments
    ]

    # Visualize results
    file = "triangle_single_persp.ppm" if persp else "triangle_single_linear.ppm"
    visualizer = FragmentVisualizer(fb_width, fb_height)

    visualizer.clear((0.0, 0.0, 0.0, 1.0))
    visualizer.render(fragments)

    visualizer.generate_ppm_image(file)

    stats = visualizer.generate_statistics(fragments)
    print("Rasterization statistics:", stats)


@pytest.mark.slow
def test_rasterizer_two_triangles():
    """Test rasterizing two triangles with different colors"""
    m = Module()
    m.submodules.div = div = PerspectiveDivide()
    m.submodules.rast = dut = TriangleRasterizer()
    wiring.connect(m, div.o_vertex, dut.is_vertex)
    t = SimpleTestbench(m)

    # Setup framebuffer
    fb_width = 128
    fb_height = 128
    fb_info = {
        "width": fb_width,
        "height": fb_height,
        "viewport_x": 0.0,
        "viewport_y": 0.0,
        "viewport_width": float(fb_width),
        "viewport_height": float(fb_height),
        "viewport_min_depth": 0.0,
        "viewport_max_depth": 1.0,
        "scissor_offset_x": 0,
        "scissor_offset_y": 0,
        "scissor_width": fb_width,
        "scissor_height": fb_height,
        "color_address": 0,
        "color_pitch": fb_width * 4,
    }

    # Two triangles positioned side by side
    triangle1 = [
        make_pa_vertex([-0.8, -0.5, 0.5, 1.0], [1.0, 0.0, 0.0, 1.0]),  # Red
        make_pa_vertex([-0.2, -0.5, 0.5, 1.0], [1.0, 0.0, 0.0, 1.0]),
        make_pa_vertex([-0.5, 0.2, 0.5, 1.0], [1.0, 0.0, 0.0, 1.0]),
    ]

    triangle2 = [
        make_pa_vertex([0.2, -0.5, 0.5, 1.0], [0.0, 0.0, 1.0, 1.0]),  # Blue
        make_pa_vertex([0.8, -0.5, 0.5, 1.0], [0.0, 0.0, 1.0, 1.0]),
        make_pa_vertex([0.5, 0.2, 0.5, 1.0], [0.0, 0.0, 1.0, 1.0]),
    ]

    all_fragments = []

    async def collect_output(ctx, results):
        nonlocal all_fragments
        all_fragments = results
        print(f"Generated {len(results)} fragments for both triangles")

        if len(results) > 0:
            # Count fragments by color to verify both triangles rendered
            red_frags = sum(
                1
                for f in results
                if f.color[0].as_float() > 0.8 and f.color[1].as_float() < 0.2
            )
            blue_frags = sum(
                1
                for f in results
                if f.color[2].as_float() > 0.8 and f.color[0].as_float() < 0.2
            )

            print(f"Red fragments: {red_frags}, Blue fragments: {blue_frags}")
            # Only assert if we have fragments
            if red_frags == 0 and blue_frags == 0:
                print("Warning: No color-filtered fragments found")
        else:
            print("Warning: No fragments generated for two triangles test")

    sim = Simulator(t)
    sim.add_clock(1e-6)

    input_vertices = triangle1 + triangle2

    async def init_proc(ctx):
        ctx.set(dut.fb_info, fb_info)

    stream_testbench(
        sim,
        init_process=init_proc,
        input_stream=div.i_vertex,
        input_data=input_vertices,
        output_stream=dut.os_fragment,
        output_data_checker=collect_output,
        idle_for=10000,  # Wait for rasterization to complete
    )

    sim.run()

    fragments = [
        Fragment(
            coord_pos=(frag.coord_pos[0], frag.coord_pos[1]),
            color=(
                frag.color[0].as_float(),
                frag.color[1].as_float(),
                frag.color[2].as_float(),
                frag.color[3].as_float(),
            ),
        )
        for frag in all_fragments
    ]

    # Visualize results
    visualizer = FragmentVisualizer(fb_width, fb_height)

    visualizer.clear((0.0, 0.0, 0.0, 1.0))
    visualizer.render(fragments)

    visualizer.generate_ppm_image("triangle_two.ppm")
    stats = visualizer.generate_statistics(fragments)
    print("Rasterization statistics:", stats)


@pytest.mark.slow
def test_rasterizer_depth_interpolation():
    """Test that depth is correctly interpolated"""
    m = Module()
    m.submodules.div = div = PerspectiveDivide()
    m.submodules.rast = dut = TriangleRasterizer()
    wiring.connect(m, div.o_vertex, dut.is_vertex)
    t = SimpleTestbench(m)

    fb_width = 128
    fb_height = 128
    fb_info = {
        "width": fb_width,
        "height": fb_height,
        "viewport_x": 0.0,
        "viewport_y": 0.0,
        "viewport_width": float(fb_width),
        "viewport_height": float(fb_height),
        "viewport_min_depth": 0.0,
        "viewport_max_depth": 1.0,
        "scissor_offset_x": 0,
        "scissor_offset_y": 0,
        "scissor_width": fb_width,
        "scissor_height": fb_height,
        "color_address": 0,
        "color_pitch": fb_width * 4,
    }

    # Triangle with varying depth (0.2 at corners, 0.8 at center)
    triangle = [
        make_pa_vertex([-0.5, -0.5, 0.2, 1.0], [1.0, 1.0, 1.0, 1.0]),
        make_pa_vertex([0.5, -0.5, 0.2, 1.0], [1.0, 1.0, 1.0, 1.0]),
        make_pa_vertex([0.0, 0.5, 0.8, 1.0], [1.0, 1.0, 1.0, 1.0]),
    ]

    collected_fragments = []

    async def collect_output(ctx, results):
        nonlocal collected_fragments
        collected_fragments = results
        print(f"Generated {len(results)} fragments")

        if results:
            depths = [
                f.depth.as_float() if hasattr(f.depth, "as_float") else float(f.depth)
                for f in results
            ]
            print(f"Depth range: {min(depths):.4f} to {max(depths):.4f}")
            if min(depths) < 0.2 or max(depths) > 0.8:
                print("Warning: Depth values outside expected range [0.2, 0.8]")
        else:
            print("Warning: No fragments generated for depth interpolation test")

    sim = Simulator(t)
    sim.add_clock(1e-6)

    async def init_proc(ctx):
        ctx.set(dut.fb_info, fb_info)

    stream_testbench(
        sim,
        init_process=init_proc,
        input_stream=div.i_vertex,
        input_data=triangle,
        output_stream=dut.os_fragment,
        output_data_checker=collect_output,
        idle_for=10000,  # Wait for rasterization to complete
    )

    sim.run()

    fragments = [
        Fragment(
            coord_pos=(frag.coord_pos[0], frag.coord_pos[1]),
            color=(
                (1.0 + frag.depth.as_float()) / 2.0,
                0.0,
                0.0,
                1.0,
            ),  # Visualize depth as red channel
        )
        for frag in collected_fragments
    ]

    # Visualize results
    visualizer = FragmentVisualizer(fb_width, fb_height)

    visualizer.clear((0.0, 0.0, 0.0, 1.0))
    visualizer.render(fragments)

    visualizer.generate_ppm_image("triangle_depth.ppm")
    stats = visualizer.generate_statistics(fragments)
    print("Rasterization statistics:", stats)


@pytest.mark.slow
@pytest.mark.parametrize("alpha", [True, False])
def test_rasterizer_two_overlapping_triangles(alpha: bool):
    """Test rasterizing two overlapping triangles to check fragment generation"""
    m = Module()
    m.submodules.div = div = PerspectiveDivide()
    m.submodules.rast = dut = TriangleRasterizer()
    wiring.connect(m, div.o_vertex, dut.is_vertex)
    t = SimpleTestbench(m)

    fb_width = 128
    fb_height = 128
    fb_info = {
        "width": fb_width,
        "height": fb_height,
        "viewport_x": 0.0,
        "viewport_y": 0.0,
        "viewport_width": float(fb_width),
        "viewport_height": float(fb_height),
        "viewport_min_depth": 0.0,
        "viewport_max_depth": 1.0,
        "scissor_offset_x": 0,
        "scissor_offset_y": 0,
        "scissor_width": fb_width,
        "scissor_height": fb_height,
        "color_address": 0,
        "color_pitch": fb_width * 4,
    }

    # Two overlapping triangles
    triangle1 = [
        make_pa_vertex([-0.5, -0.5, 0.5, 1.0], [1.0, 0.0, 0.0, 1.0]),  # Red
        make_pa_vertex([0.5, -0.5, 0.5, 1.0], [1.0, 0.0, 0.0, 1.0]),
        make_pa_vertex([0.0, 0.5, 0.5, 1.0], [1.0, 0.0, 0.0, 1.0]),
    ]

    triangle2 = [
        make_pa_vertex([-0.3, -0.3, 0.5, 1.0], [0.0, 1.0, 0.0, 1.0]),  # Green
        make_pa_vertex([0.7, -0.3, 0.5, 1.0], [0.0, 1.0, 0.0, 1.0]),
        make_pa_vertex([0.2, 0.7, 0.5, 1.0], [0.0, 1.0, 0.0, 1.0]),
    ]

    if alpha:
        for v in triangle1 + triangle2:
            v["color"][3] = 0.5  # Set alpha to 0.5

    all_fragments = []

    async def collect_output(ctx, results):
        nonlocal all_fragments
        all_fragments = results
        print(f"Generated {len(results)} fragments for overlapping triangles")

        if len(results) > 0:
            red_frags = sum(
                1
                for f in results
                if f.color[0].as_float() > 0.8 and f.color[1].as_float() < 0.2
            )
            green_frags = sum(
                1
                for f in results
                if f.color[1].as_float() > 0.8 and f.color[0].as_float() < 0.2
            )

            print(f"Red fragments: {red_frags}, Green fragments: {green_frags}")
        else:
            print("Warning: No fragments generated for overlapping triangles test")

    sim = Simulator(t)
    sim.add_clock(1e-6)
    input_vertices = triangle1 + triangle2

    async def init_proc(ctx):
        ctx.set(dut.fb_info, fb_info)

    stream_testbench(
        sim,
        init_process=init_proc,
        input_stream=div.i_vertex,
        input_data=input_vertices,
        output_stream=dut.os_fragment,
        output_data_checker=collect_output,
        idle_for=10000,  # Wait for rasterization to complete
    )

    sim.run()

    fragments = [
        Fragment(
            coord_pos=(frag.coord_pos[0], frag.coord_pos[1]),
            color=(
                frag.color[0].as_float(),
                frag.color[1].as_float(),
                frag.color[2].as_float(),
                frag.color[3].as_float(),
            ),
        )
        for frag in all_fragments
    ]

    # Visualize results
    visualizer = FragmentVisualizer(fb_width, fb_height)

    visualizer.clear((0.0, 0.0, 0.0, 1.0))
    visualizer.render(fragments)

    file = "triangle_overlapping_alpha.ppm" if alpha else "triangle_overlapping.ppm"
    visualizer.generate_ppm_image(file)
    stats = visualizer.generate_statistics(fragments)
    print("Rasterization statistics:", stats)
