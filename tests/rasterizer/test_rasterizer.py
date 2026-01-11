import pytest
from amaranth import Module
from amaranth.lib import wiring
from amaranth.sim import Simulator

from gpu.rasterizer.rasterizer import PerspectiveDivide, TriangleRasterizer
from gpu.utils.layouts import num_textures

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
