import pytest
from amaranth.sim import Simulator

from gpu.primitive_assembly.cores import PrimitiveAssembly
from gpu.utils import fixed
from gpu.utils.layouts import num_textures
from gpu.utils.types import CullFace, FixedPoint, FrontFace, PrimitiveType

from ..utils.streams import stream_testbench
from ..utils.testbench import SimpleTestbench


def make_pa_vertex(pos, color, color_back=None):
    return {
        "position_ndc": pos,
        "texcoords": [[0.0, 0.0, 0.0, 1.0] for _ in range(num_textures)],
        "color": color,
        "color_back": color_back if color_back is not None else color,
    }


def assert_rasterizer_vertex(payload, pos, color, front):
    got_pos = [c.as_float() for c in payload.position_ndc]
    got_color = [c.as_float() for c in payload.color]
    assert got_pos == pytest.approx(pos)
    assert got_color == pytest.approx(color)
    assert int(payload.front_facing) == front


@pytest.mark.parametrize(
    "pos,color",
    [
        ([0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]),
        (
            [0.5, -0.5, 0.5, 1.0, -1.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        ),
    ],
)
def test_points_passthrough_hypothesis(pos, color):
    # Force w to 1.0 to match NDC expectations
    pos = [fixed.Const(v, FixedPoint).as_float() for v in pos]
    color = [fixed.Const(v, FixedPoint).as_float() for v in color]

    dut = PrimitiveAssembly()
    t = SimpleTestbench(dut)

    vertices = [
        make_pa_vertex(pos[i : i + 4], color[i : i + 4]) for i in range(0, len(pos), 4)
    ]

    print("Testing points passthrough with vertices:", vertices)

    async def init_proc(ctx):
        ctx.set(dut.config.type, PrimitiveType.POINTS)
        ctx.set(dut.config.cull, CullFace.NONE)
        ctx.set(dut.config.winding, FrontFace.CCW)

    async def checker(ctx, results):
        assert len(results) == len(vertices)
        for i in range(len(vertices)):
            assert_rasterizer_vertex(
                results[i], pos[4 * i : 4 * i + 4], color[4 * i : 4 * i + 4], 1
            )

    sim = Simulator(t)
    sim.add_clock(1e-6)
    stream_testbench(
        sim,
        init_process=init_proc,
        input_stream=dut.i,
        input_data=vertices,
        output_stream=dut.o,
        output_data_checker=checker,
        idle_for=50,
    )

    sim.run()


def test_lines_passthrough():
    dut = PrimitiveAssembly()
    t = SimpleTestbench(dut)

    line = [
        make_pa_vertex([0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]),
        make_pa_vertex([0.5, -0.5, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]),
    ]

    async def init_proc(ctx):
        ctx.set(dut.config.type, PrimitiveType.LINES)
        ctx.set(dut.config.cull, CullFace.NONE)
        ctx.set(dut.config.winding, FrontFace.CCW)

    async def checker(ctx, results):
        assert len(results) == 2
        assert_rasterizer_vertex(
            results[0], line[0]["position_ndc"], line[0]["color"], 1
        )
        assert_rasterizer_vertex(
            results[1], line[1]["position_ndc"], line[1]["color"], 1
        )

    sim = Simulator(t)
    sim.add_clock(1e-6)
    stream_testbench(
        sim,
        init_process=init_proc,
        input_stream=dut.i,
        input_data=line,
        output_stream=dut.o,
        output_data_checker=checker,
        idle_for=50,
    )

    sim.run()


@pytest.mark.parametrize("front_face", [FrontFace.CCW, FrontFace.CW])
@pytest.mark.parametrize(
    "cull_face", [CullFace.NONE, CullFace.BACK, CullFace.FRONT, CullFace.FRONT_AND_BACK]
)
@pytest.mark.parametrize(
    "tri, winding_order",
    [
        (
            [
                make_pa_vertex(
                    [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]
                ),
                make_pa_vertex(
                    [1.0, 0.0, 0.0, 1.0], [1.0, 0.5, 0.0, 1.0], [0.0, 1.0, 0.5, 1.0]
                ),
                make_pa_vertex(
                    [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.5, 1.0], [0.0, 0.5, 1.0, 1.0]
                ),
            ],
            FrontFace.CCW,
        ),
        (
            [
                make_pa_vertex(
                    [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]
                ),
                make_pa_vertex(
                    [0.0, 1.0, 0.0, 1.0], [1.0, 0.5, 0.0, 1.0], [0.0, 1.0, 0.5, 1.0]
                ),
                make_pa_vertex(
                    [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.5, 1.0], [0.0, 0.5, 1.0, 1.0]
                ),
            ],
            FrontFace.CW,
        ),
    ],
)
def test_triangles_winding_and_front_face(tri, winding_order, front_face, cull_face):
    dut = PrimitiveAssembly()
    t = SimpleTestbench(dut)

    ff_expected = winding_order == front_face
    cols = ["color" if ff_expected else "color_back"] * 3

    if ff_expected:
        should_be_culled = bool(cull_face & CullFace.FRONT)
    else:
        should_be_culled = bool(cull_face & CullFace.BACK)

    async def init_proc(ctx):
        ctx.set(dut.config.type, PrimitiveType.TRIANGLES)
        ctx.set(dut.config.cull, cull_face)
        ctx.set(dut.config.winding, front_face)

    async def checker(ctx, results):
        if should_be_culled:
            assert results == []
            return

        assert len(results) == len(tri)

        for i in range(3):
            use_key = cols[i]
            exp_color = tri[i][use_key]
            assert_rasterizer_vertex(
                results[i], tri[i]["position_ndc"], exp_color, ff_expected
            )

    sim = Simulator(t)
    sim.add_clock(1e-6)
    stream_testbench(
        sim,
        init_process=init_proc,
        input_stream=dut.i,
        input_data=tri,
        output_stream=dut.o,
        output_data_checker=checker,
        idle_for=50,
    )

    sim.run()


def test_triangle_front_facing():
    dut = PrimitiveAssembly()
    t = SimpleTestbench(dut)

    # CCW winding => front-facing with default FrontFace.CCW
    tri = [
        make_pa_vertex(
            [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]
        ),
        make_pa_vertex(
            [1.0, 0.0, 0.0, 1.0], [1.0, 0.5, 0.0, 1.0], [0.0, 1.0, 0.5, 1.0]
        ),
        make_pa_vertex(
            [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.5, 1.0], [0.0, 0.5, 1.0, 1.0]
        ),
    ]

    async def init_proc(ctx):
        ctx.set(dut.config.type, PrimitiveType.TRIANGLES)
        ctx.set(dut.config.cull, CullFace.NONE)
        ctx.set(dut.config.winding, FrontFace.CCW)

    async def checker(ctx, results):
        assert len(results) == 3
        assert_rasterizer_vertex(results[0], tri[0]["position_ndc"], tri[0]["color"], 1)
        assert_rasterizer_vertex(results[1], tri[1]["position_ndc"], tri[1]["color"], 1)
        assert_rasterizer_vertex(results[2], tri[2]["position_ndc"], tri[2]["color"], 1)

    sim = Simulator(t)
    sim.add_clock(1e-6)
    stream_testbench(
        sim,
        init_process=init_proc,
        input_stream=dut.i,
        input_data=tri,
        output_stream=dut.o,
        output_data_checker=checker,
        idle_for=1000,
    )

    sim.run()


def test_triangle_back_face_culled():
    dut = PrimitiveAssembly()
    t = SimpleTestbench(dut)

    # CW winding with CCW front-face definition => back-facing
    tri = [
        make_pa_vertex(
            [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]
        ),
        make_pa_vertex(
            [0.0, 1.0, 0.0, 1.0], [1.0, 0.5, 0.0, 1.0], [0.0, 1.0, 0.5, 1.0]
        ),
        make_pa_vertex(
            [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.5, 1.0], [0.0, 0.5, 1.0, 1.0]
        ),
    ]

    async def init_proc(ctx):
        ctx.set(dut.config.type, PrimitiveType.TRIANGLES)
        ctx.set(dut.config.cull, CullFace.BACK)
        ctx.set(dut.config.winding, FrontFace.CCW)

    async def checker(ctx, results):
        # Back-face culled => no output
        assert results == []

    sim = Simulator(t)
    sim.add_clock(1e-6)
    stream_testbench(
        sim,
        init_process=init_proc,
        input_stream=dut.i,
        input_data=tri,
        output_stream=dut.o,
        output_data_checker=checker,
        idle_for=30,
    )

    sim.run()


def test_triangle_back_face_uses_back_color():
    dut = PrimitiveAssembly()
    t = SimpleTestbench(dut)

    tri = [
        make_pa_vertex(
            [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]
        ),
        make_pa_vertex(
            [0.0, 1.0, 0.0, 1.0], [1.0, 0.5, 0.0, 1.0], [0.0, 1.0, 0.5, 1.0]
        ),
        make_pa_vertex(
            [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.5, 1.0], [0.0, 0.5, 1.0, 1.0]
        ),
    ]

    async def init_proc(ctx):
        ctx.set(dut.config.type, PrimitiveType.TRIANGLES)
        ctx.set(dut.config.cull, CullFace.NONE)
        ctx.set(dut.config.winding, FrontFace.CCW)

    async def checker(ctx, results):
        assert len(results) == 3
        assert_rasterizer_vertex(
            results[0], tri[0]["position_ndc"], tri[0]["color_back"], 0
        )
        assert_rasterizer_vertex(
            results[1], tri[1]["position_ndc"], tri[1]["color_back"], 0
        )
        assert_rasterizer_vertex(
            results[2], tri[2]["position_ndc"], tri[2]["color_back"], 0
        )

    sim = Simulator(t)
    sim.add_clock(1e-6)
    stream_testbench(
        sim,
        init_process=init_proc,
        input_stream=dut.i,
        input_data=tri,
        output_stream=dut.o,
        output_data_checker=checker,
        idle_for=30,
    )

    sim.run()
