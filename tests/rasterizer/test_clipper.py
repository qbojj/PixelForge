import pytest
from amaranth import *
from amaranth.sim import Simulator

from gpu.rasterizer.cores import PrimitiveClipper
from gpu.utils.layouts import num_textures
from gpu.utils.types import PrimitiveType

from ..utils.streams import stream_testbench


def make_vertex(x, y, z, w=1.0, r=1.0, g=1.0, b=1.0, a=1.0):
    """Helper to create a vertex with NDC position and color."""
    return {
        "position_ndc": [x, y, z, w],
        "texcoords": [[0.0, 0.0, 0.0, 1.0] for _ in range(num_textures)],
        "color": [r, g, b, a],
        "front_facing": 1,
    }


@pytest.mark.parametrize(
    "test_name,prim_type,input_vertices,expected_count",
    [
        # Trivial accept: all vertices inside
        (
            "triangle_fully_inside",
            PrimitiveType.TRIANGLES,
            [
                make_vertex(0.0, 0.0, 0.0),
                make_vertex(0.5, 0.0, 0.0),
                make_vertex(0.0, 0.5, 0.0),
            ],
            1,  # 1 triangle output
        ),
        # Trivial reject: all vertices outside same plane (+x)
        (
            "triangle_fully_outside_plus_x",
            PrimitiveType.TRIANGLES,
            [
                make_vertex(1.5, 0.0, 0.0),
                make_vertex(2.0, 0.0, 0.0),
                make_vertex(1.5, 0.5, 0.0),
            ],
            0,  # No output
        ),
        # Trivial reject: all vertices outside same plane (-x)
        (
            "triangle_fully_outside_minus_x",
            PrimitiveType.TRIANGLES,
            [
                make_vertex(-1.5, 0.0, 0.0),
                make_vertex(-2.0, 0.0, 0.0),
                make_vertex(-1.5, 0.5, 0.0),
            ],
            0,  # No output
        ),
        # Trivial reject: all vertices outside same plane (+y)
        (
            "triangle_fully_outside_plus_y",
            PrimitiveType.TRIANGLES,
            [
                make_vertex(0.5, 1.5, 0.0),
                make_vertex(0.0, 2.0, 0.0),
                make_vertex(0.2, 1.5, 0.0),
            ],
            0,  # No output
        ),
        # Trivial reject: all vertices outside same plane (-y)
        (
            "triangle_fully_outside_minus_y",
            PrimitiveType.TRIANGLES,
            [
                make_vertex(0.5, -1.5, 0.0),
                make_vertex(0.0, -2.0, 0.0),
                make_vertex(0.3, -1.5, 0.0),
            ],
            0,  # No output
        ),
        # Trivial reject: all vertices outside same plane (+z)
        (
            "triangle_fully_outside_plus_z",
            PrimitiveType.TRIANGLES,
            [
                make_vertex(0.5, 0.0, 1.5),
                make_vertex(0.0, 0.0, 2.0),
                make_vertex(0.2, 0.0, 1.5),
            ],
            0,  # No output
        ),
        # Trivial reject: all vertices outside same plane (-z)
        (
            "triangle_fully_outside_minus_z",
            PrimitiveType.TRIANGLES,
            [
                make_vertex(0.5, 0.0, -1.5),
                make_vertex(0.0, 0.0, -2.0),
                make_vertex(0.3, 0.0, -1.5),
            ],
            0,  # No output
        ),
        # Clipping: triangle crossing +x plane
        (
            "triangle_clip_plus_x",
            PrimitiveType.TRIANGLES,
            [
                make_vertex(0.0, 0.5, 0.0),  # inside
                make_vertex(1.5, 0.0, 0.0),  # outside
                make_vertex(0.0, 0.0, 0.0),  # inside
            ],
            2,
        ),
        # Clipping: triangle crossing -x plane
        (
            "triangle_clip_minus_x",
            PrimitiveType.TRIANGLES,
            [
                make_vertex(0.0, 0.5, 0.0),  # inside
                make_vertex(-1.5, 0.0, 0.0),  # outside
                make_vertex(0.0, -0.5, 0.0),  # inside
            ],
            2,
        ),
        # Clipping: triangle crossing +y plane
        (
            "triangle_clip_minus_y",
            PrimitiveType.TRIANGLES,
            [
                make_vertex(0.0, 0.0, 0.0),  # inside
                make_vertex(0.5, 1.5, 0.0),  # outside
                make_vertex(0.5, 0.0, 0.0),  # inside
            ],
            2,
        ),
        # Clipping: triangle crossing -y plane
        (
            "triangle_clip_minus_y",
            PrimitiveType.TRIANGLES,
            [
                make_vertex(0.0, 0.0, 0.0),  # inside
                make_vertex(0.5, -1.5, 0.0),  # outside
                make_vertex(0.5, 0.0, 0.0),  # inside
            ],
            2,
        ),
        # Clipping: triangle crossing +z plane
        (
            "triangle_clip_plus_z",
            PrimitiveType.TRIANGLES,
            [
                make_vertex(0.0, 0.0, 0.0),  # inside
                make_vertex(0.5, 0.0, 1.5),  # outside
                make_vertex(0.0, 0.5, 0.0),  # inside
            ],
            2,
        ),
        # Clipping: triangle crossing -z plane
        (
            "triangle_clip_minus_z",
            PrimitiveType.TRIANGLES,
            [
                make_vertex(0.0, 0.0, 0.0),  # inside
                make_vertex(0.5, 0.0, -1.5),  # outside
                make_vertex(0.0, 0.5, 0.0),  # inside
            ],
            2,
        ),
        # Large triangle crossing multiple planes
        (
            "triangle_clip_multiple_planes",
            PrimitiveType.TRIANGLES,
            [
                make_vertex(-0.5, -0.5, 0.0),  # inside
                make_vertex(3.0, 0.0, 0.0),  # outside +x
                make_vertex(0.0, 3.0, 0.0),  # outside +y
            ],
            2,
        ),
        (
            "triangle_clip_multiple_planes_multi_split",
            PrimitiveType.TRIANGLES,
            [
                make_vertex(-0.5, -0.5, 0.0),  # inside
                make_vertex(1.5, 0.0, 0.0),  # outside +x
                make_vertex(0.0, 1.5, 0.0),  # outside +y
            ],
            3,
        ),
        (
            "triangle_clip_max_complexity",
            PrimitiveType.TRIANGLES,
            [
                make_vertex(-1.1, -1.1, 0.0),
                make_vertex(0.0, 1.1, -1.1),
                make_vertex(1.1, 0.0, 1.1),
            ],
            7,
        ),
        # Points - should pass through if inside
        (
            "point_inside",
            PrimitiveType.POINTS,
            [
                make_vertex(0.0, 0.0, 0.0),
            ],
            1,
        ),
        # Points - should be rejected if outside
        (
            "point_outside",
            PrimitiveType.POINTS,
            [
                make_vertex(2.0, 0.0, 0.0),
            ],
            0,
        ),
        # Line - fully inside
        (
            "line_inside",
            PrimitiveType.LINES,
            [
                make_vertex(0.0, 0.0, 0.0),
                make_vertex(0.5, 0.5, 0.0),
            ],
            1,
        ),
        # Line - fully outside
        (
            "line_outside",
            PrimitiveType.LINES,
            [
                make_vertex(2.0, 0.0, 0.0),
                make_vertex(2.5, 0.5, 0.0),
            ],
            0,
        ),
        # TODO: lines clipped
        # Clip with non-1 w component
        (
            "triangle_clip_nonunit_w",
            PrimitiveType.TRIANGLES,
            [
                make_vertex(0.0, 0.5, 0.0, w=2.0),  # inside
                make_vertex(3.0, 0.0, 0.0, w=2.0),  # outside
                make_vertex(0.0, 0.0, 0.0, w=2.0),  # inside
            ],
            2,
        ),
        (
            "triangle_clip_nonunit_w_mixed",
            PrimitiveType.TRIANGLES,
            [
                make_vertex(0.0, 0.5, 0.0, w=2.0),  # inside
                make_vertex(1.5, 0.0, 0.0, w=1.0),  # outside
                make_vertex(0.0, 0.0, 0.0, w=0.5),  # inside
            ],
            2,
        ),
    ],
)
def test_clipper(test_name, prim_type, input_vertices, expected_count):
    """Test primitive clipper with various cases."""
    dut = PrimitiveClipper()

    output_triangles = []

    async def output_checker(ctx, results):
        nonlocal output_triangles

        match prim_type:
            case PrimitiveType.POINTS:
                num_verts_per_prim = 1
            case PrimitiveType.LINES:
                num_verts_per_prim = 2
            case PrimitiveType.TRIANGLES:
                num_verts_per_prim = 3
            case _:
                raise ValueError("Unsupported primitive type")

        assert (
            len(results) % num_verts_per_prim == 0
        ), "Output vertex count not multiple of primitive size"

        output_prims = [
            results[i : i + num_verts_per_prim]
            for i in range(0, len(results), num_verts_per_prim)
        ]

        print(f"\nTest: {test_name}")
        print(f"Input vertices: {len(input_vertices)}")
        print(f"Output primitives: {len(output_prims)}")
        print(f"Expected count: {expected_count}")

        # For fully inside/outside cases, check exact count
        if "fully_inside" in test_name or "fully_outside" in test_name:
            assert (
                len(output_prims) == expected_count
            ), f"Expected {expected_count} primitives, got {len(output_prims)}"
        else:
            # For clipped cases, just check we got some output
            assert (
                len(output_prims) == expected_count
            ), f"Expected {expected_count} primitives, got {len(output_prims)}"

        print("Output Primitives:", output_prims)

        # Verify all output vertices are within NDC bounds
        for prim_idx, prim in enumerate(output_prims):
            for vert_idx, v in enumerate(prim):
                x, y, z, w = v.position_ndc
                print(f"{x.as_float()} {y.as_float()} {z.as_float()} {w.as_float()}")
                ndc_x = x.as_float() / w.as_float()
                ndc_y = y.as_float() / w.as_float()
                ndc_z = z.as_float() / w.as_float()
                err = 0.001
                assert (
                    -1.0 - err <= ndc_x <= 1.0 + err
                ), f"Primitive {prim_idx} Vertex {vert_idx} x out of NDC bounds: {ndc_x}"
                assert (
                    -1.0 - err <= ndc_y <= 1.0 + err
                ), f"Primitive {prim_idx} Vertex {vert_idx} y out of NDC bounds: {ndc_y}"
                assert (
                    -1.0 - err <= ndc_z <= 1.0 + err
                ), f"Primitive {prim_idx} Vertex {vert_idx} z out of NDC bounds: {ndc_z}"

    async def init_process(ctx):
        # Set primitive type
        ctx.set(dut.prim_type, prim_type)

    sim = Simulator(dut)
    sim.add_clock(1e-6)

    stream_testbench(
        sim,
        input_stream=dut.is_vertex,
        input_data=input_vertices,
        output_stream=dut.os_vertex,
        output_data_checker=output_checker,
        init_process=init_process,
        idle_for=3000,  # Allow time for clipping computation
    )

    sim.run()


def test_clipper_interpolation():
    """Test that clipping properly interpolates vertex attributes."""
    dut = PrimitiveClipper()

    # Triangle with one vertex outside, different colors to check interpolation
    input_vertices = [
        make_vertex(0.0, -0.5, 0.0, r=1.0, g=0.0, b=0.0),  # red, inside
        make_vertex(2.0, 0.0, 0.0, r=0.0, g=1.0, b=0.0),  # green, outside +x
        make_vertex(0.0, 0.5, 0.0, r=0.0, g=0.0, b=1.0),  # blue, inside
    ]

    output_triangles = []

    async def output_checker(ctx, results):
        nonlocal output_triangles

        assert len(results) % 3 == 0, "Output vertex count not multiple of 3"

        output_prims = [results[i : i + 3] for i in range(0, len(results), 3)]

        assert len(output_prims) == 2, "Expected 2 output triangles after clipping"

        print("\nClipping Interpolation Test")
        print(f"Input vertices: {len(input_vertices)}")
        print(f"Output triangles: {len(output_prims)}")

        for prim_idx, prim in enumerate(output_prims):
            for vert_idx, v in enumerate(prim):
                x, y, z, w = [p.as_float() for p in v.position_ndc]
                r, g, b, a = [s.as_float() for s in v.color]

                print(
                    f"Triangle {prim_idx} Vertex {vert_idx}: Color=({r}, {g}, {b}, {a}) Position=({x}, {y}, {z}, {w})"
                )

                # Check that color is interpolated between red and blue
                assert 0.0 <= r <= 1.0, "Red channel out of bounds"
                assert 0.0 <= g <= 1.0, "Green channel out of bounds"
                assert 0.0 <= b <= 1.0, "Blue channel out of bounds"

                # Since the outside vertex is green and clipped away,
                # we expect the output colors to be a mix of red and blue only.
                assert g <= 0.5, "Green channel should be low due to clipping"

                # color values should sum to approximately 1.0 (ignoring alpha)
                assert 0.9 <= r + g + b <= 1.1, "Color channels do not sum to ~1.0"
                assert (
                    r > 0.0 or b > 0.0
                ), "At least one of red or blue should be non-zero"
                assert a == 1.0, "Alpha channel should be 1.0"

                # colors should be in [(1,0,0), (0,0,1), (0.5,0.5,0), (0,0.5,0.5)]
                valid_colors = [
                    (1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0),
                    (0.5, 0.5, 0.0),
                    (0.0, 0.5, 0.5),
                ]

                color_found = any(
                    abs(r - vr) < 0.1 and abs(g - vg) < 0.1 and abs(b - vb) < 0.1
                    for (vr, vg, vb) in valid_colors
                )
                assert color_found, f"Unexpected color ({r}, {g}, {b}) after clipping"

    async def init_process(ctx):
        ctx.set(dut.prim_type, PrimitiveType.TRIANGLES)

    sim = Simulator(dut)
    sim.add_clock(1e-6)

    stream_testbench(
        sim,
        input_stream=dut.is_vertex,
        input_data=input_vertices,
        output_stream=dut.os_vertex,
        output_data_checker=output_checker,
        init_process=init_process,
        idle_for=300,
    )

    sim.run()
