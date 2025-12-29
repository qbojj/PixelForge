from amaranth import *
from amaranth.lib import stream, wiring
from amaranth.lib.wiring import In, Out

from ..utils import fixed
from ..utils import math as gpu_math
from ..utils.layouts import FragmentLayout, FramebufferInfoLayout, RasterizerLayout
from ..utils.transactron_utils import max_value, min_value


class TriangleRasterizer(wiring.Component):
    """Triangle rasterizer using edge-function approach.

    Converts triangles in NDC space to fragments with perspective-correct interpolation.

    Input: RasterizerLayout stream (3 vertices per triangle)
    Output: FragmentLayout stream (one per covered pixel)

    TODO: support for lines and points (for now only triangles)

    TODO: values overflow when
    """

    is_vertex: In(stream.Signature(RasterizerLayout))
    os_fragment: Out(stream.Signature(FragmentLayout))

    # Framebuffer configuration
    fb_info: In(FramebufferInfoLayout)
    ready: Out(1)

    def __init__(self, inv_steps: int = 4, subpixel_bits: int = 4):
        """Initialize the rasterizer.

        Args:
            inv_steps: Number of Newton-Raphson iterations for reciprocal
            subpixel_bits: Number of fractional bits for subpixel precision (default 4 = 16x16)
        """
        super().__init__()
        self._inv_steps = inv_steps
        self._subpixel_bits = subpixel_bits

    def elaborate(self, platform):
        m = Module()

        # Buffer for triangle vertices
        vtx = Array(Signal(RasterizerLayout) for _ in range(3))
        vtx_idx = Signal(range(3))

        fb_pos_int_bits = 12
        s_fb_type = fixed.SQ(fb_pos_int_bits, self._subpixel_bits)

        # Screen-space coordinates of vertices
        screen_x = Array(Signal(s_fb_type) for _ in range(3))
        screen_y = Array(Signal(s_fb_type) for _ in range(3))

        # Bounding box
        bb_min_x = Signal(signed(fb_pos_int_bits + 1))
        bb_min_y = Signal(signed(fb_pos_int_bits + 1))
        bb_max_x = Signal(signed(fb_pos_int_bits + 1))
        bb_max_y = Signal(signed(fb_pos_int_bits + 1))

        # Clamped bounding box (after scissor)
        min_x = Signal(unsigned(fb_pos_int_bits))
        min_y = Signal(unsigned(fb_pos_int_bits))
        max_x = Signal(unsigned(fb_pos_int_bits))
        max_y = Signal(unsigned(fb_pos_int_bits))

        # Current sample being tested
        px = Signal(fb_pos_int_bits)
        py = Signal(fb_pos_int_bits)

        # Barycentric coordinates (unnormalized edge function values)
        weight_shape = fixed.SQ(2 * fb_pos_int_bits + 1, 4)

        # Reciprocal unit for perspective-correct interpolation
        m.submodules.inv = inv = gpu_math.FixedPointInv(
            weight_shape, steps=self._inv_steps
        )

        w0 = Signal(weight_shape)
        w1 = Signal(weight_shape)
        w2 = Signal(weight_shape)
        area = Signal(weight_shape)
        area_recip = Signal.like(inv.o.payload)

        inv_w_sum = Signal(weight_shape)
        inv_w_sum_recip = Signal.like(inv.o.payload)

        # Fixed-point type for interpolation accumulators
        weight_persp = Array(Signal(fixed.UQ(1, 15)) for _ in range(3))
        weight_linear = Array(Signal(fixed.UQ(1, 15)) for _ in range(3))

        def vtx_attr(interp_type, v0, v1, v2):
            match interp_type:
                case "perspective":
                    return (
                        weight_persp[0] * v0
                        + weight_persp[1] * v1
                        + weight_persp[2] * v2
                    )
                case "linear":
                    return (
                        weight_linear[0] * v0
                        + weight_linear[1] * v1
                        + weight_linear[2] * v2
                    )
                case _:
                    raise ValueError(f"Unknown interpolation type: {interp_type}")

        # Edge function helper
        def edge_fn(ax, ay, bx, by, cx, cy):
            """Compute edge function: (B-A) × (C-A)"""
            return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

        with m.FSM():
            with m.State("COLLECT"):
                m.d.comb += [self.is_vertex.ready.eq(1), self.ready.eq(1)]
                with m.If(self.is_vertex.valid):
                    m.d.sync += vtx[vtx_idx].eq(self.is_vertex.payload)
                    with m.If(vtx_idx == 2):
                        m.d.sync += vtx_idx.eq(0)
                        m.next = "SETUP"
                    with m.Else():
                        m.d.sync += vtx_idx.eq(vtx_idx + 1)

            with m.State("SETUP"):
                # Viewport transform: NDC [-1,1] to screen space with subpixel precision
                # screen_x = (viewport_x + (ndc_x + 1) * viewport_width / 2)
                # screen_y = (viewport_y + (ndc_y + 1) * viewport_height / 2)

                for i in range(3):
                    m.d.sync += [
                        screen_x[i].eq(
                            self.fb_info.viewport_x
                            + (vtx[i].position_ndc[0] + 1) * self.fb_info.viewport_width
                            >> 1
                        ),
                        screen_y[i].eq(
                            self.fb_info.viewport_y
                            + (vtx[i].position_ndc[1] + 1)
                            * self.fb_info.viewport_height
                            >> 1
                        ),
                    ]

                m.next = "BOUNDING_BOX"

            with m.State("BOUNDING_BOX"):
                # Compute triangle area for barycentric coordinates
                area_val = edge_fn(
                    screen_x[0],
                    screen_y[0],
                    screen_x[1],
                    screen_y[1],
                    screen_x[2],
                    screen_y[2],
                )
                m.d.sync += area.eq(area_val)
                m.d.sync += Print("Triangle area:", area_val)

                # Compute bounding box in subpixel units
                # screen_x/y are already in subpixel units, just floor/ceil them
                m.d.sync += [
                    bb_min_x.eq(min_value(*[x.floor() for x in screen_x])),
                    bb_max_x.eq(max_value(*[x.floor() for x in screen_x])),
                    bb_min_y.eq(min_value(*[y.floor() for y in screen_y])),
                    bb_max_y.eq(max_value(*[y.floor() for y in screen_y])),
                ]

                m.next = "CULLING"

            with m.State("CULLING"):
                scissor_min_x = self.fb_info.scissor_offset_x
                scissor_min_y = self.fb_info.scissor_offset_y
                scissor_max_x = (
                    self.fb_info.scissor_offset_x + self.fb_info.scissor_width
                )
                scissor_max_y = (
                    self.fb_info.scissor_offset_y + self.fb_info.scissor_height
                )

                # Clamp bounding box to scissor rectangle
                m.d.sync += [
                    min_x.eq(max_value(bb_min_x, scissor_min_x)),
                    max_x.eq(min_value(bb_max_x, scissor_max_x)),
                    min_y.eq(max_value(bb_min_y, scissor_min_y)),
                    max_y.eq(min_value(bb_max_y, scissor_max_y)),
                ]

                outside_bits = Signal(4)
                m.d.comb += [
                    outside_bits[0].eq(bb_max_x < scissor_min_x),
                    outside_bits[1].eq(bb_max_y < scissor_min_y),
                    outside_bits[2].eq(bb_min_x > scissor_max_x),
                    outside_bits[3].eq(bb_min_y > scissor_max_y),
                ]

                with m.If(outside_bits.any()):
                    # Triangle is completely outside, skip it
                    m.next = "COLLECT"
                with m.Else():
                    # Request reciprocal of area for barycentric normalization
                    m.d.comb += [
                        inv.i.valid.eq(1),
                        inv.i.payload.eq(area_val),
                    ]

                    with m.If(inv.i.ready):
                        m.next = "AREA_RECIP_WAIT"

            with m.State("AREA_RECIP_WAIT"):
                # Wait for area reciprocal
                m.d.comb += inv.o.ready.eq(1)

                with m.If(inv.o.valid):
                    m.d.sync += area_recip.eq(inv.o.payload)
                    m.next = "SCAN_INIT"

            with m.State("SCAN_INIT"):
                # Initialize scan at bounding box min (in subpixel units)
                # Round to pixel centers for sample points
                m.d.sync += Print(
                    "Rasterizing triangle with area: ", area, ", inv: ", area_recip
                )
                m.d.sync += [
                    # Start at first pixel center within bounding box
                    px.eq(min_x),
                    py.eq(min_y),
                ]
                m.next = "SCAN"

            with m.State("SCAN"):
                # Compute barycentric coordinates using fixed-point screen coords (with subpixel precision)
                # px, py are integer subpixel units; convert to same fixed-point type as screen_x/y
                px_fp = Signal(s_fb_type)
                py_fp = Signal(s_fb_type)
                m.d.comb += [
                    # Centroid sampling
                    px_fp.eq(px + fixed.Const(0.5)),
                    py_fp.eq(py + fixed.Const(0.5)),
                ]

                edgev = Array(Signal(weight_shape) for _ in range(3))

                m.d.comb += [
                    edgev[0].eq(
                        edge_fn(
                            screen_x[1],
                            screen_y[1],
                            screen_x[2],
                            screen_y[2],
                            px_fp,
                            py_fp,
                        )
                    ),
                    edgev[1].eq(
                        edge_fn(
                            screen_x[2],
                            screen_y[2],
                            screen_x[0],
                            screen_y[0],
                            px_fp,
                            py_fp,
                        )
                    ),
                    edgev[2].eq(
                        edge_fn(
                            screen_x[0],
                            screen_y[0],
                            screen_x[1],
                            screen_y[1],
                            px_fp,
                            py_fp,
                        )
                    ),
                ]

                m.d.sync += [
                    w0.eq(edgev[0]),
                    w1.eq(edgev[1]),
                    w2.eq(edgev[2]),
                ]

                edge_pos = Signal(3)
                edge_neg = Signal(3)

                # Check if pixel is inside triangle (all edge functions have same sign as area)
                # TODO: do correct handling to include top-left edges only (per Vulkan spec)
                m.d.comb += edge_pos.eq(Cat([e >= 0 for e in edgev]))
                m.d.comb += edge_neg.eq(Cat([e <= 0 for e in edgev]))

                with m.If(edge_pos.all() | edge_neg.all()):
                    m.next = "EMIT"
                with m.Else():
                    m.next = "ADVANCE"

            with m.State("EMIT"):
                # Compute perspective-correct interpolation numerators
                # inv_w_sum = w0*inv_w0 + w1*inv_w1 + w2*inv_w2

                v0w = vtx[0].position_ndc[3]
                v1w = vtx[1].position_ndc[3]
                v2w = vtx[2].position_ndc[3]

                # Request reciprocal of inv_w_sum
                m.d.comb += [
                    inv_w_sum.eq(w0 * v0w + w1 * v1w + w2 * v2w),
                    inv.i.valid.eq(1),
                    inv.i.payload.eq(inv_w_sum),
                ]

                with m.If(inv.i.ready):
                    m.next = "INTERP_WAIT"

            with m.State("INTERP_WAIT"):
                # Wait for reciprocal to complete
                m.d.comb += inv.o.ready.eq(1)

                zero = fixed.Const(0.0)
                one = fixed.Const(1.0)

                wl0 = (w0 * area_recip).clamp(zero, one)
                wl1 = (w1 * area_recip).clamp(zero, one)

                m.d.sync += [
                    weight_linear[0].eq(wl0),
                    weight_linear[1].eq(wl1),
                    weight_linear[2].eq(one - wl0 - wl1),
                ]

                with m.If(inv.o.valid):
                    m.d.sync += inv_w_sum_recip.eq(inv.o.payload)
                    m.next = "CALC_PERSP_WEIGHTS"

            with m.State("CALC_PERSP_WEIGHTS"):
                wp0 = (w0 * vtx[0].position_ndc[3] * inv_w_sum_recip).clamp(zero, one)
                wp1 = (w1 * vtx[1].position_ndc[3] * inv_w_sum_recip).clamp(zero, one)

                m.d.sync += [
                    weight_persp[0].eq(wp0),
                    weight_persp[1].eq(wp1),
                    weight_persp[2].eq(one - wp0 - wp1),
                ]
                m.next = "OUTPUT"

            with m.State("OUTPUT"):
                m.d.sync += [
                    Print("Emitting fragment at (", px, ",", py, ")"),
                    Print("\tBarycentric weights:", *weight_linear),
                    Print("\tPersp weights:      ", *weight_persp),
                    Print(),
                ]
                m.d.comb += [
                    self.os_fragment.p.coord_pos[0].eq(px),
                    self.os_fragment.p.coord_pos[1].eq(py),
                ]

                # Depth uses linear interpolation per spec
                m.d.comb += self.os_fragment.p.depth.eq(
                    vtx_attr(
                        "linear",
                        vtx[0].position_ndc[2],
                        vtx[1].position_ndc[2],
                        vtx[2].position_ndc[2],
                    )
                )

                for i in range(len(self.os_fragment.p.color)):
                    m.d.comb += self.os_fragment.p.color[i].eq(
                        vtx_attr(
                            "perspective",
                            vtx[0].color[i],
                            vtx[1].color[i],
                            vtx[2].color[i],
                        )
                    )

                for tex_idx in range(len(self.os_fragment.p.texcoords)):
                    for comp_idx in range(len(self.os_fragment.p.texcoords[tex_idx])):
                        m.d.comb += self.os_fragment.p.texcoords[tex_idx][comp_idx].eq(
                            vtx_attr(
                                "perspective",
                                vtx[0].texcoords[tex_idx][comp_idx],
                                vtx[1].texcoords[tex_idx][comp_idx],
                                vtx[2].texcoords[tex_idx][comp_idx],
                            )
                        )

                m.d.comb += self.os_fragment.p.front_facing.eq(vtx[0].front_facing)
                m.d.comb += self.os_fragment.valid.eq(1)

                with m.If(self.os_fragment.ready):
                    m.next = "ADVANCE"

            with m.State("ADVANCE"):
                with m.If(px < max_x):
                    m.d.sync += px.eq(px + 1)
                    m.next = "SCAN"
                with m.Elif(py < max_y):
                    m.d.sync += px.eq(min_x)
                    m.d.sync += py.eq(py + 1)
                    m.next = "SCAN"
                with m.Else():
                    m.next = "COLLECT"

        return m
