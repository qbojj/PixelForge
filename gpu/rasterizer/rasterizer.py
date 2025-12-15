from amaranth import *
from amaranth.lib import stream, wiring
from amaranth.lib.wiring import In, Out
from transactron.utils.amaranth_ext.functions import max_value, min_value

from ..utils import math as gpu_math
from ..utils.layouts import FragmentLayout, FramebufferInfoLayout, RasterizerLayout
from ..utils.types import FixedPoint


class TriangleRasterizer(wiring.Component):
    """Triangle rasterizer using edge-function approach.

    Converts triangles in NDC space to fragments with perspective-correct interpolation.

    Input: RasterizerLayout stream (3 vertices per triangle)
    Output: FragmentLayout stream (one per covered pixel)

    TODO: support for lines and points (for now only triangles)
    """

    is_vertex: In(stream.Signature(RasterizerLayout))
    os_fragment: Out(stream.Signature(FragmentLayout))

    # Framebuffer configuration
    fb_info: In(FramebufferInfoLayout)

    def __init__(self, inv_steps: int = 4):
        super().__init__()
        self._inv_steps = inv_steps

    def elaborate(self, platform):
        m = Module()

        # Reciprocal unit for perspective-correct interpolation
        m.submodules.inv = inv = gpu_math.FixedPointInv(
            FixedPoint, steps=self._inv_steps
        )

        # Buffer for triangle vertices
        vtx = Array(Signal(RasterizerLayout) for _ in range(3))
        vtx_idx = Signal(range(3))

        # Screen-space coordinates (after viewport transform)
        screen_x = Array(Signal(FixedPoint) for _ in range(3))
        screen_y = Array(Signal(FixedPoint) for _ in range(3))

        # Bounding box (integer pixel coordinates)
        bb_min_x = Signal(signed(32))
        bb_min_y = Signal(signed(32))
        bb_max_x = Signal(signed(32))
        bb_max_y = Signal(signed(32))

        # Clamped bounding box (after scissor)
        min_x = Signal(unsigned(32))
        min_y = Signal(unsigned(32))
        max_x = Signal(unsigned(32))
        max_y = Signal(unsigned(32))
        completely_outside = Signal()

        # Current pixel being tested (integer coordinates)
        px = Signal(signed(32))
        py = Signal(signed(32))

        # Barycentric coordinates (unnormalized edge function values)
        w0 = Signal(FixedPoint)
        w1 = Signal(FixedPoint)
        w2 = Signal(FixedPoint)
        area = Signal(FixedPoint)
        area_recip = Signal(FixedPoint)

        # Interpolation accumulators
        inv_w_sum = Signal(FixedPoint)
        inv_w_sum_recip = Signal(FixedPoint)
        depth_num = Signal(FixedPoint)
        color_num = Array(Signal(FixedPoint) for _ in range(4))
        texcoord_num = Array(
            [Array([Signal(FixedPoint) for _ in range(4)]) for _ in range(2)]
        )

        # Edge function helper
        def edge_fn(ax, ay, bx, by, cx, cy):
            """Compute edge function: (B-A) × (C-A)"""
            return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

        # Default handshake
        m.d.comb += [
            self.is_vertex.ready.eq(0),
            self.os_fragment.valid.eq(0),
        ]

        with m.FSM():
            with m.State("COLLECT"):
                m.d.comb += self.is_vertex.ready.eq(1)
                with m.If(self.is_vertex.valid):
                    m.d.sync += vtx[vtx_idx].eq(self.is_vertex.payload)
                    with m.If(vtx_idx == 2):
                        m.d.sync += vtx_idx.eq(0)
                        m.next = "SETUP"
                    with m.Else():
                        m.d.sync += vtx_idx.eq(vtx_idx + 1)

            with m.State("SETUP"):
                # Viewport transform: NDC [-1,1] to screen space
                # screen_x = viewport_x + (ndc_x + 1) * viewport_width / 2
                # screen_y = viewport_y + (ndc_y + 1) * viewport_height / 2
                viewport_x = self.fb_info.viewport_x
                viewport_y = self.fb_info.viewport_y

                for i in range(3):
                    m.d.sync += [
                        screen_x[i].eq(
                            viewport_x
                            + (
                                (vtx[i].position_ndc[0] + 1)
                                * self.fb_info.viewport_width
                                >> 1
                            )
                        ),
                        screen_y[i].eq(
                            viewport_y
                            + (
                                (vtx[i].position_ndc[1] + 1)
                                * self.fb_info.viewport_height
                                >> 1
                            )
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

                # Compute bounding box
                m.d.sync += [
                    bb_min_x.eq(min_value(*[x.floor() for x in screen_x])),
                    bb_max_x.eq(max_value(*[x.ceil() for x in screen_x])),
                    bb_min_y.eq(min_value(*[y.floor() for y in screen_y])),
                    bb_max_y.eq(max_value(*[y.ceil() for y in screen_y])),
                ]

                m.next = "CULLING"

            with m.State("CULLING"):
                # Scissor rectangle bounds (convert to fixed-point for comparison)
                scissor_min_x = self.fb_info.scissor_offset_x
                scissor_min_y = self.fb_info.scissor_offset_y
                scissor_max_x = scissor_min_x + self.fb_info.scissor_width - 1
                scissor_max_y = scissor_min_y + self.fb_info.scissor_height - 1

                # Clamp bounding box to scissor rectangle
                m.d.sync += [
                    min_x.eq(Mux(bb_min_x < scissor_min_x, scissor_min_x, bb_min_x)),
                    min_y.eq(Mux(bb_min_y < scissor_min_y, scissor_min_y, bb_min_y)),
                    max_x.eq(Mux(bb_max_x > scissor_max_x, scissor_max_x, bb_max_x)),
                    max_y.eq(Mux(bb_max_y > scissor_max_y, scissor_max_y, bb_max_y)),
                ]

                m.d.comb += [
                    completely_outside.eq(
                        (bb_max_x < scissor_min_x)
                        | (bb_max_y < scissor_min_y)
                        | (bb_min_x > scissor_max_x)
                        | (bb_min_y > scissor_max_y)
                    ),
                ]

                with m.If(completely_outside):
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
                # Initialize scan at bounding box min
                m.d.sync += [
                    px.eq(min_x),
                    py.eq(min_y),
                ]
                m.next = "SCAN"

            with m.State("SCAN"):
                # Compute barycentric coordinates using fixed-point screen coords
                # Note: px, py are integer pixel coordinates; convert to fixed-point for edge fn
                # Fixed-point integer (px, 0) is represented as px << frac_bits
                # But for Amaranth, we construct this directly in the edge function

                e0 = edge_fn(screen_x[1], screen_y[1], screen_x[2], screen_y[2], px, py)
                e1 = edge_fn(screen_x[2], screen_y[2], screen_x[0], screen_y[0], px, py)
                e2 = edge_fn(screen_x[0], screen_y[0], screen_x[1], screen_y[1], px, py)
                # Check if pixel is inside triangle (all edge functions have same sign as area)
                inside = ((e0 >= 0) & (e1 >= 0) & (e2 >= 0)) | (
                    (e0 <= 0) & (e1 <= 0) & (e2 <= 0)
                )

                m.d.sync += [
                    w0.eq(e0),
                    w1.eq(e1),
                    w2.eq(e2),
                ]

                with m.If(inside):
                    m.next = "EMIT"
                with m.Else():
                    m.next = "ADVANCE"

            with m.State("EMIT"):
                # Compute perspective-correct interpolation numerators
                # inv_w_sum = w0*inv_w0 + w1*inv_w1 + w2*inv_w2
                m.d.comb += [
                    inv_w_sum.eq(
                        w0 * vtx[0].position_ndc[3]
                        + w1 * vtx[1].position_ndc[3]
                        + w2 * vtx[2].position_ndc[3]
                    ),
                ]

                m.d.sync += [
                    # Depth uses linear (non-perspective-correct) interpolation per spec
                    depth_num.eq(
                        w0 * vtx[0].position_ndc[2]
                        + w1 * vtx[1].position_ndc[2]
                        + w2 * vtx[2].position_ndc[2]
                    ),
                ]

                # Compute color numerators
                for i in range(4):
                    m.d.sync += color_num[i].eq(
                        w0 * vtx[0].color[i] * vtx[0].position_ndc[3]
                        + w1 * vtx[1].color[i] * vtx[1].position_ndc[3]
                        + w2 * vtx[2].color[i] * vtx[2].position_ndc[3]
                    )

                # Compute texcoord numerators
                for tex_idx in range(2):
                    for comp_idx in range(4):
                        m.d.sync += texcoord_num[tex_idx][comp_idx].eq(
                            w0
                            * vtx[0].texcoords[tex_idx][comp_idx]
                            * vtx[0].position_ndc[3]
                            + w1
                            * vtx[1].texcoords[tex_idx][comp_idx]
                            * vtx[1].position_ndc[3]
                            + w2
                            * vtx[2].texcoords[tex_idx][comp_idx]
                            * vtx[2].position_ndc[3]
                        )

                # Request reciprocal of inv_w_sum
                m.d.comb += [
                    inv.i.valid.eq(1),
                    inv.i.payload.eq(inv_w_sum),
                ]

                with m.If(inv.i.ready):
                    m.next = "INTERP_WAIT"

            with m.State("INTERP_WAIT"):
                # Wait for reciprocal to complete
                m.d.comb += inv.o.ready.eq(1)

                with m.If(inv.o.valid):
                    m.d.sync += inv_w_sum_recip.eq(inv.o.payload)
                    m.next = "OUTPUT"

            with m.State("OUTPUT"):
                # Output interpolated values
                # Depth uses normalized barycentric coords (linear interpolation per spec)
                m.d.comb += self.os_fragment.valid.eq(1)

                m.d.comb += [
                    self.os_fragment.p.coord_pos[0].eq(px),
                    self.os_fragment.p.coord_pos[1].eq(py),
                ]

                m.d.comb += self.os_fragment.p.depth.eq(depth_num * area_recip)

                for i in range(len(color_num)):
                    m.d.comb += self.os_fragment.p.color[i].eq(
                        color_num[i] * inv_w_sum_recip
                    )

                # Connect texcoords
                for tex_idx in range(len(self.os_fragment.p.texcoords)):
                    for comp_idx in range(len(self.os_fragment.p.texcoords[tex_idx])):
                        m.d.comb += self.os_fragment.p.texcoords[tex_idx][comp_idx].eq(
                            texcoord_num[tex_idx][comp_idx] * inv_w_sum_recip
                        )

                with m.If(self.os_fragment.ready):
                    m.next = "ADVANCE"

            with m.State("ADVANCE"):
                # Move to next pixel (scanline order)
                with m.If(px >= max_x):
                    m.d.sync += px.eq(min_x)
                    with m.If(py >= max_y):
                        # Done with triangle
                        m.next = "COLLECT"
                    with m.Else():
                        m.d.sync += py.eq(py + 1)
                        m.next = "SCAN"
                with m.Else():
                    m.d.sync += px.eq(px + 1)
                    m.next = "SCAN"

        return m
