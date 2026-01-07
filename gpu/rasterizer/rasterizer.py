from amaranth import *
from amaranth.lib import data, stream, wiring
from amaranth.lib.wiring import In, Out

from ..utils import fixed
from ..utils import math as gpu_math
from ..utils.layouts import (
    FragmentLayout,
    FramebufferInfoLayout,
    RasterizerLayout,
    texture_coord_shape,
)
from ..utils.stream import AnyDistributor, AnyRecombiner
from ..utils.transactron_utils import max_value, min_value, sum_value
from ..utils.types import FixedPoint_fb

_weight_shape = fixed.SQ(2 * texture_coord_shape.width + 1, FixedPoint_fb.f_bits)
_area_recip_shape = fixed.SQ(max(_weight_shape.f_bits, 2), _weight_shape.i_bits)


class TriangleContext(data.Struct):
    vtx: data.ArrayLayout(RasterizerLayout, 3)
    screen_x: data.ArrayLayout(
        fixed.SQ(texture_coord_shape.width, FixedPoint_fb.f_bits), 3
    )
    screen_y: data.ArrayLayout(
        fixed.SQ(texture_coord_shape.width, FixedPoint_fb.f_bits), 3
    )
    area: _weight_shape
    area_recip: _area_recip_shape
    min_x: unsigned(texture_coord_shape.width)
    min_y: unsigned(texture_coord_shape.width)
    max_x: unsigned(texture_coord_shape.width)
    max_y: unsigned(texture_coord_shape.width)


class PixelTask(data.Struct):
    px: unsigned(texture_coord_shape.width)
    py: unsigned(texture_coord_shape.width)


class TrianglePrep(wiring.Component):
    """Triangle setup: collects 3 vertices, applies viewport/scissor and outputs context."""

    is_vertex: In(stream.Signature(RasterizerLayout))
    o_ctx: Out(stream.Signature(TriangleContext))

    fb_info: In(FramebufferInfoLayout)
    ready: Out(1)

    def __init__(self, inv_steps: int = 4):
        super().__init__()
        self._inv_steps = inv_steps
        self._subpixel_bits = FixedPoint_fb.f_bits

    def elaborate(self, platform):
        m = Module()

        fb_pos_int_bits = texture_coord_shape.width
        s_fb_type = fixed.SQ(fb_pos_int_bits, self._subpixel_bits)
        weight_shape = fixed.SQ(2 * fb_pos_int_bits + 1, self._subpixel_bits)
        recip_shape = fixed.SQ(max(weight_shape.f_bits, 2), weight_shape.i_bits)

        vtx = Array(Signal(RasterizerLayout) for _ in range(3))
        vtx_idx = Signal(range(3))

        screen_x = Array(Signal(s_fb_type) for _ in range(3))
        screen_y = Array(Signal(s_fb_type) for _ in range(3))

        bb_min_x = Signal(signed(fb_pos_int_bits + 1))
        bb_min_y = Signal(signed(fb_pos_int_bits + 1))
        bb_max_x = Signal(signed(fb_pos_int_bits + 1))
        bb_max_y = Signal(signed(fb_pos_int_bits + 1))

        min_x = Signal(unsigned(fb_pos_int_bits))
        min_y = Signal(unsigned(fb_pos_int_bits))
        max_x = Signal(unsigned(fb_pos_int_bits))
        max_y = Signal(unsigned(fb_pos_int_bits))

        area = Signal(weight_shape)
        area_recip = Signal(recip_shape)

        m.submodules.inv = inv = gpu_math.FixedPointInv(
            weight_shape, steps=self._inv_steps
        )

        def edge_fn(ax, ay, bx, by, cx, cy):
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
                area_val = edge_fn(
                    screen_x[0],
                    screen_y[0],
                    screen_x[1],
                    screen_y[1],
                    screen_x[2],
                    screen_y[2],
                )

                m.d.sync += [
                    area.eq(area_val),
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
                    self.fb_info.scissor_offset_x + self.fb_info.scissor_width - 1
                )
                scissor_max_y = (
                    self.fb_info.scissor_offset_y + self.fb_info.scissor_height - 1
                )

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

                with m.If(outside_bits.any() | (area == 0)):
                    m.next = "COLLECT"
                with m.Else():
                    m.d.comb += [inv.i.valid.eq(1), inv.i.payload.eq(area)]
                    with m.If(inv.i.ready):
                        m.next = "AREA_RECIP_WAIT"

            with m.State("AREA_RECIP_WAIT"):
                m.d.comb += inv.o.ready.eq(1)
                with m.If(inv.o.valid):
                    m.d.sync += area_recip.eq(inv.o.payload)
                    m.next = "OUTPUT_CTX"

            with m.State("OUTPUT_CTX"):
                with m.If(self.o_ctx.ready | ~self.o_ctx.valid):
                    ctx = self.o_ctx.p
                    m.d.sync += [
                        ctx.area.eq(area),
                        ctx.area_recip.eq(area_recip),
                        ctx.min_x.eq(min_x),
                        ctx.min_y.eq(min_y),
                        ctx.max_x.eq(max_x),
                        ctx.max_y.eq(max_y),
                    ]
                    for i in range(3):
                        m.d.sync += [
                            ctx.vtx[i].eq(vtx[i]),
                            ctx.screen_x[i].eq(screen_x[i]),
                            ctx.screen_y[i].eq(screen_y[i]),
                        ]
                    m.d.sync += self.o_ctx.valid.eq(1)
                    m.next = "WAIT_CONSUME"

            with m.State("WAIT_CONSUME"):
                with m.If(self.o_ctx.ready):
                    m.d.sync += self.o_ctx.valid.eq(0)
                    m.next = "COLLECT"

        return m


class FragmentGenerator(wiring.Component):
    """Fragment generator: barycentric test + interpolation for a single pixel."""

    i_pxpy: In(stream.Signature(PixelTask))
    ctx: In(TriangleContext)
    ctx_valid: In(1)
    o_fragment: Out(stream.Signature(FragmentLayout))
    o_done: Out(1)

    def __init__(self, inv_steps: int = 4):
        super().__init__()
        self._inv_steps = inv_steps

    def elaborate(self, platform):
        m = Module()

        fb_pos_int_bits = texture_coord_shape.width
        s_fb_type = fixed.SQ(fb_pos_int_bits, FixedPoint_fb.f_bits)
        weight_shape = fixed.SQ(2 * fb_pos_int_bits + 1, FixedPoint_fb.f_bits)
        recip_shape = fixed.SQ(max(weight_shape.f_bits, 2), weight_shape.i_bits)

        px_lat = Signal(unsigned(fb_pos_int_bits))
        py_lat = Signal(unsigned(fb_pos_int_bits))
        px_fp_reg = Signal(s_fb_type)
        py_fp_reg = Signal(s_fb_type)

        w0 = Signal(weight_shape)
        w1 = Signal(weight_shape)
        w2 = Signal(weight_shape)

        # Shared multiplier to reduce DSP usage
        mul_a = Signal(s_fb_type)
        mul_b = Signal(s_fb_type)
        mul_p = Signal(weight_shape)
        m.d.comb += mul_p.eq(mul_a * mul_b)

        # Edge computation intermediates (differences and partial products)
        d0_ba_x = Signal(s_fb_type)
        d0_ba_y = Signal(s_fb_type)
        d0_pa_x = Signal(s_fb_type)
        d0_pa_y = Signal(s_fb_type)
        tmp0 = Signal(weight_shape)

        d1_ba_x = Signal(s_fb_type)
        d1_ba_y = Signal(s_fb_type)
        d1_pa_x = Signal(s_fb_type)
        d1_pa_y = Signal(s_fb_type)
        tmp1 = Signal(weight_shape)

        d2_ba_x = Signal(s_fb_type)
        d2_ba_y = Signal(s_fb_type)
        d2_pa_x = Signal(s_fb_type)
        d2_pa_y = Signal(s_fb_type)
        tmp2 = Signal(weight_shape)

        weight_linear = Array(Signal(fixed.UQ(1, 15)) for _ in range(3))
        weight_persp = Array(Signal(fixed.UQ(1, 15)) for _ in range(3))

        inv_w_sum = Signal(weight_shape)
        inv_w_sum_recip = Signal(recip_shape)

        m.submodules.inv = inv = gpu_math.FixedPointInv(
            weight_shape, steps=self._inv_steps
        )

        def edge_fn(ax, ay, bx, by, cx, cy):
            return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

        def vtx_attr(interp_type, v0, v1, v2):
            if interp_type == "perspective":
                return (
                    weight_persp[0] * v0 + weight_persp[1] * v1 + weight_persp[2] * v2
                )
            return weight_linear[0] * v0 + weight_linear[1] * v1 + weight_linear[2] * v2

        inside = Signal()
        inside_next = Signal()

        with m.FSM():
            with m.State("IDLE"):
                m.d.comb += self.i_pxpy.ready.eq(self.ctx_valid)
                with m.If(self.i_pxpy.valid & self.i_pxpy.ready & self.ctx_valid):
                    m.d.sync += [
                        px_lat.eq(self.i_pxpy.payload.px),
                        py_lat.eq(self.i_pxpy.payload.py),
                        px_fp_reg.eq(self.i_pxpy.payload.px + fixed.Const(0.5)),
                        py_fp_reg.eq(self.i_pxpy.payload.py + fixed.Const(0.5)),
                    ]
                    # Precompute differences for edge 0: A=V1, B=V2, P=(px,py)
                    m.d.sync += [
                        d0_ba_x.eq(self.ctx.screen_x[2] - self.ctx.screen_x[1]),
                        d0_ba_y.eq(self.ctx.screen_y[2] - self.ctx.screen_y[1]),
                        d0_pa_x.eq(
                            (self.i_pxpy.payload.px + fixed.Const(0.5))
                            - self.ctx.screen_x[1]
                        ),
                        d0_pa_y.eq(
                            (self.i_pxpy.payload.py + fixed.Const(0.5))
                            - self.ctx.screen_y[1]
                        ),
                    ]
                    m.next = "EDGE0_MUL1"

            with m.State("EDGE0_MUL1"):
                m.d.comb += [mul_a.eq(d0_ba_x), mul_b.eq(d0_pa_y)]
                m.d.sync += tmp0.eq(mul_p)
                m.next = "EDGE0_MUL2"

            with m.State("EDGE0_MUL2"):
                m.d.comb += [mul_a.eq(d0_ba_y), mul_b.eq(d0_pa_x)]
                m.d.sync += w0.eq(tmp0 - mul_p)
                # Prepare edge 1: A=V2, B=V0
                m.d.sync += [
                    d1_ba_x.eq(self.ctx.screen_x[0] - self.ctx.screen_x[2]),
                    d1_ba_y.eq(self.ctx.screen_y[0] - self.ctx.screen_y[2]),
                    d1_pa_x.eq(px_fp_reg - self.ctx.screen_x[2]),
                    d1_pa_y.eq(py_fp_reg - self.ctx.screen_y[2]),
                ]
                m.next = "EDGE1_MUL1"

            with m.State("EDGE1_MUL1"):
                m.d.comb += [mul_a.eq(d1_ba_x), mul_b.eq(d1_pa_y)]
                m.d.sync += tmp1.eq(mul_p)
                m.next = "EDGE1_MUL2"

            with m.State("EDGE1_MUL2"):
                m.d.comb += [mul_a.eq(d1_ba_y), mul_b.eq(d1_pa_x)]
                m.d.sync += w1.eq(tmp1 - mul_p)
                # Prepare edge 2: A=V0, B=V1
                m.d.sync += [
                    d2_ba_x.eq(self.ctx.screen_x[1] - self.ctx.screen_x[0]),
                    d2_ba_y.eq(self.ctx.screen_y[1] - self.ctx.screen_y[0]),
                    d2_pa_x.eq(px_fp_reg - self.ctx.screen_x[0]),
                    d2_pa_y.eq(py_fp_reg - self.ctx.screen_y[0]),
                ]
                m.next = "EDGE2_MUL1"

            with m.State("EDGE2_MUL1"):
                m.d.comb += [mul_a.eq(d2_ba_x), mul_b.eq(d2_pa_y)]
                m.d.sync += tmp2.eq(mul_p)
                m.next = "EDGE2_MUL2"

            with m.State("EDGE2_MUL2"):
                m.d.comb += [mul_a.eq(d2_ba_y), mul_b.eq(d2_pa_x)]
                m.d.sync += w2.eq(tmp2 - mul_p)
                m.next = "EDGE_INSIDE"

            with m.State("EDGE_INSIDE"):
                edge_pos = Signal(3)
                edge_neg = Signal(3)
                m.d.comb += [
                    edge_pos.eq(Cat([w0 >= 0, w1 >= 0, w2 >= 0])),
                    edge_neg.eq(Cat([w0 <= 0, w1 <= 0, w2 <= 0])),
                    inside_next.eq((edge_pos == 0b111) | (edge_neg == 0b111)),
                ]
                m.d.sync += inside.eq(inside_next)
                m.next = "INSIDE_CHECK"

            with m.State("INSIDE_CHECK"):
                with m.If(~inside):
                    m.d.comb += self.o_done.eq(1)
                    m.next = "IDLE"
                with m.Else():
                    zero = fixed.Const(0.0)
                    one = fixed.Const(1.0)
                    wl0 = (w0 * self.ctx.area_recip).clamp(zero, one)
                    wl1 = (w1 * self.ctx.area_recip).clamp(zero, one)
                    m.d.sync += [
                        weight_linear[0].eq(wl0),
                        weight_linear[1].eq(wl1),
                        weight_linear[2].eq(one - wl0 - wl1),
                    ]

                    v0w = self.ctx.vtx[0].position_ndc[3]
                    v1w = self.ctx.vtx[1].position_ndc[3]
                    v2w = self.ctx.vtx[2].position_ndc[3]
                    m.d.comb += [
                        inv_w_sum.eq(w0 * v0w + w1 * v1w + w2 * v2w),
                        inv.i.valid.eq(1),
                        inv.i.payload.eq(inv_w_sum),
                    ]
                    with m.If(inv.i.ready):
                        m.next = "INV_WAIT"

            with m.State("INV_WAIT"):
                m.d.comb += inv.o.ready.eq(1)
                with m.If(inv.o.valid):
                    zero = fixed.Const(0.0)
                    one = fixed.Const(1.0)
                    m.d.sync += inv_w_sum_recip.eq(inv.o.payload)
                    m.next = "PERSPECTIVE_WEIGHTS"

            with m.State("PERSPECTIVE_WEIGHTS"):
                wp0 = (w0 * self.ctx.vtx[0].position_ndc[3] * inv_w_sum_recip).clamp(
                    zero, one
                )
                wp1 = (w1 * self.ctx.vtx[1].position_ndc[3] * inv_w_sum_recip).clamp(
                    zero, one
                )
                m.d.sync += [
                    weight_persp[0].eq(wp0),
                    weight_persp[1].eq(wp1),
                    weight_persp[2].eq(one - wp0 - wp1),
                ]
                m.next = "OUTPUT"

            with m.State("OUTPUT"):
                m.d.comb += [
                    self.o_fragment.p.coord_pos[0].eq(px_lat),
                    self.o_fragment.p.coord_pos[1].eq(py_lat),
                    self.o_fragment.p.depth.eq(
                        vtx_attr(
                            "linear",
                            self.ctx.vtx[0].position_ndc[2],
                            self.ctx.vtx[1].position_ndc[2],
                            self.ctx.vtx[2].position_ndc[2],
                        )
                    ),
                    self.o_fragment.p.front_facing.eq(self.ctx.vtx[0].front_facing),
                ]

                for i in range(len(self.o_fragment.p.color)):
                    m.d.comb += self.o_fragment.p.color[i].eq(
                        vtx_attr(
                            "perspective",
                            self.ctx.vtx[0].color[i],
                            self.ctx.vtx[1].color[i],
                            self.ctx.vtx[2].color[i],
                        )
                    )

                for tex_idx in range(len(self.o_fragment.p.texcoords)):
                    for comp_idx in range(len(self.o_fragment.p.texcoords[tex_idx])):
                        m.d.comb += self.o_fragment.p.texcoords[tex_idx][comp_idx].eq(
                            vtx_attr(
                                "perspective",
                                self.ctx.vtx[0].texcoords[tex_idx][comp_idx],
                                self.ctx.vtx[1].texcoords[tex_idx][comp_idx],
                                self.ctx.vtx[2].texcoords[tex_idx][comp_idx],
                            )
                        )

                m.d.comb += self.o_fragment.valid.eq(1)
                with m.If(self.o_fragment.ready):
                    m.d.comb += self.o_done.eq(1)
                    m.next = "IDLE"

        return m


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
    ready: Out(1)

    def __init__(self, inv_steps: int = 4, num_generators: int = 1):
        super().__init__()
        self._inv_steps = inv_steps
        self._num_generators = num_generators
        self._subpixel_bits = FixedPoint_fb.f_bits

    def elaborate(self, platform):
        m = Module()

        m.submodules.setup = setup = TrianglePrep(inv_steps=self._inv_steps)

        ctx_buf = Signal(TriangleContext)
        ctx_active = Signal()
        tasks_done = Signal()
        inflight = Signal(range(self._num_generators + 1))

        px = Signal(unsigned(texture_coord_shape.width))
        py = Signal(unsigned(texture_coord_shape.width))

        m.d.comb += setup.fb_info.eq(self.fb_info)

        allow_new_triangle = ~ctx_active

        m.d.comb += [
            setup.is_vertex.valid.eq(self.is_vertex.valid & allow_new_triangle),
            setup.is_vertex.payload.eq(self.is_vertex.payload),
            self.is_vertex.ready.eq(setup.is_vertex.ready & allow_new_triangle),
            self.ready.eq(setup.ready & allow_new_triangle),
        ]

        m.d.comb += setup.o_ctx.ready.eq(~ctx_active)
        with m.If(setup.o_ctx.valid & setup.o_ctx.ready):
            m.d.sync += [
                ctx_buf.eq(setup.o_ctx.payload),
                ctx_active.eq(1),
                tasks_done.eq(0),
                inflight.eq(0),
                px.eq(setup.o_ctx.payload.min_x),
                py.eq(setup.o_ctx.payload.min_y),
            ]

        task_last_x = Signal()
        task_last_y = Signal()
        m.d.comb += [
            task_last_x.eq(px >= ctx_buf.max_x),
            task_last_y.eq(py >= ctx_buf.max_y),
        ]

        m.submodules.distrib = distrib = AnyDistributor(PixelTask, self._num_generators)
        m.submodules.recomb = recomb = AnyRecombiner(
            FragmentLayout, self._num_generators
        )

        tasks_avail = ctx_active & ~tasks_done
        m.d.comb += [
            distrib.i.valid.eq(tasks_avail),
            distrib.i.payload.px.eq(px),
            distrib.i.payload.py.eq(py),
        ]

        issue = Signal()
        m.d.comb += issue.eq(distrib.i.valid & distrib.i.ready)

        with m.If(issue):
            m.d.sync += inflight.eq(inflight + 1)
            with m.If(~task_last_x):
                m.d.sync += px.eq(px + 1)
            with m.Elif(~task_last_y):
                m.d.sync += [px.eq(ctx_buf.min_x), py.eq(py + 1)]
            with m.Else():
                m.d.sync += tasks_done.eq(1)

        done_total = Signal(range(self._num_generators + 1))

        fragments = []
        done_signals = []
        for idx in range(self._num_generators):
            m.submodules[f"fg_{idx}"] = fg = FragmentGenerator(
                inv_steps=self._inv_steps
            )
            fragments.append(fg)

            wiring.connect(m, distrib.o[idx], fg.i_pxpy)
            m.d.comb += [
                fg.ctx.eq(ctx_buf),
                fg.ctx_valid.eq(ctx_active),
            ]

            wiring.connect(m, fg.o_fragment, recomb.i[idx])
            done_signals.append(fg.o_done)

        m.d.comb += done_total.eq(sum_value(*done_signals))

        inflight_after = Signal.like(inflight)
        m.d.comb += inflight_after.eq(inflight + issue - done_total)

        with m.If(issue | (done_total > 0)):
            m.d.sync += inflight.eq(inflight_after)

        with m.If(tasks_done & (inflight_after == 0)):
            m.d.sync += ctx_active.eq(0)

        wiring.connect(m, recomb.o, wiring.flipped(self.os_fragment))

        return m
