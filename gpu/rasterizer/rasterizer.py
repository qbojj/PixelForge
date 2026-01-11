from amaranth import *
from amaranth.lib import data, stream, wiring
from amaranth.lib.wiring import In, Out

from ..utils import fixed
from ..utils import math as gpu_math
from ..utils.layouts import (
    FragmentLayout,
    FramebufferInfoLayout,
    RasterizerLayout,
    RasterizerLayoutNDC,
)
from ..utils.stream import AnyDistributor, AnyRecombiner
from ..utils.transactron_utils import max_value, min_value, sum_value
from ..utils.types import FixedPoint, FixedPoint_fb

_weight_shape = fixed.SQ(2 * FixedPoint_fb.i_bits + 1, FixedPoint_fb.f_bits)
_area_recip_shape = fixed.SQ(_weight_shape.f_bits, _weight_shape.i_bits)
_persp_div_shape = fixed.UQ(1, 17)


class PerspectiveDivide(wiring.Component):
    """Perspective divide: divides NDC coordinates by w to produce perspective-divided values.

    Input: RasterizerLayout stream (position_ndc with x, y, z, w)
    Output: RasterizerLayoutNDC stream (position_ndc with x/w, y/w, z/w, 1/w in UQ(1,17) format)
    """

    ready: Out(1)

    i_vertex: In(stream.Signature(RasterizerLayout))
    o_vertex: Out(stream.Signature(RasterizerLayoutNDC))

    def __init__(self, inv_steps: int = 4):
        super().__init__()
        self._inv_steps = inv_steps

    def elaborate(self, platform):
        m = Module()

        persp_type = _persp_div_shape

        # Reciprocal of w (1/w)
        inv_w = Signal(FixedPoint)

        # Temporary storage for vertex during processing
        vtx_buf = Signal(RasterizerLayout)

        # Shared multiplier for perspective division (one per coordinate)
        mul_a = Signal(FixedPoint)
        mul_b = Signal(FixedPoint)
        mul_p = Signal.like(mul_a * mul_b)
        m.d.comb += mul_p.eq(mul_a * mul_b)

        # Output storage
        div_x = Signal(persp_type)
        div_y = Signal(persp_type)
        div_z = Signal(persp_type)

        m.submodules.inv = inv = gpu_math.FixedPointInv(
            FixedPoint, steps=self._inv_steps
        )

        with m.FSM():
            with m.State("IDLE"):
                m.d.comb += self.ready.eq(1)
                m.d.comb += self.i_vertex.ready.eq(1)
                with m.If(self.i_vertex.valid):
                    m.d.sync += vtx_buf.eq(self.i_vertex.payload)
                    m.next = "START_INV"

            with m.State("START_INV"):
                m.d.sync += Print("W value: ", vtx_buf.position_ndc[3])
                m.d.comb += [
                    inv.i.valid.eq(1),
                    inv.i.payload.eq(vtx_buf.position_ndc[3]),
                ]
                with m.If(inv.i.ready):
                    m.next = "WAIT_INV"

            with m.State("WAIT_INV"):
                m.d.comb += inv.o.ready.eq(1)
                with m.If(inv.o.valid):
                    m.d.sync += inv_w.eq(inv.o.payload)
                    m.next = "DIVIDE_X"

            with m.State("DIVIDE_X"):
                m.d.comb += [mul_a.eq(vtx_buf.position_ndc[0]), mul_b.eq(inv_w)]
                m.d.sync += div_x.eq(((mul_p + 1) >> 1).saturate(persp_type))
                m.next = "DIVIDE_Y"

            with m.State("DIVIDE_Y"):
                m.d.comb += [mul_a.eq(vtx_buf.position_ndc[1]), mul_b.eq(inv_w)]
                m.d.sync += div_y.eq(((mul_p + 1) >> 1).saturate(persp_type))
                m.next = "DIVIDE_Z"

            with m.State("DIVIDE_Z"):
                m.d.comb += [mul_a.eq(vtx_buf.position_ndc[2]), mul_b.eq(inv_w)]
                m.d.sync += div_z.eq(((mul_p + 1) >> 1).saturate(persp_type))
                m.next = "OUTPUT"

            with m.State("OUTPUT"):
                m.d.comb += self.o_vertex.valid.eq(1)
                m.d.comb += [
                    self.o_vertex.p.position_ndc[0].eq(div_x),
                    self.o_vertex.p.position_ndc[1].eq(div_y),
                    self.o_vertex.p.position_ndc[2].eq(div_z),
                    self.o_vertex.p.w.eq(vtx_buf.position_ndc[3]),
                    self.o_vertex.p.inv_w.eq(inv_w),
                    self.o_vertex.p.color.eq(vtx_buf.color),
                    self.o_vertex.p.texcoords.eq(vtx_buf.texcoords),
                    self.o_vertex.p.front_facing.eq(vtx_buf.front_facing),
                ]
                with m.If(self.o_vertex.ready):
                    m.d.sync += Print("vtx in: ", vtx_buf)
                    m.d.sync += Print("vtx out: ", self.o_vertex.p)
                    m.next = "IDLE"

        return m


class TriangleContext(data.Struct):
    vtx: data.ArrayLayout(RasterizerLayoutNDC, 3)
    screen_x: data.ArrayLayout(FixedPoint_fb, 3)
    screen_y: data.ArrayLayout(FixedPoint_fb, 3)
    area: _weight_shape
    area_recip: _area_recip_shape
    min_x: unsigned(FixedPoint_fb.i_bits)
    min_y: unsigned(FixedPoint_fb.i_bits)
    max_x: unsigned(FixedPoint_fb.i_bits)
    max_y: unsigned(FixedPoint_fb.i_bits)


class PixelTask(data.Struct):
    px: unsigned(FixedPoint_fb.i_bits)
    py: unsigned(FixedPoint_fb.i_bits)


class TrianglePrep(wiring.Component):
    """Triangle setup: collects 3 vertices, applies viewport/scissor and outputs context."""

    is_vertex: In(stream.Signature(RasterizerLayoutNDC))
    o_ctx: Out(stream.Signature(TriangleContext))

    fb_info: In(FramebufferInfoLayout)
    ready: Out(1)

    def __init__(self, inv_steps: int = 4):
        super().__init__()
        self._inv_steps = inv_steps

    def elaborate(self, platform):
        m = Module()

        s_fb_type = FixedPoint_fb
        weight_shape = _weight_shape
        recip_shape = _area_recip_shape

        # Single shared multiplier (time-multiplexed)
        mul_a = Signal(s_fb_type)
        mul_b = Signal(s_fb_type)
        mul_p = Signal(weight_shape)
        m.d.comb += mul_p.eq(mul_a * mul_b)

        vtx = Signal(data.ArrayLayout(RasterizerLayoutNDC, 3))
        vtx_idx = Signal(range(3))

        screen_x = Array(Signal(s_fb_type) for _ in range(3))
        screen_y = Array(Signal(s_fb_type) for _ in range(3))

        bb_min_x = Signal(signed(FixedPoint_fb.i_bits + 1))
        bb_min_y = Signal(signed(FixedPoint_fb.i_bits + 1))
        bb_max_x = Signal(signed(FixedPoint_fb.i_bits + 1))
        bb_max_y = Signal(signed(FixedPoint_fb.i_bits + 1))

        min_x = Signal(unsigned(FixedPoint_fb.i_bits))
        min_y = Signal(unsigned(FixedPoint_fb.i_bits))
        max_x = Signal(unsigned(FixedPoint_fb.i_bits))
        max_y = Signal(unsigned(FixedPoint_fb.i_bits))

        area = Signal(weight_shape)
        area_recip = Signal(recip_shape)

        # Edge deltas for area calc using the shared multiplier
        dx10 = Signal(s_fb_type)
        dy10 = Signal(s_fb_type)
        dx20 = Signal(s_fb_type)
        dy20 = Signal(s_fb_type)
        area_temp = Signal(weight_shape)

        m.submodules.inv = inv = gpu_math.FixedPointInv(
            weight_shape, steps=self._inv_steps
        )

        calc_screen_idx = Signal(range(6))

        with m.FSM():
            with m.State("COLLECT"):
                m.d.comb += [self.is_vertex.ready.eq(1), self.ready.eq(vtx_idx == 0)]
                with m.If(self.is_vertex.valid):
                    m.d.sync += vtx[vtx_idx].eq(self.is_vertex.p)
                    with m.If(vtx_idx == 2):
                        m.d.sync += vtx_idx.eq(0)
                        m.d.sync += calc_screen_idx.eq(0)
                        m.next = "CALC_SCREEN"
                    with m.Else():
                        m.d.sync += vtx_idx.eq(vtx_idx + 1)

            with m.State("CALC_SCREEN"):
                scale = Signal(FixedPoint_fb)
                offset = Signal(FixedPoint_fb)
                scalar = Signal(fixed.UQ(16, 16))

                with m.If(calc_screen_idx < 3):
                    m.d.comb += [
                        scale.eq(self.fb_info.viewport_width),
                        offset.eq(self.fb_info.viewport_x),
                        scalar.eq(vtx[calc_screen_idx].position_ndc[0]),
                    ]
                with m.Else():
                    m.d.comb += [
                        scale.eq(self.fb_info.viewport_height),
                        offset.eq(self.fb_info.viewport_y),
                        scalar.eq(
                            vtx[(calc_screen_idx - 3).as_unsigned()].position_ndc[1]
                        ),
                    ]

                result = Signal(FixedPoint_fb)
                m.d.comb += result.eq((scale * scalar + offset).saturate(FixedPoint_fb))

                with m.If(calc_screen_idx < 3):
                    m.d.sync += screen_x[calc_screen_idx].eq(result)
                with m.Else():
                    m.d.sync += screen_y[calc_screen_idx - 3].eq(result)

                with m.If(calc_screen_idx == 5):
                    m.next = "BOUNDING_BOX"
                with m.Else():
                    m.d.sync += calc_screen_idx.eq(calc_screen_idx + 1)

            with m.State("BOUNDING_BOX"):
                # Prepare edge deltas for area = (x1-x0)*(y2-y0) - (y1-y0)*(x2-x0)
                m.d.sync += [
                    Print("screen_x: ", *screen_x),
                    Print("screen_y: ", *screen_y),
                    Print("vtx: ", *vtx),
                ]
                m.d.sync += [
                    dx10.eq(screen_x[1] - screen_x[0]),
                    dy10.eq(screen_y[1] - screen_y[0]),
                    dx20.eq(screen_x[2] - screen_x[0]),
                    dy20.eq(screen_y[2] - screen_y[0]),
                    bb_min_x.eq(min_value(*[x.floor() for x in screen_x])),
                    bb_max_x.eq(max_value(*[x.floor() for x in screen_x])),
                    bb_min_y.eq(min_value(*[y.floor() for y in screen_y])),
                    bb_max_y.eq(max_value(*[y.floor() for y in screen_y])),
                ]
                m.next = "AREA_MUL1"

            with m.State("AREA_MUL1"):
                m.d.comb += [mul_a.eq(dx10), mul_b.eq(dy20)]
                m.d.sync += area_temp.eq(mul_p)
                m.next = "AREA_MUL2"

            with m.State("AREA_MUL2"):
                m.d.comb += [mul_a.eq(dy10), mul_b.eq(dx20)]
                m.d.sync += area.eq(area_temp - mul_p)
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
                    m.d.sync += Print("Culling triangle: outside scissor or zero area")
                    m.d.sync += Print("outside_bits: {}", outside_bits)
                    m.d.sync += Print("area: {}", area)
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

        s_fb_type = FixedPoint_fb
        weight_shape = _weight_shape
        recip_shape = _area_recip_shape

        px_lat = Signal(unsigned(FixedPoint_fb.i_bits))
        py_lat = Signal(unsigned(FixedPoint_fb.i_bits))
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

        weight_linear = Signal(data.ArrayLayout(fixed.UQ(1, 17), 3))
        weight_persp = Signal(data.ArrayLayout(fixed.UQ(1, 17), 3))

        inv_w_sum = Signal(weight_shape)
        inv_w_sum_recip = Signal(recip_shape)

        # Shared multiplier for weights/perspective sums (reduces implicit DSPs)
        weight_mul_a = Signal(weight_shape)
        weight_mul_b = Signal(recip_shape)
        weight_mul_p = Signal.like(weight_mul_a * weight_mul_b)
        m.d.comb += weight_mul_p.eq(weight_mul_a * weight_mul_b)

        # Shared multiplier for interpolation (UQ(1,17) output with saturation)
        mul_a_interp = Signal(fixed.UQ(1, 17))
        mul_b_interp = Signal(fixed.UQ(1, 17))
        mul_p_interp = Signal.like(mul_a_interp * mul_b_interp)
        m.d.comb += mul_p_interp.eq(mul_a_interp * mul_b_interp)

        # Interpolation results (depth and color components)
        depth_sat = Signal(_persp_div_shape)
        color_sat = Array(Signal(_persp_div_shape) for _ in range(4))

        m.submodules.inv = inv = gpu_math.FixedPointInv(
            weight_shape, steps=self._inv_steps
        )

        zero = fixed.Const(0.0)
        one = fixed.Const(1.0)

        persp_pre = Signal(data.ArrayLayout(weight_shape, 2))

        with m.FSM():
            with m.State("IDLE"):
                m.d.comb += self.i_pxpy.ready.eq(self.ctx_valid)
                with m.If(self.i_pxpy.valid & self.ctx_valid):
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

                m.d.comb += [
                    weight_mul_a.eq(w0),
                    weight_mul_b.eq(self.ctx.area_recip),
                ]
                m.d.sync += weight_linear[0].eq(weight_mul_p.clamp(zero, one))

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

                m.d.comb += [
                    weight_mul_a.eq(w0),
                    weight_mul_b.eq(self.ctx.vtx[0].w),
                ]
                m.d.sync += inv_w_sum.eq(weight_mul_p)

                m.next = "EDGE2_MUL1"

            with m.State("EDGE2_MUL1"):
                m.d.comb += [mul_a.eq(d2_ba_x), mul_b.eq(d2_pa_y)]
                m.d.sync += tmp2.eq(mul_p)

                m.d.comb += [
                    weight_mul_a.eq(w1),
                    weight_mul_b.eq(self.ctx.area_recip),
                ]
                m.d.sync += weight_linear[1].eq(weight_mul_p.clamp(zero, one))

                m.next = "EDGE2_MUL2"

            with m.State("EDGE2_MUL2"):
                m.d.comb += [mul_a.eq(d2_ba_y), mul_b.eq(d2_pa_x)]
                m.d.sync += w2.eq(tmp2 - mul_p)

                m.d.sync += weight_linear[2].eq(
                    one - weight_linear[0] - weight_linear[1]
                )

                m.d.comb += [
                    weight_mul_a.eq(w1),
                    weight_mul_b.eq(self.ctx.vtx[1].w),
                ]
                m.d.sync += inv_w_sum.eq(inv_w_sum + weight_mul_p)

                m.next = "EDGE_INSIDE"

            with m.State("EDGE_INSIDE"):
                edge_pos = Signal(3)
                edge_neg = Signal(3)
                m.d.comb += [
                    edge_pos.eq(Cat([w0 >= 0, w1 >= 0, w2 >= 0])),
                    edge_neg.eq(Cat([w0 <= 0, w1 <= 0, w2 <= 0])),
                ]

                m.d.comb += [
                    weight_mul_a.eq(w2),
                    weight_mul_b.eq(self.ctx.vtx[2].w),
                ]
                m.d.sync += inv_w_sum.eq(inv_w_sum + weight_mul_p)
                m.d.comb += inv.i.payload.eq(inv_w_sum + weight_mul_p)

                with m.If(edge_pos.all() | edge_neg.all()):
                    m.d.comb += inv.i.valid.eq(1)
                    with m.If(inv.i.ready):
                        m.next = "GET_PRE_PERSP_0"
                with m.Else():
                    m.d.comb += self.o_done.eq(1)
                    m.next = "IDLE"

            with m.State("GET_PRE_PERSP_0"):
                m.d.comb += [weight_mul_a.eq(w0), weight_mul_b.eq(self.ctx.vtx[0].w)]
                m.d.sync += persp_pre[0].eq(weight_mul_p)

                m.d.comb += [
                    mul_a_interp.eq(self.ctx.vtx[0].position_ndc[2]),
                    mul_b_interp.eq(weight_linear[0]),
                ]
                m.d.sync += depth_sat.eq(mul_p_interp.saturate(_persp_div_shape))

                m.next = "GET_PRE_PERSP_1"

            with m.State("GET_PRE_PERSP_1"):
                m.d.comb += [weight_mul_a.eq(w1), weight_mul_b.eq(self.ctx.vtx[1].w)]
                m.d.sync += persp_pre[1].eq(weight_mul_p)

                m.d.comb += [
                    mul_a_interp.eq(self.ctx.vtx[1].position_ndc[2]),
                    mul_b_interp.eq(weight_linear[1]),
                ]
                m.d.sync += depth_sat.eq(
                    (depth_sat + mul_p_interp).saturate(_persp_div_shape)
                )

                m.next = "INV_WAIT"

            with m.State("INV_WAIT"):
                m.d.comb += [
                    mul_a_interp.eq(self.ctx.vtx[2].position_ndc[2]),
                    mul_b_interp.eq(weight_linear[2]),
                ]
                m.d.comb += inv.o.ready.eq(1)
                with m.If(inv.o.valid):
                    m.d.sync += inv_w_sum_recip.eq(inv.o.payload)
                    m.d.sync += depth_sat.eq(
                        (depth_sat + mul_p_interp).saturate(_persp_div_shape)
                    )

                    m.next = "PERSPECTIVE_W0_M1"

            with m.State("PERSPECTIVE_W0_M1"):
                m.d.comb += [
                    weight_mul_a.eq(persp_pre[0]),
                    weight_mul_b.eq(inv_w_sum_recip),
                ]
                m.d.sync += weight_persp[0].eq(weight_mul_p.clamp(zero, one))

                m.next = "PERSPECTIVE_W1"

            with m.State("PERSPECTIVE_W1"):
                m.d.comb += [
                    weight_mul_a.eq(persp_pre[1]),
                    weight_mul_b.eq(inv_w_sum_recip),
                ]
                m.d.sync += weight_persp[1].eq((weight_mul_p).clamp(zero, one))

                m.d.comb += [
                    mul_a_interp.eq(self.ctx.vtx[0].color[0]),
                    mul_b_interp.eq(weight_persp[0]),
                ]
                m.d.sync += color_sat[0].eq(mul_p_interp.saturate(_persp_div_shape))

                m.next = "INTERP_COLOR_0_1"

            with m.State("INTERP_COLOR_0_1"):
                m.d.sync += weight_persp[2].eq(one - weight_persp[0] - weight_persp[1])

                m.d.comb += [
                    mul_a_interp.eq(self.ctx.vtx[1].color[0]),
                    mul_b_interp.eq(weight_persp[1]),
                ]
                m.d.sync += color_sat[0].eq(
                    (color_sat[0] + mul_p_interp).saturate(_persp_div_shape)
                )
                m.next = "INTERP_COLOR_0_2"

            with m.State("INTERP_COLOR_0_2"):
                m.d.sync += Print("Weights linear ", *weight_linear)
                m.d.sync += Print("Weights persp ", *weight_persp)

                m.d.comb += [
                    mul_a_interp.eq(self.ctx.vtx[2].color[0]),
                    mul_b_interp.eq(weight_persp[2]),
                ]
                m.d.sync += color_sat[0].eq(
                    (color_sat[0] + mul_p_interp).saturate(_persp_div_shape)
                )
                m.next = "INTERP_COLOR_1"

            with m.State("INTERP_COLOR_1"):
                m.d.comb += [
                    mul_a_interp.eq(self.ctx.vtx[0].color[1]),
                    mul_b_interp.eq(weight_persp[0]),
                ]
                m.d.sync += color_sat[1].eq(mul_p_interp.saturate(_persp_div_shape))
                m.next = "INTERP_COLOR_1_1"

            with m.State("INTERP_COLOR_1_1"):
                m.d.comb += [
                    mul_a_interp.eq(self.ctx.vtx[1].color[1]),
                    mul_b_interp.eq(weight_persp[1]),
                ]
                m.d.sync += color_sat[1].eq(
                    (color_sat[1] + mul_p_interp).saturate(_persp_div_shape)
                )
                m.next = "INTERP_COLOR_1_2"

            with m.State("INTERP_COLOR_1_2"):
                m.d.comb += [
                    mul_a_interp.eq(self.ctx.vtx[2].color[1]),
                    mul_b_interp.eq(weight_persp[2]),
                ]
                m.d.sync += color_sat[1].eq(
                    (color_sat[1] + mul_p_interp).saturate(_persp_div_shape)
                )
                m.next = "INTERP_COLOR_2"

            with m.State("INTERP_COLOR_2"):
                m.d.comb += [
                    mul_a_interp.eq(self.ctx.vtx[0].color[2]),
                    mul_b_interp.eq(weight_persp[0]),
                ]
                m.d.sync += color_sat[2].eq(mul_p_interp.saturate(_persp_div_shape))
                m.next = "INTERP_COLOR_2_1"

            with m.State("INTERP_COLOR_2_1"):
                m.d.comb += [
                    mul_a_interp.eq(self.ctx.vtx[1].color[2]),
                    mul_b_interp.eq(weight_persp[1]),
                ]
                m.d.sync += color_sat[2].eq(
                    (color_sat[2] + mul_p_interp).saturate(_persp_div_shape)
                )
                m.next = "INTERP_COLOR_2_2"

            with m.State("INTERP_COLOR_2_2"):
                m.d.comb += [
                    mul_a_interp.eq(self.ctx.vtx[2].color[2]),
                    mul_b_interp.eq(weight_persp[2]),
                ]
                m.d.sync += color_sat[2].eq(
                    (color_sat[2] + mul_p_interp).saturate(_persp_div_shape)
                )
                m.next = "INTERP_COLOR_3"

            with m.State("INTERP_COLOR_3"):
                m.d.comb += [
                    mul_a_interp.eq(self.ctx.vtx[0].color[3]),
                    mul_b_interp.eq(weight_persp[0]),
                ]
                m.d.sync += color_sat[3].eq(mul_p_interp.saturate(_persp_div_shape))
                m.next = "INTERP_COLOR_3_1"

            with m.State("INTERP_COLOR_3_1"):
                m.d.comb += [
                    mul_a_interp.eq(self.ctx.vtx[1].color[3]),
                    mul_b_interp.eq(weight_persp[1]),
                ]
                m.d.sync += color_sat[3].eq(
                    (color_sat[3] + mul_p_interp).saturate(_persp_div_shape)
                )
                m.next = "INTERP_COLOR_3_2"

            with m.State("INTERP_COLOR_3_2"):
                m.d.comb += [
                    mul_a_interp.eq(self.ctx.vtx[2].color[3]),
                    mul_b_interp.eq(weight_persp[2]),
                ]
                m.d.sync += color_sat[3].eq(
                    (color_sat[3] + mul_p_interp).saturate(_persp_div_shape)
                )
                m.next = "OUTPUT"

            with m.State("OUTPUT"):
                m.d.comb += [
                    self.o_fragment.p.coord_pos[0].eq(px_lat),
                    self.o_fragment.p.coord_pos[1].eq(py_lat),
                    self.o_fragment.p.depth.eq(depth_sat.clamp(zero, one)),
                    self.o_fragment.p.color[0].eq(color_sat[0].clamp(zero, one)),
                    self.o_fragment.p.color[1].eq(color_sat[1].clamp(zero, one)),
                    self.o_fragment.p.color[2].eq(color_sat[2].clamp(zero, one)),
                    self.o_fragment.p.color[3].eq(color_sat[3].clamp(zero, one)),
                    self.o_fragment.p.front_facing.eq(self.ctx.vtx[0].front_facing),
                ]

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

    is_vertex: In(stream.Signature(RasterizerLayoutNDC))
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

        px = Signal(unsigned(FixedPoint_fb.i_bits))
        py = Signal(unsigned(FixedPoint_fb.i_bits))

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
