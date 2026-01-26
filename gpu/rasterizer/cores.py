from amaranth import *
from amaranth.lib import data, stream, wiring
from amaranth.lib.wiring import In, Out

from gpu.utils.stream import WideStreamOutput

from ..utils import fixed
from ..utils import math as gpu_math
from ..utils.layouts import (
    FragmentLayout,
    FramebufferInfoLayout,
    RasterizerLayout,
    RasterizerLayoutNDC,
    num_textures,
)
from ..utils.stream import AnyDistributor, AnyRecombiner
from ..utils.transactron_utils import max_value, min_value, popcount
from ..utils.types import CullFace, FixedPoint, FixedPoint_fb, FrontFace, PrimitiveType
from .layouts import PrimitiveAssemblyConfigLayout

_weight_shape = fixed.SQ(2 * FixedPoint_fb.i_bits + 1, FixedPoint_fb.f_bits)
_area_recip_shape = fixed.SQ(_weight_shape.f_bits, _weight_shape.i_bits)
_persp_div_shape = fixed.UQ(1, 17)


class PrimitiveClipper(wiring.Component):
    """Primitive clipper for rasterizer stage.

    - Input: stream of `RasterizerLayout` vertices (assembled order depends on primitive type).
    - Output: stream of `RasterizerLayout` vertices with primitives fully inside the clip volume.
    - Registers: primitive type (point/line/triangle), cull face, winding order.
    - Culling: applied for triangles only (front/back based on area sign and winding).
    - Clipping: trivial accept/reject against NDC cube.
    """

    i: In(stream.Signature(RasterizerLayout))
    o: Out(stream.Signature(RasterizerLayout))

    prim_type: In(PrimitiveType)
    ready: Out(1)

    def elaborate(self, platform):
        m = Module()
        # Reciprocal unit for t computation (t = num / den = num * inv(den))
        m.submodules.inv = inv = gpu_math.FixedPointInv(FixedPoint, steps=4)

        buf = Array(Signal(RasterizerLayout) for _ in range(3))
        idx = Signal(range(3))
        needed = Signal(range(4))

        # Clipping work buffers (up to 9 vertices after clipping against 6 planes)
        clip_buf = Array(
            Array(Signal(RasterizerLayout) for _ in range(9)) for _ in range(2)
        )
        clip_count = Array(Signal(range(10)) for _ in range(2))
        clip_src = Signal()  # ping-pong buffer index
        clip_plane = Signal(range(6))  # current plane being clipped against
        clip_idx = Signal(range(9))  # current vertex index during clipping
        # Interpolation helper signals
        t_num = Signal(FixedPoint)
        t_den = Signal(FixedPoint)
        t_recip = Signal(FixedPoint)
        emit_next = Signal()  # entering case: emit intersection and next vertex
        edge_curr_idx = Signal(range(9))
        edge_next_idx = Signal(range(9))

        # Single shared multiplier for t and interpolation (time-multiplexed)
        mul_in_a = Signal(FixedPoint)
        mul_in_b = Signal(FixedPoint)
        mul_p = Signal(FixedPoint)
        m.d.comb += mul_p.eq(mul_in_a * mul_in_b)

        t_reg = Signal(FixedPoint)
        lerp_a_reg = Signal(FixedPoint)
        lerp_phase = Signal()  # 0=setup multiply, 1=accumulate/write
        # Number of components to lerp: 4 pos + 4 color + 4*num_textures texcoords
        total_fields = 8 + 4 * num_textures
        lerp_stage = Signal(range(max(1, total_fields)))
        out_count_reg = Signal(range(10))
        src_reg = Signal()
        dst_reg = Signal()

        # Primitive vertex count based on register
        with m.Switch(self.prim_type):
            with m.Case(PrimitiveType.POINTS):
                m.d.comb += needed.eq(1)
            with m.Case(PrimitiveType.LINES):
                m.d.comb += needed.eq(2)
            with m.Default():
                m.d.comb += needed.eq(3)

        m.submodules.w_out = w_out = WideStreamOutput(self.o.p.shape(), 3)
        wiring.connect(m, wiring.flipped(self.o), w_out.o)

        with m.If(w_out.o.ready & w_out.o.valid):
            m.d.sync += Print("clipper vtx out: ", w_out.o.p)

        with m.FSM():
            with m.State("COLLECT"):
                m.d.comb += self.ready.eq((idx == 0) & w_out.i.ready)
                m.d.comb += self.i.ready.eq(1)
                with m.If(self.i.valid):
                    m.d.sync += buf[idx].eq(self.i.payload)
                    m.d.sync += Print("clipper vtx in: ", self.i.payload)
                    with m.If(idx == (needed - 1)):
                        m.d.sync += idx.eq(0)
                        m.next = "CHECK"
                    with m.Else():
                        m.d.sync += idx.eq(idx + 1)

            with m.State("CHECK"):
                # Compute clip codes for trivial accept/reject (no polygon splitting).
                # Helper function to compute clip code for a vertex
                def compute_clip_code(vtx):
                    x, y, z, w = vtx.position_ndc
                    bits = [
                        x > w,  # +x
                        x < -w,  # -x
                        y > w,  # +y
                        y < -w,  # -y
                        z > w,  # +z
                        z < -w,  # -z
                    ]
                    return Cat(bits)

                codes = Array(Signal(6) for _ in range(3))
                for i in range(3):
                    m.d.comb += codes[i].eq(compute_clip_code(buf[i]))

                m.d.sync += [
                    Print(
                        Format(
                            "vtx0: {}, vtx1: {}, vtx2: {}",
                            buf[0].position_ndc,
                            buf[1].position_ndc,
                            buf[2].position_ndc,
                        )
                    ),
                    Print(
                        Format(
                            "Clip codes: {:06b}, {:06b}, {:06b}",
                            codes[0],
                            codes[1],
                            codes[2],
                        )
                    ),
                ]
                with m.If((codes[0] & codes[1] & codes[2]) != 0):
                    m.d.sync += Print("Trivial reject")
                    # Fully outside; drop primitive.
                    m.next = "COLLECT"
                with m.Elif((codes[0] | codes[1] | codes[2]) == 0):
                    # Fully inside; forward primitive.
                    m.d.comb += [
                        w_out.i.p.data[0].eq(buf[0]),
                        w_out.i.p.data[1].eq(buf[1]),
                        w_out.i.p.data[2].eq(buf[2]),
                        w_out.i.p.n.eq(needed),
                        w_out.i.valid.eq(1),
                    ]
                    with m.If(w_out.i.ready):
                        m.d.sync += Print("Trivial accept")
                        m.next = "COLLECT"
                with m.Else():
                    # Needs clipping (only triangles and lines should reach here).
                    m.next = "CLIP"

            with m.State("CLIP"):
                # Sutherland-Hodgman clipping for triangles only
                # Initialize first buffer with triangle vertices
                with m.If(needed == 3):
                    m.d.sync += [
                        clip_buf[0][0].eq(buf[0]),
                        clip_buf[0][1].eq(buf[1]),
                        clip_buf[0][2].eq(buf[2]),
                        clip_count[0].eq(3),
                        clip_plane.eq(0),
                        clip_src.eq(0),
                    ]
                    m.next = "CLIP_PLANE"
                with m.Else():
                    # TODO: implement line clipping (Cohen-Sutherland or Liang-Barsky)
                    m.next = "COLLECT"

            with m.State("CLIP_PLANE"):
                # Clip against current plane
                # Planes: 0: +x, 1: -x, 2: +y, 3: -y, 4: +z, 5: -z
                src = clip_src
                dst = ~clip_src
                m.d.sync += Print(
                    Format(
                        "CLIP_PLANE enter plane {} src {} dst {} count {}",
                        clip_plane,
                        src,
                        dst,
                        clip_count[src],
                    )
                )

                # If source polygon is empty or clipped away, skip to emit
                with m.If(clip_count[src] == 0):
                    m.next = "CLIP_EMIT"
                with m.Elif(clip_plane > 5):
                    # All planes processed
                    m.next = "CLIP_EMIT"
                with m.Else():
                    m.d.sync += [
                        clip_count[dst].eq(0),
                        clip_idx.eq(0),
                    ]
                    m.next = "CLIP_EDGE"

            with m.State("CLIP_EDGE"):
                # Clip edge across current plane
                src = clip_src
                dst = ~clip_src
                curr_idx = clip_idx
                next_idx = Mux(clip_idx == clip_count[src] - 1, 0, clip_idx + 1)

                curr_v = clip_buf[src][curr_idx]
                next_v = clip_buf[src][next_idx]

                m.d.sync += Print(
                    Format(
                        "CLIP_EDGE plane {} src {} curr {} next {} count {}",
                        clip_plane,
                        src,
                        curr_idx,
                        next_idx,
                        clip_count[src],
                    )
                )

                c_x, c_y, c_z, c_w = curr_v.position_ndc
                n_x, n_y, n_z, n_w = next_v.position_ndc

                m.d.sync += Print(
                    Format(
                        "   curr ({}, {}, {}, {}), next ({}, {}, {}, {})",
                        c_x,
                        c_y,
                        c_z,
                        c_w,
                        n_x,
                        n_y,
                        n_z,
                        n_w,
                    )
                )

                # Compute distance from plane in clip space using ±w comparisons
                # plane 0: +x (x <= w) => dist = w - x
                # plane 1: -x (x >= -w) => dist = x + w
                # plane 2: +y (y <= w) => dist = w - y
                # plane 3: -y (y >= -w) => dist = y + w
                # plane 4: +z (z <= w) => dist = w - z (far)
                # plane 5: -z (z >= -w) => dist = z + w (near)
                curr_dist = Signal(FixedPoint)
                next_dist = Signal(FixedPoint)

                with m.Switch(clip_plane):
                    with m.Case(0):
                        m.d.comb += [
                            curr_dist.eq(c_w - c_x),
                            next_dist.eq(n_w - n_x),
                        ]
                    with m.Case(1):
                        m.d.comb += [
                            curr_dist.eq(c_x + c_w),
                            next_dist.eq(n_x + n_w),
                        ]
                    with m.Case(2):
                        m.d.comb += [
                            curr_dist.eq(c_w - c_y),
                            next_dist.eq(n_w - n_y),
                        ]
                    with m.Case(3):
                        m.d.comb += [
                            curr_dist.eq(c_y + c_w),
                            next_dist.eq(n_y + n_w),
                        ]
                    with m.Case(4):
                        m.d.comb += [
                            curr_dist.eq(c_w - c_z),
                            next_dist.eq(n_w - n_z),
                        ]
                    with m.Case(5):
                        m.d.comb += [
                            curr_dist.eq(c_z + c_w),
                            next_dist.eq(n_z + n_w),
                        ]

                curr_inside = curr_dist >= 0
                next_inside = next_dist >= 0

                m.d.sync += Print(
                    Format(
                        "   dists {} {} inside {} {}",
                        curr_dist,
                        next_dist,
                        curr_inside,
                        next_inside,
                    )
                )

                out_count = clip_count[dst]

                # Prepare indices for use across states
                m.d.sync += [
                    edge_curr_idx.eq(curr_idx),
                    edge_next_idx.eq(next_idx),
                ]

                # Both inside: emit next vertex and move to next edge
                with m.If(curr_inside & next_inside):
                    m.d.sync += [
                        clip_buf[dst][out_count].eq(next_v),
                        clip_count[dst].eq(out_count + 1),
                    ]
                    # Move to next edge
                    with m.If(clip_idx == clip_count[src] - 1):
                        # Done with this plane
                        m.d.sync += clip_src.eq(dst)
                        with m.If(clip_plane == 5):
                            # All planes done
                            m.next = "CLIP_EMIT"
                        with m.Else():
                            m.d.sync += clip_plane.eq(clip_plane + 1)
                            m.next = "CLIP_PLANE"
                    with m.Else():
                        m.d.sync += clip_idx.eq(clip_idx + 1)
                # Exiting: emit intersection
                with m.Elif(curr_inside & ~next_inside):
                    # Compute t via reciprocal: request inv(t_den)
                    m.d.sync += [
                        t_num.eq(curr_dist),
                        t_den.eq(curr_dist - next_dist),
                        emit_next.eq(0),
                    ]
                    m.next = "CLIP_INV_REQ"
                # Entering: emit intersection and next vertex
                with m.Elif(~curr_inside & next_inside):
                    # Compute t via reciprocal: request inv(t_den)
                    m.d.sync += [
                        t_num.eq(curr_dist),
                        t_den.eq(curr_dist - next_dist),
                        emit_next.eq(1),
                    ]
                    m.next = "CLIP_INV_REQ"
                # Both outside: emit nothing, move to next edge
                with m.Else():
                    # Move to next edge
                    with m.If(clip_idx == clip_count[src] - 1):
                        # Done with this plane
                        m.d.sync += clip_src.eq(dst)
                        with m.If(clip_plane == 5):
                            # All planes done
                            m.next = "CLIP_EMIT"
                        with m.Else():
                            m.d.sync += clip_plane.eq(clip_plane + 1)
                            m.next = "CLIP_PLANE"
                    with m.Else():
                        m.d.sync += clip_idx.eq(clip_idx + 1)

            # Request reciprocal for t_den
            with m.State("CLIP_INV_REQ"):
                m.d.comb += [
                    inv.i.valid.eq(1),
                    inv.i.payload.eq(t_den),
                ]
                with m.If(inv.i.ready):
                    m.next = "CLIP_INV_WAIT"

            with m.State("CLIP_INV_WAIT"):
                m.d.comb += inv.o.ready.eq(1)
                with m.If(inv.o.valid):
                    m.d.sync += [
                        t_recip.eq(inv.o.p),
                        src_reg.eq(clip_src),
                        dst_reg.eq(~clip_src),
                        out_count_reg.eq(clip_count[~clip_src]),
                    ]
                    m.next = "CLIP_T_MUL"

            with m.State("CLIP_T_MUL"):
                # Multiply t = t_num * t_recip using the shared multiplier
                m.d.sync += [
                    mul_in_a.eq(t_num),
                    mul_in_b.eq(t_recip),
                ]
                m.next = "CLIP_T_LATCH"

            with m.State("CLIP_T_LATCH"):
                # Latch t and prepare for iterative interpolation
                m.d.sync += [
                    t_reg.eq(mul_p.reshape(FixedPoint.f_bits)),
                    lerp_stage.eq(0),
                    lerp_phase.eq(0),
                ]
                m.next = "CLIP_LERP"

            with m.State("CLIP_LERP"):
                # Iteratively interpolate components using the shared multiplier
                src = src_reg
                dst = dst_reg
                curr_idx = edge_curr_idx
                next_idx = edge_next_idx
                curr_v = clip_buf[src][curr_idx]
                next_v = clip_buf[src][next_idx]
                out_idx = out_count_reg

                result = Signal(FixedPoint)
                m.d.comb += result.eq(
                    (lerp_a_reg + mul_p.reshape(FixedPoint.f_bits)).reshape(
                        FixedPoint.f_bits
                    )
                )

                with m.If(lerp_phase == 0):
                    # Setup multiply for current component
                    with m.Switch(lerp_stage):
                        for i in range(4):
                            with m.Case(i):
                                a_v = fixed.Value.cast(
                                    curr_v.position_ndc[i], FixedPoint.f_bits
                                )
                                b_v = fixed.Value.cast(
                                    next_v.position_ndc[i], FixedPoint.f_bits
                                )
                                m.d.sync += [
                                    lerp_a_reg.eq(a_v),
                                    mul_in_a.eq(t_reg),
                                    mul_in_b.eq(b_v - a_v),
                                ]
                        for i in range(4):
                            with m.Case(4 + i):
                                a_v = fixed.Value.cast(
                                    curr_v.color[i], FixedPoint.f_bits
                                )
                                b_v = fixed.Value.cast(
                                    next_v.color[i], FixedPoint.f_bits
                                )
                                m.d.sync += [
                                    lerp_a_reg.eq(a_v),
                                    mul_in_a.eq(t_reg),
                                    mul_in_b.eq(b_v - a_v),
                                ]
                        if num_textures > 0:
                            for t_idx in range(num_textures):
                                for comp in range(4):
                                    stage_idx = 8 + t_idx * 4 + comp
                                    with m.Case(stage_idx):
                                        a_v = fixed.Value.cast(
                                            curr_v.texcoords[t_idx][comp],
                                            FixedPoint.f_bits,
                                        )
                                        b_v = fixed.Value.cast(
                                            next_v.texcoords[t_idx][comp],
                                            FixedPoint.f_bits,
                                        )
                                        m.d.sync += [
                                            lerp_a_reg.eq(a_v),
                                            mul_in_a.eq(t_reg),
                                            mul_in_b.eq(b_v - a_v),
                                        ]
                    m.d.sync += lerp_phase.eq(1)
                with m.Else():
                    # Write interpolated component and advance
                    with m.Switch(lerp_stage):
                        for i in range(4):
                            with m.Case(i):
                                m.d.sync += (
                                    clip_buf[dst][out_idx].position_ndc[i].eq(result)
                                )
                        for i in range(4):
                            with m.Case(4 + i):
                                m.d.sync += clip_buf[dst][out_idx].color[i].eq(result)
                        if num_textures > 0:
                            for t_idx in range(num_textures):
                                for comp in range(4):
                                    stage_idx = 8 + t_idx * 4 + comp
                                    with m.Case(stage_idx):
                                        m.d.sync += (
                                            clip_buf[dst][out_idx]
                                            .texcoords[t_idx][comp]
                                            .eq(result)
                                        )

                    with m.If(lerp_stage == (total_fields - 1)):
                        # Finished interpolated vertex; finalize bookkeeping
                        with m.If(emit_next):
                            m.d.sync += [
                                clip_buf[dst][out_idx + 1].eq(next_v),
                                clip_count[dst].eq(out_idx + 2),
                            ]
                        with m.Else():
                            m.d.sync += clip_count[dst].eq(out_idx + 1)

                        # Continue to next edge or plane
                        with m.If(clip_idx == clip_count[src] - 1):
                            m.d.sync += clip_src.eq(dst)
                            with m.If(clip_plane == 5):
                                m.next = "CLIP_EMIT"
                            with m.Else():
                                m.d.sync += clip_plane.eq(clip_plane + 1)
                                m.next = "CLIP_PLANE"
                        with m.Else():
                            m.d.sync += clip_idx.eq(clip_idx + 1)
                            m.next = "CLIP_EDGE"
                        m.d.sync += lerp_phase.eq(0)
                    with m.Else():
                        m.d.sync += [
                            lerp_stage.eq(lerp_stage + 1),
                            lerp_phase.eq(0),
                        ]

            with m.State("CLIP_EMIT"):
                # Emit clipped polygon as triangle fan
                final_buf = clip_src
                final_count = clip_count[final_buf]

                m.d.sync += Print(
                    Format(
                        "CLIP_EMIT count {} plane {} src {}",
                        final_count,
                        clip_plane,
                        final_buf,
                    )
                )

                with m.If(final_count < 3):
                    # Clipped to nothing
                    m.next = "COLLECT"
                with m.Else():
                    # Emit first triangle
                    m.d.sync += clip_idx.eq(2)
                    m.next = "CLIP_OUTPUT"

            with m.State("CLIP_OUTPUT"):
                # Output triangle as fan: (0, idx-1, idx)
                final_buf = clip_src
                final_count = clip_count[final_buf]

                m.d.comb += [
                    w_out.i.p.data[0].eq(clip_buf[final_buf][0]),
                    w_out.i.p.data[1].eq(clip_buf[final_buf][clip_idx - 1]),
                    w_out.i.p.data[2].eq(clip_buf[final_buf][clip_idx]),
                    w_out.i.p.n.eq(3),
                    w_out.i.valid.eq(1),
                ]
                with m.If(w_out.i.ready):
                    # Check if this is the last triangle
                    with m.If(clip_idx == final_count - 1):
                        # Done, go back to COLLECT
                        m.next = "COLLECT"
                    with m.Else():
                        # More triangles to emit, increment and stay in CLIP_OUTPUT
                        m.d.sync += clip_idx.eq(clip_idx + 1)

        return m


class PerspectiveDivide(wiring.Component):
    """Perspective divide: divides NDC coordinates by w to produce perspective-divided values.

    Input: RasterizerLayout stream (position_ndc with x, y, z, w)
    Output: RasterizerLayoutNDC stream (position_ndc with x/w, y/w, z/w, 1/w in UQ(1,17) format)
    """

    ready: Out(1)

    i: In(stream.Signature(RasterizerLayout))
    o: Out(stream.Signature(RasterizerLayoutNDC))

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
                m.d.comb += self.i.ready.eq(1)
                with m.If(self.i.valid):
                    m.d.sync += vtx_buf.eq(self.i.payload)
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
                m.d.comb += self.o.valid.eq(1)
                m.d.comb += [
                    self.o.p.position_ndc[0].eq(div_x),
                    self.o.p.position_ndc[1].eq(div_y),
                    self.o.p.position_ndc[2].eq(div_z),
                    self.o.p.inv_w.eq(inv_w),
                    self.o.p.color.eq(vtx_buf.color),
                    self.o.p.texcoords.eq(vtx_buf.texcoords),
                ]
                with m.If(self.o.ready):
                    m.d.sync += Print("Input vertex: ", vtx_buf)
                    m.d.sync += Print("Output vertex: ", self.o.p)
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
    front_facing: unsigned(1)


class PixelTask(data.Struct):
    px: unsigned(FixedPoint_fb.i_bits)
    py: unsigned(FixedPoint_fb.i_bits)


class TrianglePrep(wiring.Component):
    """Triangle setup: collects 3 vertices, applies viewport/scissor and outputs context."""

    i: In(stream.Signature(RasterizerLayoutNDC))
    o: Out(stream.Signature(TriangleContext))

    fb_info: In(FramebufferInfoLayout)
    pa_conf: In(PrimitiveAssemblyConfigLayout)
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

        tri_front_facing = Signal()

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
                m.d.comb += [self.i.ready.eq(1), self.ready.eq(vtx_idx == 0)]
                with m.If(self.i.valid):
                    m.d.sync += vtx[vtx_idx].eq(self.i.p)
                    with m.If(vtx_idx == 2):
                        m.d.sync += vtx_idx.eq(0)
                        m.d.sync += calc_screen_idx.eq(0)
                        m.next = "CALC_SCREEN"
                    with m.Else():
                        m.d.sync += vtx_idx.eq(vtx_idx + 1)

            with m.State("CALC_SCREEN"):
                scale = Signal(FixedPoint_fb)
                offset = Signal(FixedPoint_fb)
                scalar = Signal.like(vtx[0].position_ndc[0])

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

                ff = Signal()
                with m.Switch(self.pa_conf.winding):
                    with m.Case(FrontFace.CCW):
                        m.d.comb += ff.eq(area > 0)
                    with m.Case(FrontFace.CW):
                        m.d.comb += ff.eq(area < 0)

                m.d.sync += tri_front_facing.eq(ff)

                with m.If(
                    ff & ((self.pa_conf.cull & CullFace.FRONT) == CullFace.FRONT)
                ):
                    m.d.sync += Print("Culling front face")
                    m.next = "COLLECT"
                with m.Elif(
                    ~ff & ((self.pa_conf.cull & CullFace.BACK) == CullFace.BACK)
                ):
                    m.d.sync += Print("Culling back face")
                    m.next = "COLLECT"
                with m.Elif(outside_bits.any() | (area == 0)):
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
                m.d.comb += self.o.p.area.eq(area)
                m.d.comb += self.o.p.area_recip.eq(area_recip)
                m.d.comb += self.o.p.min_x.eq(min_x)
                m.d.comb += self.o.p.min_y.eq(min_y)
                m.d.comb += self.o.p.max_x.eq(max_x)
                m.d.comb += self.o.p.max_y.eq(max_y)
                m.d.comb += self.o.p.front_facing.eq(tri_front_facing)
                m.d.comb += [self.o.p.vtx[i].eq(vtx[i]) for i in range(3)]
                m.d.comb += [self.o.p.screen_x[i].eq(screen_x[i]) for i in range(3)]
                m.d.comb += [self.o.p.screen_y[i].eq(screen_y[i]) for i in range(3)]
                m.d.comb += self.o.valid.eq(1)
                with m.If(self.o.ready):
                    m.d.sync += Print("Output ctx: ", self.o.p)
                    m.next = "COLLECT"

        return m


class FragmentGenerator(wiring.Component):
    """Fragment generator: barycentric test + interpolation for a single pixel."""

    i: In(stream.Signature(PixelTask))
    o: Out(stream.Signature(FragmentLayout))

    ctx: In(TriangleContext)

    d_x: In(data.ArrayLayout(FixedPoint_fb, 3))
    d_y: In(data.ArrayLayout(FixedPoint_fb, 3))

    winding_ccw: In(1)
    is_top_left: In(3)

    o_done: Out(1)

    def __init__(self, inv_steps: int = 4):
        super().__init__()
        self._inv_steps = inv_steps

    @staticmethod
    def max_pipelined_elements() -> int:
        return 1

    def elaborate(self, platform):
        m = Module()

        s_fb_type = FixedPoint_fb
        weight_shape = _weight_shape
        recip_shape = _area_recip_shape

        px_lat = Signal(unsigned(FixedPoint_fb.i_bits))
        py_lat = Signal(unsigned(FixedPoint_fb.i_bits))
        px_fp_reg = Signal(s_fb_type)
        py_fp_reg = Signal(s_fb_type)

        w = Signal(data.ArrayLayout(weight_shape, 3))

        # Shared multiplier to reduce DSP usage
        mul_a = Signal(s_fb_type)
        mul_b = Signal(s_fb_type)
        mul_p = Signal(weight_shape)
        m.d.comb += mul_p.eq(mul_a * mul_b)

        # Edge computation intermediates (differences and partial products)
        dp_x = Signal(data.ArrayLayout(s_fb_type, 3))
        dp_y = Signal(data.ArrayLayout(s_fb_type, 3))

        weight_linear = Signal(data.ArrayLayout(fixed.UQ(1, 17), 3))
        weight_persp = Signal(data.ArrayLayout(fixed.UQ(1, 17), 3))

        inv_w_sum = Signal(FixedPoint)
        inv_w_sum_recip = Signal(FixedPoint)

        # Shared multiplier for weights/perspective sums (reduces implicit DSPs)
        weight_mul_a = Signal(weight_shape)
        weight_mul_b = Signal(recip_shape)
        weight_mul_p = Signal.like(weight_mul_a * weight_mul_b)
        m.d.comb += weight_mul_p.eq(weight_mul_a * weight_mul_b)

        # Perspective-correct bary multipliers
        persp_mul_a = Signal(FixedPoint)  # persp_pre is FixedPoint
        persp_mul_b = Signal(FixedPoint)  # 1/w is FixedPoint
        persp_mul_p = Signal.like(persp_mul_a * persp_mul_b)
        m.d.comb += persp_mul_p.eq(persp_mul_a * persp_mul_b)

        # Shared multiplier for interpolation (UQ(1,17) output with saturation)
        mul_a_interp = Signal(fixed.UQ(1, 17))
        mul_b_interp = Signal(fixed.UQ(1, 17))
        mul_p_interp = Signal.like(mul_a_interp * mul_b_interp)
        m.d.comb += mul_p_interp.eq(mul_a_interp * mul_b_interp)

        # Interpolation results (depth and color components)
        depth_sat = Signal(_persp_div_shape)
        color_sat = Array(Signal(_persp_div_shape) for _ in range(4))

        m.submodules.inv = inv = gpu_math.FixedPointInv(
            FixedPoint, steps=self._inv_steps
        )

        zero = fixed.Const(0.0)
        one = fixed.Const(1.0)

        persp_pre = Signal(data.ArrayLayout(weight_shape, 2))

        edge_inside = Signal(3)

        with m.FSM():
            with m.State("IDLE"):
                m.d.comb += self.i.ready.eq(1)
                with m.If(self.i.valid):
                    m.d.sync += [
                        px_lat.eq(self.i.payload.px),
                        py_lat.eq(self.i.payload.py),
                        px_fp_reg.eq(self.i.payload.px + fixed.Const(0.5)),
                        py_fp_reg.eq(self.i.payload.py + fixed.Const(0.5)),
                    ]
                    m.next = "CALC_EDGE_OFFS"

            with m.State("CALC_EDGE_OFFS"):
                m.d.sync += [
                    dp_x[0].eq(px_fp_reg - self.ctx.screen_x[1]),
                    dp_y[0].eq(py_fp_reg - self.ctx.screen_y[1]),
                    dp_x[1].eq(px_fp_reg - self.ctx.screen_x[2]),
                    dp_y[1].eq(py_fp_reg - self.ctx.screen_y[2]),
                    dp_x[2].eq(px_fp_reg - self.ctx.screen_x[0]),
                    dp_y[2].eq(py_fp_reg - self.ctx.screen_y[0]),
                ]
                m.next = "EDGE0_MUL1"

            with m.State("EDGE0_MUL1"):
                m.d.comb += [mul_a.eq(self.d_x[0]), mul_b.eq(dp_y[0])]
                m.d.sync += w[0].eq(mul_p)
                m.next = "EDGE0_MUL2"

            with m.State("EDGE0_MUL2"):
                m.d.comb += [mul_a.eq(self.d_y[0]), mul_b.eq(dp_x[0])]
                m.d.sync += w[0].eq(w[0] - mul_p)
                m.next = "EDGE1_MUL1"

            with m.State("EDGE1_MUL1"):
                m.d.comb += [mul_a.eq(self.d_x[1]), mul_b.eq(dp_y[1])]
                m.d.sync += w[1].eq(mul_p)

                m.d.comb += [
                    weight_mul_a.eq(w[0]),
                    weight_mul_b.eq(self.ctx.area_recip),
                ]
                m.d.sync += weight_linear[0].eq(weight_mul_p)

                m.next = "EDGE1_MUL2"

            with m.State("EDGE1_MUL2"):
                m.d.comb += [mul_a.eq(self.d_y[1]), mul_b.eq(dp_x[1])]
                m.d.sync += w[1].eq(w[1] - mul_p)

                m.d.comb += [
                    persp_mul_a.eq(weight_linear[0] << 4),  # UQ1.17 to Q13.13
                    persp_mul_b.eq(self.ctx.vtx[0].inv_w),
                ]
                m.d.sync += inv_w_sum.eq(persp_mul_p)
                m.d.sync += persp_pre[0].eq(persp_mul_p)

                m.next = "EDGE2_MUL1"

            with m.State("EDGE2_MUL1"):
                m.d.comb += [mul_a.eq(self.d_x[2]), mul_b.eq(dp_y[2])]
                m.d.sync += w[2].eq(mul_p)

                m.d.comb += [
                    weight_mul_a.eq(w[1]),
                    weight_mul_b.eq(self.ctx.area_recip),
                ]
                m.d.sync += weight_linear[1].eq(weight_mul_p)

                m.next = "EDGE2_MUL2"

            with m.State("EDGE2_MUL2"):
                m.d.comb += [mul_a.eq(self.d_y[2]), mul_b.eq(dp_x[2])]
                m.d.sync += w[2].eq(w[2] - mul_p)

                m.d.sync += weight_linear[2].eq(
                    one - weight_linear[0] - weight_linear[1]
                )

                m.d.comb += [
                    persp_mul_a.eq(weight_linear[1] << 4),
                    persp_mul_b.eq(self.ctx.vtx[1].inv_w),
                ]
                m.d.sync += inv_w_sum.eq(inv_w_sum + persp_mul_p)
                m.d.sync += persp_pre[1].eq(persp_mul_p)

                m.next = "EDGE_INSIDE"

            with m.State("EDGE_INSIDE"):
                m.d.comb += [
                    # perform edge tests with top-left rule
                    edge_inside[i].eq(
                        Mux(self.winding_ccw, w[i] > 0, w[i] < 0)
                        | ((w[i] == 0) & self.is_top_left[i])
                    )
                    for i in range(3)
                ]

                m.d.comb += [
                    persp_mul_a.eq(weight_linear[2] << 4),
                    persp_mul_b.eq(self.ctx.vtx[2].inv_w),
                ]
                m.d.comb += inv.i.payload.eq(inv_w_sum + persp_mul_p)

                with m.If(edge_inside.all()):
                    m.d.comb += inv.i.valid.eq(1)
                    with m.If(inv.i.ready):
                        m.next = "GET_PRE_PERSP_0"
                with m.Else():
                    m.d.comb += self.o_done.eq(1)
                    m.next = "IDLE"

            with m.State("GET_PRE_PERSP_0"):
                m.d.comb += [
                    mul_a_interp.eq(self.ctx.vtx[0].position_ndc[2]),
                    mul_b_interp.eq(weight_linear[0]),
                ]
                m.d.sync += depth_sat.eq(mul_p_interp.saturate(_persp_div_shape))

                m.next = "GET_PRE_PERSP_1"

            with m.State("GET_PRE_PERSP_1"):
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
                    persp_mul_a.eq(persp_pre[0]),
                    persp_mul_b.eq(inv_w_sum_recip),
                ]
                m.d.sync += weight_persp[0].eq(persp_mul_p)

                m.next = "PERSPECTIVE_W1"

            with m.State("PERSPECTIVE_W1"):
                m.d.comb += [
                    persp_mul_a.eq(persp_pre[1]),
                    persp_mul_b.eq(inv_w_sum_recip),
                ]
                m.d.sync += weight_persp[1].eq(persp_mul_p)

                m.d.comb += [
                    mul_a_interp.eq(self.ctx.vtx[0].color[0].clamp(zero, one)),
                    mul_b_interp.eq(weight_persp[0]),
                ]
                m.d.sync += color_sat[0].eq(mul_p_interp.saturate(_persp_div_shape))

                m.next = "INTERP_COLOR_0_1"

            with m.State("INTERP_COLOR_0_1"):
                m.d.sync += weight_persp[2].eq(one - weight_persp[0] - weight_persp[1])

                m.d.comb += [
                    mul_a_interp.eq(self.ctx.vtx[1].color[0].clamp(zero, one)),
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
                    mul_a_interp.eq(self.ctx.vtx[2].color[0].clamp(zero, one)),
                    mul_b_interp.eq(weight_persp[2]),
                ]
                m.d.sync += color_sat[0].eq(
                    (color_sat[0] + mul_p_interp).saturate(_persp_div_shape)
                )
                m.next = "INTERP_COLOR_1"

            with m.State("INTERP_COLOR_1"):
                m.d.comb += [
                    mul_a_interp.eq(self.ctx.vtx[0].color[1].clamp(zero, one)),
                    mul_b_interp.eq(weight_persp[0]),
                ]
                m.d.sync += color_sat[1].eq(mul_p_interp.saturate(_persp_div_shape))
                m.next = "INTERP_COLOR_1_1"

            with m.State("INTERP_COLOR_1_1"):
                m.d.comb += [
                    mul_a_interp.eq(self.ctx.vtx[1].color[1].clamp(zero, one)),
                    mul_b_interp.eq(weight_persp[1]),
                ]
                m.d.sync += color_sat[1].eq(
                    (color_sat[1] + mul_p_interp).saturate(_persp_div_shape)
                )
                m.next = "INTERP_COLOR_1_2"

            with m.State("INTERP_COLOR_1_2"):
                m.d.comb += [
                    mul_a_interp.eq(self.ctx.vtx[2].color[1].clamp(zero, one)),
                    mul_b_interp.eq(weight_persp[2]),
                ]
                m.d.sync += color_sat[1].eq(
                    (color_sat[1] + mul_p_interp).saturate(_persp_div_shape)
                )
                m.next = "INTERP_COLOR_2"

            with m.State("INTERP_COLOR_2"):
                m.d.comb += [
                    mul_a_interp.eq(self.ctx.vtx[0].color[2].clamp(zero, one)),
                    mul_b_interp.eq(weight_persp[0]),
                ]
                m.d.sync += color_sat[2].eq(mul_p_interp.saturate(_persp_div_shape))
                m.next = "INTERP_COLOR_2_1"

            with m.State("INTERP_COLOR_2_1"):
                m.d.comb += [
                    mul_a_interp.eq(self.ctx.vtx[1].color[2].clamp(zero, one)),
                    mul_b_interp.eq(weight_persp[1]),
                ]
                m.d.sync += color_sat[2].eq(
                    (color_sat[2] + mul_p_interp).saturate(_persp_div_shape)
                )
                m.next = "INTERP_COLOR_2_2"

            with m.State("INTERP_COLOR_2_2"):
                m.d.comb += [
                    mul_a_interp.eq(self.ctx.vtx[2].color[2].clamp(zero, one)),
                    mul_b_interp.eq(weight_persp[2]),
                ]
                m.d.sync += color_sat[2].eq(
                    (color_sat[2] + mul_p_interp).saturate(_persp_div_shape)
                )
                m.next = "INTERP_COLOR_3"

            with m.State("INTERP_COLOR_3"):
                m.d.comb += [
                    mul_a_interp.eq(self.ctx.vtx[0].color[3].clamp(zero, one)),
                    mul_b_interp.eq(weight_persp[0]),
                ]
                m.d.sync += color_sat[3].eq(mul_p_interp.saturate(_persp_div_shape))
                m.next = "INTERP_COLOR_3_1"

            with m.State("INTERP_COLOR_3_1"):
                m.d.comb += [
                    mul_a_interp.eq(self.ctx.vtx[1].color[3].clamp(zero, one)),
                    mul_b_interp.eq(weight_persp[1]),
                ]
                m.d.sync += color_sat[3].eq(
                    (color_sat[3] + mul_p_interp).saturate(_persp_div_shape)
                )
                m.next = "INTERP_COLOR_3_2"

            with m.State("INTERP_COLOR_3_2"):
                m.d.comb += [
                    mul_a_interp.eq(self.ctx.vtx[2].color[3].clamp(zero, one)),
                    mul_b_interp.eq(weight_persp[2]),
                ]
                m.d.sync += color_sat[3].eq(
                    (color_sat[3] + mul_p_interp).saturate(_persp_div_shape)
                )
                m.next = "OUTPUT"

            with m.State("OUTPUT"):
                m.d.comb += [
                    self.o.p.coord_pos[0].eq(px_lat),
                    self.o.p.coord_pos[1].eq(py_lat),
                    self.o.p.depth.eq(depth_sat.clamp(zero, one)),
                    self.o.p.color[0].eq(color_sat[0].clamp(zero, one)),
                    self.o.p.color[1].eq(color_sat[1].clamp(zero, one)),
                    self.o.p.color[2].eq(color_sat[2].clamp(zero, one)),
                    self.o.p.color[3].eq(color_sat[3].clamp(zero, one)),
                    self.o.p.front_facing.eq(self.ctx.front_facing),
                ]

                m.d.comb += self.o.valid.eq(1)
                with m.If(self.o.ready):
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

    i: In(stream.Signature(TriangleContext))
    o: Out(stream.Signature(FragmentLayout))

    # Framebuffer configuration
    fb_info: In(FramebufferInfoLayout)
    ready: Out(1)

    def __init__(self, inv_steps: int = 3, num_generators: int = 1):
        super().__init__()
        self._inv_steps = inv_steps
        self._num_generators = num_generators
        self._subpixel_bits = FixedPoint_fb.f_bits

    def elaborate(self, platform):
        m = Module()

        ctx_buf = Signal(TriangleContext)
        inflight = Signal(
            range(
                FragmentGenerator.max_pipelined_elements() * self._num_generators * 2
                + 1
            )
        )

        px = Signal(unsigned(FixedPoint_fb.i_bits))
        py = Signal(unsigned(FixedPoint_fb.i_bits))

        d_x = Signal(data.ArrayLayout(FixedPoint_fb, 3))
        d_y = Signal(data.ArrayLayout(FixedPoint_fb, 3))

        winding_ccw = Signal()
        is_top = Signal(3)
        is_left = Signal(3)

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

        inflight_reset = Signal()
        inflight_inc = Signal()

        with m.FSM():
            with m.State("IDLE"):
                m.d.comb += self.ready.eq(1)
                m.d.comb += self.i.ready.eq(1)
                with m.If(self.i.valid):
                    m.d.comb += inflight_reset.eq(1)
                    m.d.sync += [
                        ctx_buf.eq(self.i.payload),
                        px.eq(self.i.payload.min_x),
                        py.eq(self.i.payload.min_y),
                        winding_ccw.eq(self.i.payload.area > 0),
                    ]
                    m.next = "PREP_EDGES"

            with m.State("PREP_EDGES"):
                m.d.sync += [
                    d_x[0].eq(ctx_buf.screen_x[2] - ctx_buf.screen_x[1]),
                    d_y[0].eq(ctx_buf.screen_y[2] - ctx_buf.screen_y[1]),
                    d_x[1].eq(ctx_buf.screen_x[0] - ctx_buf.screen_x[2]),
                    d_y[1].eq(ctx_buf.screen_y[0] - ctx_buf.screen_y[2]),
                    d_x[2].eq(ctx_buf.screen_x[1] - ctx_buf.screen_x[0]),
                    d_y[2].eq(ctx_buf.screen_y[1] - ctx_buf.screen_y[0]),
                ]
                m.next = "CATEGORIZE_EDGES"

            with m.State("CATEGORIZE_EDGES"):
                for i in range(3):
                    m.d.sync += [
                        is_top[i].eq(
                            (d_y[i] == 0)
                            & Mux(
                                winding_ccw,
                                d_x[i] > 0,
                                d_x[i] < 0,
                            )
                        ),
                        is_left[i].eq(
                            Mux(
                                winding_ccw,
                                d_y[i] < 0,
                                d_y[i] > 0,
                            )
                        ),
                    ]
                m.next = "RASTERIZE"

            with m.State("RASTERIZE"):
                m.d.comb += [
                    distrib.i.valid.eq(1),
                    distrib.i.p.px.eq(px),
                    distrib.i.p.py.eq(py),
                ]
                with m.If(distrib.i.ready):
                    m.d.comb += inflight_inc.eq(1)
                    with m.If(~task_last_x):
                        m.d.sync += px.eq(px + 1)
                    with m.Elif(~task_last_y):
                        m.d.sync += [px.eq(ctx_buf.min_x), py.eq(py + 1)]
                    with m.Else():
                        m.next = "WAIT_DONE"
            with m.State("WAIT_DONE"):
                with m.If(inflight == 0):
                    m.next = "IDLE"

        fragments = []
        done_vec = Signal(self._num_generators)
        for idx in range(self._num_generators):
            m.submodules[f"fg_{idx}"] = fg = FragmentGenerator(
                inv_steps=self._inv_steps
            )
            fragments.append(fg)

            wiring.connect(m, distrib.o[idx], fg.i)
            m.d.comb += fg.ctx.eq(ctx_buf)
            m.d.comb += fg.d_x.eq(d_x)
            m.d.comb += fg.d_y.eq(d_y)
            m.d.comb += fg.winding_ccw.eq(winding_ccw)
            m.d.comb += fg.is_top_left.eq(is_top | is_left)

            wiring.connect(m, fg.o, recomb.i[idx])
            m.d.comb += done_vec[idx].eq(fg.o_done)

        with m.If(inflight_reset):
            m.d.sync += inflight.eq(0)
        with m.Else():
            m.d.sync += inflight.eq(inflight + inflight_inc - popcount(done_vec))

        wiring.connect(m, recomb.o, wiring.flipped(self.o))

        return m
