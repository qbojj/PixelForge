from amaranth import *
from amaranth.lib import data, stream, wiring
from amaranth.lib.wiring import In, Out

from ..utils.layouts import PrimitiveAssemblyLayout, RasterizerLayout
from ..utils.types import CullFace, FixedPoint, FrontFace, PrimitiveType


class PrimitiveAssemblyConfigLayout(data.Struct):
    """Primitive assembly configuration"""

    type: PrimitiveType
    cull: CullFace
    winding: FrontFace


class PrimitiveAssembly(wiring.Component):
    """Primitive assembly core

    Assembles incoming shaded vertices into primitives for rasterization stage.

    Input: PrimitiveAssemblyLayout
    Output: RasterizerLayout
    """

    is_vertex: In(stream.Signature(PrimitiveAssemblyLayout))
    os_primitive: Out(stream.Signature(RasterizerLayout))
    ready: Out(1)

    prim_config: In(PrimitiveAssemblyConfigLayout)

    def __init__(self):
        super().__init__()

    def elaborate(self, platform):
        m = Module()

        with m.Switch(self.prim_config.type):
            with m.Case(PrimitiveType.POINTS, PrimitiveType.LINES):
                # Simple pass-through for points and lines
                m.d.comb += self.ready.eq(1)
                m.d.comb += [
                    self.is_vertex.ready.eq(self.os_primitive.ready),
                    self.os_primitive.valid.eq(self.is_vertex.valid),
                    self.os_primitive.p.position_ndc.eq(self.is_vertex.p.position_ndc),
                    self.os_primitive.p.texcoords.eq(self.is_vertex.p.texcoords),
                    self.os_primitive.p.color.eq(self.is_vertex.p.color),
                    self.os_primitive.p.front_facing.eq(1),
                ]
            with m.Case(PrimitiveType.TRIANGLES):
                # calculate front facing
                trinagle = Array(Signal.like(self.is_vertex.payload) for _ in range(3))
                idx = Signal(range(3))
                front_facing = Signal()

                def send_vertex(m, ff, idx):
                    m.d.comb += [
                        self.os_primitive.valid.eq(1),
                        self.os_primitive.p.position_ndc.eq(trinagle[idx].position_ndc),
                        self.os_primitive.p.texcoords.eq(trinagle[idx].texcoords),
                        self.os_primitive.p.color.eq(
                            Mux(ff, trinagle[idx].color, trinagle[idx].color_back)
                        ),
                        self.os_primitive.p.front_facing.eq(ff),
                    ]

                with m.FSM():
                    with m.State("WAIT_VERTEX"):
                        m.d.comb += [self.is_vertex.ready.eq(1), self.ready.eq(1)]
                        with m.If(self.is_vertex.valid):
                            m.d.sync += [
                                trinagle[idx].eq(self.is_vertex.payload),
                                idx.eq(idx + 1),
                            ]
                            with m.If(idx + 1 == 3):
                                m.d.sync += idx.eq(0)
                                m.next = "SEND_PRIMITIVE"
                    with m.State("SEND_PRIMITIVE"):
                        # Compute vectors with widened intermediates to preserve sign
                        v0 = Array(Signal(FixedPoint) for _ in range(2))
                        v1 = Array(Signal(FixedPoint) for _ in range(2))
                        area = Signal(FixedPoint)

                        """
                        a = -1/2 * SUM_{i=0}^{n-1}{x_i*y_{i+1 mod n} - x_{i+1 mod n}*y_i}
                        front facing iif:
                        CCW: a > 0
                        CW: a < 0
                        3 vertices:
                        (x0,y0), (x1,y1), (x2,y2)
                        a = -1/2 * (x0*y1 - x1*y0 + x1*y2 - x2*y1 + x2*y0 - x0*y2)
                        a = -1/2 * ( (x1 - x0)*(y2 - y1) - (x2 - x1)*(y1 - y0) )
                        """

                        vtx0 = trinagle[0].position_ndc
                        vtx1 = trinagle[1].position_ndc
                        vtx2 = trinagle[2].position_ndc

                        m.d.comb += [
                            v0[0].eq(vtx1[0] - vtx0[0]),
                            v0[1].eq(vtx1[1] - vtx0[1]),
                            v1[0].eq(vtx2[0] - vtx1[0]),
                            v1[1].eq(vtx2[1] - vtx1[1]),
                            # Twice the signed area (negative for CW when front face is CCW)
                            area.eq((v0[0] * v1[1]) - (v1[0] * v0[1])),
                        ]

                        ff = Signal()
                        with m.Switch(self.prim_config.winding):
                            with m.Case(FrontFace.CCW):
                                m.d.comb += ff.eq(area > 0)
                            with m.Case(FrontFace.CW):
                                m.d.comb += ff.eq(area < 0)

                        m.d.sync += Print(Format("vtx0: {}", vtx0))
                        m.d.sync += Print(Format("vtx1: {}", vtx1))
                        m.d.sync += Print(Format("vtx2: {}", vtx2))

                        m.d.sync += Print(Format("v0: {} {}", v0[0], v0[1]))
                        m.d.sync += Print(Format("v1: {} {}", v1[0], v1[1]))

                        m.d.sync += Print(
                            Format(
                                "Area: {}, Front Facing: {}, prim_wind: {}, cull: {}",
                                area,
                                ff,
                                self.prim_config.winding,
                                self.prim_config.cull,
                            )
                        )

                        cull_v = self.prim_config.cull.as_value()

                        with m.If(ff & ((cull_v & CullFace.FRONT.value) != 0)):
                            m.d.sync += Print("Culling front face")
                            m.next = "WAIT_VERTEX"  # cull primitive
                        with m.Elif(~ff & ((cull_v & CullFace.BACK.value) != 0)):
                            m.d.sync += Print("Culling back face")
                            m.next = "WAIT_VERTEX"  # cull primitive
                        with m.Else():
                            # Register front_facing for use in subsequent states
                            m.d.sync += front_facing.eq(ff)
                            send_vertex(m, ff, 0)
                            with m.If(self.os_primitive.ready):
                                m.next = "SEND_VERTEX_1"
                    with m.State("SEND_VERTEX_1"):
                        send_vertex(m, front_facing, 1)
                        with m.If(self.os_primitive.ready):
                            m.next = "SEND_VERTEX_2"
                    with m.State("SEND_VERTEX_2"):
                        send_vertex(m, front_facing, 2)
                        with m.If(self.os_primitive.ready):
                            m.next = "WAIT_VERTEX"

        return m
