from amaranth import *
from amaranth.lib import stream, wiring
from amaranth.lib.wiring import In, Out
from amaranth_soc import csr

from ..utils.layouts import PrimitiveAssemblyLayout, RasterizerLayout
from ..utils.types import CullFace, FixedPoint, FrontFace, PrimitiveType


class PrimitiveAssembly(wiring.Component):
    """Primitive assembly core

    Assembles incoming shaded vertices into primitives for rasterization stage.

    Input: PrimitiveAssemblyLayout
    Output: RasterizerLayout
    """

    is_vertex: In(stream.Signature(PrimitiveAssemblyLayout))
    os_primitive: Out(stream.Signature(RasterizerLayout))

    class PrimitiveReg(csr.Register, access="rw"):
        type: csr.Field(csr.action.RW, PrimitiveType)

    class PrimitiveCullReg(csr.Register, access="rw"):
        cull: csr.Field(csr.action.RW, CullFace)

    class PrimitiveWindingReg(csr.Register, access="rw"):
        winding: csr.Field(csr.action.RW, FrontFace)

    def __init__(self):
        super().__init__()
        regs = csr.Builder(addr_width=8, data_width=8)
        self.prim_type = regs.add(
            name="prim_type", reg=self.PrimitiveReg(), offset=0x00
        )
        self.prim_cull = regs.add(
            name="prim_cull", reg=self.PrimitiveCullReg(), offset=0x04
        )
        self.prim_winding = regs.add(
            name="prim_winding", reg=self.PrimitiveWindingReg(), offset=0x08
        )
        self.csr_bridge = csr.Bridge(regs.as_memory_map())
        self.csr_bus = self.csr_bridge.bus

    def elaborate(self, platform):
        m = Module()

        m.d.comb += self.ready.eq(~self.os_primitive.valid)

        with m.If(self.os_primitive.ready):
            m.d.sync += self.os_primitive.valid.eq(0)

        with m.Switch(self.prim_type.f.type.data):
            with m.Case(PrimitiveType.POINTS, PrimitiveType.LINES):
                with m.If(self.is_vertex.valid):
                    m.d.comb += self.is_vertex.ready.eq(1)

                    m.d.sync += [
                        self.os_primitive.valid.eq(1),
                        self.os_primitive.p.position_proj.eq(
                            self.is_vertex.p.position_proj
                        ),
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
                    m.d.sync += [
                        self.os_primitive.valid.eq(1),
                        self.os_primitive.p.position_proj.eq(
                            trinagle[idx].p.position_proj
                        ),
                        self.os_primitive.p.texcoords.eq(trinagle[idx].p.texcoords),
                        self.os_primitive.p.color.eq(
                            Mux(ff, trinagle[idx].p.color, trinagle[idx].p.color_back)
                        ),
                        self.os_primitive.p.front_facing.eq(front_facing),
                    ]

                with m.FSM():
                    with m.State("WAIT_VERTEX"):
                        with m.If(self.is_vertex.valid):
                            m.d.comb += self.is_vertex.ready.eq(1)

                            m.d.sync += [
                                trinagle[idx].eq(self.is_vertex.payload),
                                idx.eq(idx + 1),
                            ]
                            with m.If(idx + 1 == 3):
                                m.d.sync += idx.eq(0)
                                m.next = "SEND_PRIMITIVE"
                    with m.State("SEND_PRIMITIVE"):
                        # Compute vectors
                        v0 = Array(Signal(FixedPoint) for _ in range(2))
                        v1 = Array(Signal(FixedPoint) for _ in range(2))
                        a = Signal(FixedPoint)

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

                        vtx0 = trinagle[0].p.position_proj
                        vtx1 = trinagle[1].p.position_proj
                        vtx2 = trinagle[2].p.position_proj

                        m.d.comb += [
                            v0[0].eq(vtx1.x - vtx0.x),
                            v0[1].eq(vtx1.y - vtx0.y),
                            v1[0].eq(vtx2.x - vtx1.x),
                            v1[1].eq(vtx2.y - vtx1.y),
                            a.eq(-((v0[0] * v1[1]) - (v1[0] * v0[1])) >> 1),
                        ]

                        ff = Signal()
                        with m.Switch(self.prim_winding.f.winding.data):
                            with m.Case(FrontFace.CCW):
                                m.d.comb += ff.eq(a > 0)
                            with m.Case(FrontFace.CW):
                                m.d.comb += ff.eq(a < 0)

                        with m.If(ff & (self.prim_cull.f.cull.data & CullFace.FRONT)):
                            m.next = "WAIT_VERTEX"  # cull primitive
                        with m.Elif(~ff & (self.prim_cull.f.cull.data & CullFace.BACK)):
                            m.next = "WAIT_VERTEX"  # cull primitive
                        with m.Elif(self.os_primitive.ready | ~self.os_primitive.valid):
                            send_vertex(m, ff, 0)
                            m.next = "SEND_VERTEX_1"
                    with m.State("SEND_VERTEX_1"):
                        with m.If(self.os_primitive.ready | ~self.os_primitive.valid):
                            send_vertex(m, ff, 1)
                            m.next = "SEND_VERTEX_2"
                    with m.State("SEND_VERTEX_2"):
                        with m.If(self.os_primitive.ready | ~self.os_primitive.valid):
                            send_vertex(m, ff, 2)
                            m.next = "WAIT_VERTEX"

        return m
