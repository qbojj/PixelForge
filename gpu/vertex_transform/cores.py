from amaranth import *
from amaranth.lib import data, stream, wiring
from amaranth.lib.wiring import Out
from amaranth_soc import csr

from gpu.utils.math import FixedPointInv, SimpleOpModule
from gpu.utils.stream import StreamToVector, VectorToStream

from ..utils.layouts import (
    ShadingVertexLayout,
    VertexLayout,
    num_textures,
)
from ..utils.types import FixedPoint, FixedPoint_mem, Vector3


class VertexTransform(wiring.Component):
    """Vertex transformation core

    Transforms incoming vertices from model space to clip space.
    Also outputs transformed vertices for shading stage.

    Input: VertexLayout
    Output: ShadingVertexLayout

    Uses following registers for collumn-major transformation matrices:
    - position_mv: Model-View matrix (4x4)
    - position_p: Projection matrix (4x4)
    - normal_mv_inv_t: Inverse transpose of Model-View matrix (3x3)
    - texture_transforms: array of texture transformation matrices (4x4) for each texture

    TODO: support for configurable amount of multiplyer circuits (for now 4)
    """

    is_vertex: stream.Signature(VertexLayout)
    os_vertex: stream.Signature(ShadingVertexLayout)

    ready: Out(1)

    class MatrixReg(csr.Register, access="rw"):
        def __init__(self, size=16):
            super().__init__(
                csr.Field(csr.action.RW, data.ArrayLayout(FixedPoint_mem, size))
            )

    class EnablementReg(csr.Register, access="rw"):
        def __init__(self):
            super().__init__(
                {
                    "normal": csr.Field(csr.action.RW, 1),
                }
                + {
                    f"texture_{i}": csr.Field(csr.action.RW, 1)
                    for i in range(num_textures)
                }
            )

    def __init__(self):
        super().__init__()
        regs = csr.Builder(addr_width=10, data_width=8)

        self.enabled = regs.add("enable", self.EnablementReg())

        with regs.Cluster("position"):
            self.position_mv = regs.add("MV", self.MatrixReg(16))
            self.position_p = regs.add("P", self.MatrixReg(16))

        with regs.Cluster("normal"):
            self.normal_mv_inv_t = regs.add("MV_inv_t", self.MatrixReg(9))
            self.texture_transforms = []

        with regs.Cluster("textures"):
            for i in range(num_textures):
                with regs.Cluster(str(i)):  # TODO: Change to regs.Index(i):
                    self.texture_transforms.append(
                        regs.add("texture_transform", self.MatrixReg(16))
                    )

        self.csr_bridge = csr.Bridge(regs.as_memory_map())
        self.csr_bus = self.csr_bridge.bus

    def elaborate(self, platform) -> Module:
        m = Module()

        m.submodules += [self.csr_bridge]

        i_data = Signal.like(self.is_vertex.payload)
        o_data = Signal.like(self.os_vertex.payload)

        mul_a = Signal(FixedPoint)
        mul_b = Signal(FixedPoint)
        mul_result = Signal(FixedPoint)
        m.d.comb += mul_result.eq(mul_a * mul_b)
        cum_result = Signal(FixedPoint)

        attr_info = [
            {
                "name": "POSITION_MV",
                "result": o_data.position_view,
                "matrix": self.position_mv.f.data,
                "vector": i_data.position,
                "enabled": C(1),
                "dim": 4,
            },
            {
                "name": "POSITION_P",
                "result": o_data.position_proj,
                "matrix": self.position_p.f.data,
                "vector": o_data.position_view,
                "enabled": C(1),
                "dim": 4,
            },
            {
                "name": "NORMAL",
                "result": o_data.normal_view,
                "matrix": self.normal_mv_inv_t.f.data,
                "vector": i_data.normal,
                "enabled": self.enabled.f.data.normal,
                "dim": 3,
            },
        ] + [
            {
                "name": f"TEXTURE_{i}",
                "result": o_data.texcoords[i],
                "matrix": self.texture_transforms[i].f.data,
                "vector": i_data.texcoords[i],
                "enabled": self.enabled.f.data[f"texture_{i}"],
                "dim": 4,
            }
            for i in range(num_textures)
        ]

        with m.If(self.os_vertex.ready):
            m.d.sync += self.os_vertex.valid.eq(0)

        with m.FSM():
            with m.State("IDLE"):
                m.d.comb += self.ready.eq(~self.os_vertex.valid)
                with m.If(self.is_vertex.valid & self.os_vertex.ready):
                    m.d.sync += [
                        i_data.eq(self.is_vertex.payload),
                    ]
                    m.next = f"{attr_info[0]['name']}_INIT"

            for i, attr in enumerate(attr_info):
                base = f"{attr['name']}"
                next_state = (
                    f"{attr_info[i+1]['name']}_INIT"
                    if i + 1 < len(attr_info)
                    else "SEND"
                )

                with m.State(f"{base}_INIT"):
                    m.d.sync += cum_result.eq(0)
                    with m.If(attr["enabled"]):
                        m.next = f"{base}_0_0"
                    with m.Else():
                        # skip transformation - return 0,0,0,1 vector
                        for j in range(len(attr["result"])):
                            m.d.sync += attr["result"][j].eq(0.0 if j < 3 else 1.0)
                        m.next = next_state

                for i in range(attr["dim"]):
                    for j in range(attr["dim"]):
                        with m.State(f"{base}_{i}_{j}"):
                            m.d.sync += cum_result.eq(
                                cum_result + mul_result,
                            )
                            m.d.comb += [
                                mul_a.eq(attr["vector"][j]),
                                mul_b.eq_reinterpret(
                                    attr["matrix"][i * attr["dim"] + j]
                                ),
                            ]

                            m.next = (
                                f"{base}_{i}_{j+1}"
                                if j < attr["dim"] - 1
                                else (f"{base}_STORE_{i}")
                            )
                    with m.State(f"{base}_STORE_{i}"):
                        m.d.sync += attr["result"][i].eq(cum_result)
                        m.d.sync += cum_result.eq(0)
                        m.next = (
                            next_state if i == attr["dim"] - 1 else f"{base}_{i+1}_0"
                        )

            with m.State("SEND"):
                with m.If(~self.os_vertex.valid | self.os_vertex.ready):
                    m.d.sync += [
                        self.os_vertex.payload.eq(o_data),
                        self.os_vertex.valid.eq(1),
                    ]
                    m.next = "IDLE"

        return m


class PerspectiveDivide(wiring.Component):
    """
    Performs (x', y', z', w') = (x/w, y/w, z/w, 1/w) for position

    Result will be in NDC space.
    """

    i: In(stream.Signature(ShadingVertexLayout))
    o: Out(stream.Signature(ShadingVertexLayout))

    def enaborate(self, platorm) -> Module:
        m = Module()

        m.submodule.v2s_a = v2s_a = VectorToStream(Vector3)
        m.submodule.dup = dup = wiring.DuplicateStream(FixedPoint, 3)
        m.submodule.mul = mul = SimpleOpModule(lambda a, b: a * b, FixedPoint)
        m.submodule.s2v = s2v = StreamToVector(Vector3)
        m.submodule.inverse = inverse = FixedPointInv(FixedPoint)

        wiring.connect(m, v2s_a.o, mul.a)
        wiring.connect(m, mul.o, s2v.i)

        with m.If(self.o.ready):
            m.d.sync += self.o.valid.eq(0)

        with m.If(inverse.i.ready):
            m.d.sync += inverse.i.valid.eq(0)

        with m.If(v2s_a.i.ready):
            m.d.sync += v2s_a.i.valid.eq(0)

        with m.FSM():
            with m.State("IDLE"):
                with m.If(self.i.valid):
                    m.d.sync += [
                        v2s_a.i.payload.eq(Cat(self.i.payload.position_proj[:3])),
                        v2s_a.i.valid.eq(1),
                        inverse.i.payload.eq(self.i.payload.position_proj[3]),
                        inverse.i.valid.eq(1),
                    ]
                    m.next = "CALC_INV_W"
            with m.State("CALC_INV_W"):
                with m.If(inverse.o.valid):
                    m.d.sync += [
                        dup.i.p.eq(inverse.o.p),
                        dup.i.valid.eq(1),
                    ]
                    m.next = "DIVIDE"
            with m.State("DIVIDE"):
                # calculate x*1/w, y*1/w, z*1/w, 1/w
                with m.If(s2v.o.valid):
                    m.d.comb += [
                        dup.o.ready.eq(1),
                        s2v.o.ready.eq(1),
                        inverse.i.ready.eq(1),
                        self.i.ready.eq(1),
                    ]

                    m.d.sync += [
                        self.o.p.position_view.eq(self.i.payload.position_view),
                        self.o.p.position_proj[:3].eq(s2v.o.p),
                        self.o.p.position_proj[3].eq(inverse.o.p),
                        self.o.p.normal_view.eq(self.i.payload.normal_view),
                        self.o.p.texcoords.eq(self.i.payload.texcoords),
                        self.o.p.color.eq(self.i.payload.color),
                        self.o.valid.eq(1),
                    ]

                    m.next = "IDLE"

        return m
