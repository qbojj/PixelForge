from amaranth import *
from amaranth.lib import data, stream, wiring
from amaranth.lib.wiring import In, Out

from ..utils.layouts import ShadingVertexLayout, VertexLayout, num_textures
from ..utils.types import FixedPoint, FixedPoint_mem


class VertexTransform(wiring.Component):
    """Vertex transformation core

    Transforms incoming vertices from model space to clip space.
    Also outputs transformed vertices for shading stage.

    Input: VertexLayout
    Output: ShadingVertexLayout

    Uses following wires for column-major transformation matrices:
    - position_mv: Model-View matrix (4x4)
    - position_p: Projection matrix (4x4)
    - normal_mv_inv_t: Inverse transpose of Model-View matrix (3x3)
    - texture_transforms: array of texture transformation matrices (4x4) for each texture

    TODO: support for configurable amount of multiplyer circuits (for now 4)
    """

    i: stream.Interface
    o: stream.Interface

    position_mv: Signal
    position_p: Signal
    normal_mv_inv_t: Signal
    texture_transform: Signal

    ready: Signal

    def __init__(self):
        super().__init__(
            {
                "i": In(stream.Signature(VertexLayout)),
                "o": Out(stream.Signature(ShadingVertexLayout)),
                "position_mv": In(data.ArrayLayout(FixedPoint_mem, 16)),
                "position_p": In(data.ArrayLayout(FixedPoint_mem, 16)),
                "normal_mv_inv_t": In(data.ArrayLayout(FixedPoint_mem, 9)),
                "texture_transform": In(data.ArrayLayout(FixedPoint_mem, 16)),
                "ready": Out(1),
            }
        )

    def elaborate(self, platform) -> Module:
        # TODO: use actual memory for matrices

        m = Module()

        i_data = Signal.like(self.i.payload)
        o_data = Signal.like(self.o.payload)

        mul_a = Signal(FixedPoint)
        mul_b = Signal(FixedPoint)
        mul_result = Signal(FixedPoint)
        m.d.comb += mul_result.eq(mul_a * mul_b)
        cum_result = Signal(FixedPoint)

        attr_info = [
            {
                "name": "POSITION_MV",
                "result": o_data.position_view,
                "matrix": self.position_mv,
                "vector": i_data.position,
                "dim": 4,
            },
            {
                "name": "POSITION_P",
                "result": o_data.position_proj,
                "matrix": self.position_p,
                "vector": o_data.position_view,
                "dim": 4,
            },
            {
                "name": "NORMAL",
                "result": o_data.normal_view,
                "matrix": self.normal_mv_inv_t,
                "vector": i_data.normal,
                "dim": 3,
            },
        ] + [
            {
                "name": f"TEXTURE_{i}",
                "result": o_data.texcoords[i],
                "matrix": self.texture_transform,
                "vector": i_data.texcoords[i],
                "dim": 4,
            }
            for i in range(num_textures)
        ]

        with m.FSM():
            with m.State("IDLE"):
                m.d.comb += self.ready.eq(1)
                with m.If(self.i.valid):
                    m.d.comb += self.i.ready.eq(1)
                    m.d.sync += [
                        i_data.eq(self.i.payload),
                        o_data.color.eq(self.i.p.color),
                    ]
                    m.d.sync += Print("VertexTransform vtx in: ", self.i.payload)
                    m.next = f"{attr_info[0]['name']}_INIT"

            for i, attr in enumerate(attr_info):
                base = f"{attr['name']}"
                next_state = (
                    f"{attr_info[i + 1]['name']}_INIT"
                    if i + 1 < len(attr_info)
                    else "SEND"
                )

                with m.State(f"{base}_INIT"):
                    m.d.sync += cum_result.eq(0)
                    m.next = f"{base}_0_0"

                for i in range(attr["dim"]):
                    for j in range(attr["dim"]):
                        with m.State(f"{base}_{i}_{j}"):
                            m.d.sync += cum_result.eq(
                                cum_result + mul_result,
                            )
                            m.d.comb += [
                                mul_a.eq(attr["vector"][j]),
                                mul_b.eq(attr["matrix"][i * attr["dim"] + j]),
                            ]

                            m.next = (
                                f"{base}_{i}_{j + 1}"
                                if j < attr["dim"] - 1
                                else (f"{base}_STORE_{i}")
                            )
                    with m.State(f"{base}_STORE_{i}"):
                        m.d.sync += attr["result"][i].eq(cum_result)
                        m.d.sync += cum_result.eq(0)
                        m.next = (
                            next_state if i == attr["dim"] - 1 else f"{base}_{i + 1}_0"
                        )

            with m.State("SEND"):
                m.d.comb += [
                    self.o.payload.eq(o_data),
                    self.o.valid.eq(1),
                ]
                with m.If(self.o.ready):
                    m.next = "IDLE"

        return m
