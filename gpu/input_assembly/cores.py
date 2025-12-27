from dataclasses import dataclass

import amaranth_soc.wishbone.bus as wb
from amaranth import *
from amaranth.lib import data, stream, wiring
from amaranth.lib.wiring import In, Out
from amaranth.utils import exact_log2
from transactron import *
from transactron.lib import *

from ..utils.layouts import (
    VertexLayout,
    num_textures,
    wb_bus_addr_width,
    wb_bus_data_width,
)
from ..utils.types import (
    FixedPoint_mem,
    IndexKind,
    InputTopology,
    address_shape,
    index_shape,
)
from .layouts import InputData, InputMode

__all__ = [
    "IndexGenerator",
    "InputTopologyProcessor",
    "InputAssembly",
]


class IndexGenerator(wiring.Component):
    """Generates index stream based on index stream description register.

    Gets index stream description and outputs index stream.

    TODO: add memory burst support
    """

    os_index: Out(stream.Signature(index_shape))
    bus: Out(
        wb.Signature(
            addr_width=wb_bus_addr_width, data_width=wb_bus_data_width, granularity=8
        )
    )
    ready: Out(1)

    c_address: In(address_shape)
    c_count: In(unsigned(32))
    c_kind: In(IndexKind)
    start: In(1)

    def elaborate(self, platform) -> Module:
        m = Module()

        address = Signal.like(self.c_address)
        kind = self.c_kind
        count = self.c_count

        index_increment = Signal(3)
        index_shift = Signal(2)
        with m.Switch(kind):
            with m.Case(IndexKind.U8):
                m.d.comb += [
                    index_increment.eq(1),
                    index_shift.eq(0),
                ]
            with m.Case(IndexKind.U16):
                m.d.comb += [
                    index_increment.eq(2),
                    index_shift.eq(1),
                ]
            with m.Case(IndexKind.U32):
                m.d.comb += [
                    index_increment.eq(4),
                    index_shift.eq(2),
                ]
            with m.Default():
                m.d.comb += [
                    index_increment.eq(0),
                    index_shift.eq(0),
                ]

        data_read = Signal.like(self.bus.dat_r)

        offset = address[: exact_log2(self.bus.data_width // 8)]
        extended_data = Signal(index_shape)

        with m.Switch(kind):
            with m.Case(IndexKind.U8):
                m.d.comb += extended_data.eq(data_read.word_select(offset[0:], 8))
            with m.Case(IndexKind.U16):
                m.d.comb += extended_data.eq(data_read.word_select(offset[1:], 16))
            with m.Case(IndexKind.U32):
                m.d.comb += extended_data.eq(data_read.word_select(offset[2:], 32))

        cur_idx = Signal.like(count)

        with m.If(self.os_index.ready):
            m.d.sync += self.os_index.valid.eq(0)

        with m.FSM():
            with m.State("IDLE"):
                m.d.comb += self.ready.eq(1)
                with m.If(self.start):
                    m.d.sync += [
                        cur_idx.eq(0),
                        address.eq(self.c_address),
                    ]
                    with m.If(count == 0):
                        m.next = "IDLE"
                    with m.Elif(kind == IndexKind.NOT_INDEXED):
                        m.next = "STREAM_NON_INDEXED"
                    with m.Else():
                        m.next = "MEM_READ_INIT"

            with m.State("STREAM_NON_INDEXED"):
                with m.If(~self.os_index.valid | self.os_index.ready):
                    m.d.sync += [
                        self.os_index.payload.eq(cur_idx),
                        self.os_index.valid.eq(1),
                        cur_idx.eq(cur_idx + 1),
                    ]
                    with m.If(cur_idx + 1 == count):  # last index streamed
                        m.next = "WAIT_FLUSH"

            with m.State("MEM_READ_INIT"):
                # initiate memory read
                m.d.sync += [
                    self.bus.cyc.eq(1),
                    self.bus.adr.eq(address[len(offset) :]),
                    self.bus.we.eq(0),
                    self.bus.stb.eq(1),
                    self.bus.sel.eq(~0),
                ]
                m.next = "MEM_READ_WAIT"
            with m.State("MEM_READ_WAIT"):
                with m.If(self.bus.ack):
                    m.d.sync += data_read.eq(self.bus.dat_r)
                    m.next = "INDEX_SEND"
            with m.State("INDEX_SEND"):
                with m.If(~self.os_index.valid | self.os_index.ready):
                    next_addr = address + index_increment
                    m.d.sync += [
                        self.os_index.payload.eq(extended_data),
                        self.os_index.valid.eq(1),
                        address.eq(next_addr),
                        cur_idx.eq(cur_idx + 1),
                    ]
                    with m.If(cur_idx + 1 == count):
                        m.next = "WAIT_FLUSH"
                    with m.Elif(next_addr[: len(offset)] != 0):
                        m.next = "INDEX_SEND"
                    with m.Else():
                        m.next = (
                            "MEM_READ_INIT"  # crossed word boundary -> read next word
                        )
            with m.State("WAIT_FLUSH"):
                with m.If(~self.os_index.valid | self.os_index.ready):
                    m.next = "IDLE"

        return m


class InputTopologyProcessor(wiring.Component):
    """Processes input topology description.

    Gets input index stream and outputs vertex index stream based on input topology.
    """

    is_index: In(stream.Signature(index_shape))
    os_index: Out(stream.Signature(index_shape))
    ready: Out(1)

    c_input_topology: In(InputTopology)
    c_primitive_restart_enable: In(unsigned(1))
    c_primitive_restart_index: In(unsigned(32))
    c_base_vertex: In(unsigned(32))

    def __init__(self):
        super().__init__()

    def elaborate(self, platform) -> Module:
        m = Module()

        # values for triangle strips/fans and line strips
        v1 = Signal(index_shape)
        v2 = Signal(index_shape)
        vertex_count = Signal(2)

        # max 3 output indices per input index
        max_amplification = 3
        to_send = Array(Signal(index_shape) for _ in range(max_amplification))
        to_send_left = Signal(2)

        ready_for_input = Signal()

        m.d.comb += self.ready.eq(ready_for_input)
        m.d.comb += ready_for_input.eq(to_send_left == 0)

        with m.Switch(to_send_left):
            for i in range(1, max_amplification + 1):
                with m.Case(i):
                    m.d.comb += self.os_index.payload.eq(
                        to_send[i - 1] + self.c_base_vertex
                    )
                    m.d.comb += self.os_index.valid.eq(1)
                    with m.If(self.os_index.ready):
                        m.d.sync += to_send_left.eq(i - 1)

        m.d.comb += self.is_index.ready.eq(ready_for_input)

        with m.If(self.is_index.valid & ready_for_input):
            idx = self.is_index.payload

            with m.If(
                self.c_primitive_restart_enable
                & (idx == self.c_primitive_restart_index)
            ):
                m.d.sync += vertex_count.eq(0)  # reset on primitive restart
            with m.Else():
                with m.Switch(self.c_input_topology):
                    with m.Case(InputTopology.POINT_LIST):
                        m.d.sync += [
                            to_send[0].eq(idx),
                            to_send_left.eq(1),
                        ]
                    with m.Case(InputTopology.LINE_LIST):
                        with m.Switch(vertex_count):
                            with m.Case(0):
                                m.d.sync += [
                                    v1.eq(idx),
                                    vertex_count.eq(1),
                                ]
                            with m.Case(1):
                                m.d.sync += [
                                    to_send[1].eq(v1),
                                    to_send[0].eq(idx),
                                    to_send_left.eq(2),
                                    vertex_count.eq(0),
                                ]
                    with m.Case(InputTopology.TRIANGLE_LIST):
                        with m.Switch(vertex_count):
                            with m.Case(0):
                                m.d.sync += [
                                    v1.eq(idx),
                                    vertex_count.eq(1),
                                ]
                            with m.Case(1):
                                m.d.sync += [
                                    v2.eq(idx),
                                    vertex_count.eq(2),
                                ]
                            with m.Case(2):
                                m.d.sync += [
                                    to_send[2].eq(v1),
                                    to_send[1].eq(v2),
                                    to_send[0].eq(idx),
                                    to_send_left.eq(3),
                                    vertex_count.eq(0),
                                ]
                    with m.Case(InputTopology.LINE_STRIP):
                        with m.If(vertex_count == 0):
                            m.d.sync += [
                                v1.eq(idx),
                                vertex_count.eq(1),
                            ]
                        with m.Else():
                            m.d.sync += [
                                to_send[1].eq(v1),
                                to_send[0].eq(idx),
                                to_send_left.eq(2),
                                v1.eq(idx),
                            ]
                    with m.Case(InputTopology.TRIANGLE_STRIP):
                        with m.Switch(vertex_count):
                            with m.Case(0):
                                m.d.sync += [
                                    v1.eq(idx),
                                    vertex_count.eq(1),
                                ]
                            with m.Case(1):
                                m.d.sync += [
                                    v2.eq(idx),
                                    vertex_count.eq(2),
                                ]
                            with m.Case(2):
                                # Odd triangle -> indexes n, n+1, n+2
                                # so v1, v2, idx
                                m.d.sync += [
                                    to_send[2].eq(v1),
                                    to_send[1].eq(v2),
                                    to_send[0].eq(idx),
                                    to_send_left.eq(3),
                                    vertex_count.eq(3),
                                    v1.eq(v2),
                                    v2.eq(idx),
                                ]
                            with m.Case(3):
                                # Even triangle -> indexes n+1, n, n+2
                                # so v2, v1, idx
                                m.d.sync += [
                                    to_send[2].eq(v2),
                                    to_send[1].eq(v1),
                                    to_send[0].eq(idx),
                                    to_send_left.eq(3),
                                    v1.eq(v2),
                                    v2.eq(idx),
                                ]
                    with m.Case(InputTopology.TRIANGLE_FAN):
                        with m.Switch(vertex_count):
                            with m.Case(0):
                                m.d.sync += [
                                    v1.eq(idx),  # center vertex
                                    vertex_count.eq(1),
                                ]
                            with m.Case(1):
                                m.d.sync += [
                                    v2.eq(idx),  # first outer vertex
                                    vertex_count.eq(2),
                                ]
                            with m.Default():
                                m.d.sync += [
                                    to_send[2].eq(v1),  # center vertex
                                    to_send[1].eq(v2),  # previous outer vertex
                                    to_send[0].eq(idx),  # current outer vertex
                                    to_send_left.eq(3),
                                    v2.eq(idx),
                                ]
                    with m.Default():
                        m.d.sync += Assert(
                            0, "unsupported topology"
                        )  # unsupported topology

        return m


class InputAssemblyAttrConfigLayout(data.Struct):
    """Single vertex attribute configuration"""

    mode: InputMode
    info: InputData


class InputAssembly(wiring.Component):
    """Input Assembly stage.

    Gets index stream and outputs vertex attribute stream.

    Takes configuration for vertex attributes including position, normal, texcoords, and color.
    Each attribute can be constant or per-vertex.

    TODO: support other formats than Fixed 16.16
    TODO: add memory burst support
    """

    is_index: In(stream.Signature(index_shape))
    os_vertex: Out(stream.Signature(VertexLayout))
    bus: Out(wb.Signature(addr_width=wb_bus_addr_width, data_width=wb_bus_data_width))
    ready: Out(1)

    c_pos: In(InputAssemblyAttrConfigLayout)
    c_norm: In(InputAssemblyAttrConfigLayout)
    c_tex: In(InputAssemblyAttrConfigLayout).array(num_textures)
    c_col: In(InputAssemblyAttrConfigLayout)

    def elaborate(self, platform) -> Module:
        m = Module()

        # fetch vertex at given index based on vertex input attributes

        idx = Signal.like(self.is_index.payload)
        vtx = Signal.like(self.os_vertex.payload)

        addr = Signal.like(self.c_pos.info.per_vertex.address)

        output_next_free = ~self.os_vertex.valid | self.os_vertex.ready

        with m.If(self.os_vertex.ready):
            m.d.sync += self.os_vertex.valid.eq(0)

        @dataclass
        class AttrInfo:
            config: InputAssemblyAttrConfigLayout
            data_v: Signal

            @property
            def components(self) -> int:
                return len(self.data_v)

        attr_info = Array(
            [
                AttrInfo(config=self.c_pos, data_v=vtx.position),
                AttrInfo(config=self.c_norm, data_v=vtx.normal),
                *[
                    AttrInfo(config=self.c_tex[i], data_v=vtx.texcoords[i])
                    for i in range(num_textures)
                ],
                AttrInfo(config=self.c_col, data_v=vtx.color),
            ]
        )

        with m.FSM():
            with m.State("IDLE"):
                m.d.comb += [
                    self.ready.eq(~self.os_vertex.valid & ~self.is_index.valid),
                    self.is_index.ready.eq(1),
                ]

                with m.If(self.is_index.valid):
                    m.d.sync += idx.eq(self.is_index.payload)
                    m.next = "FETCH_ATTR_0_START"

            for attr_no, attr in enumerate(attr_info):
                base_name = f"FETCH_ATTR_{attr_no}"
                with m.State(f"{base_name}_START"):
                    config = attr.config
                    with m.Switch(config.mode):
                        with m.Case(InputMode.CONSTANT):
                            # constant value
                            m.d.sync += [
                                attr.data_v[i].eq(config.info.constant_value[i])
                                for i in range(attr.components)
                            ]
                            m.next = f"{base_name}_DONE"
                        with m.Case(InputMode.PER_VERTEX):
                            # per-vertex attribute
                            base_addr = config.info.per_vertex.address
                            stride = config.info.per_vertex.stride
                            m.d.sync += addr.eq(base_addr + idx * stride)
                            m.next = f"{base_name}_MEM_READ_COMPONENT_0"
                        with m.Default():
                            m.d.sync += Assert(0, "Unknown input mode")

                for i in range(len(attr.data_v)):
                    with m.State(f"{base_name}_MEM_READ_COMPONENT_{i}"):
                        # initiate memory read
                        m.d.sync += [
                            self.bus.cyc.eq(1),
                            self.bus.adr.eq(addr // (self.bus.data_width // 8)),
                            self.bus.we.eq(0),
                            self.bus.stb.eq(1),
                            self.bus.sel.eq(~0),
                        ]
                        m.next = f"{base_name}_MEM_WAIT_{i}"

                    with m.State(f"{base_name}_MEM_WAIT_{i}"):
                        with m.If(self.bus.ack):
                            # parse and store
                            m.d.sync += attr.data_v[i].eq(
                                FixedPoint_mem(self.bus.dat_r)
                            )

                            # deassert memory access
                            m.d.sync += [
                                self.bus.cyc.eq(0),
                                self.bus.stb.eq(0),
                            ]

                            if i + 1 < attr.components:
                                # next component

                                # 4 bytes per component (Fixed point 16.16)
                                m.d.sync += addr.eq(addr + 4)
                                m.next = f"{base_name}_MEM_READ_COMPONENT_{i + 1}"
                            else:
                                # all components read
                                m.next = f"{base_name}_DONE"

                with m.State(f"{base_name}_DONE"):
                    if attr_no == len(attr_info) - 1:
                        # last attribute -> output vertex
                        with m.If(output_next_free):
                            m.d.sync += [
                                self.os_vertex.payload.eq(vtx),
                                self.os_vertex.valid.eq(1),
                            ]
                            m.next = "IDLE"
                    else:
                        m.next = f"FETCH_ATTR_{attr_no + 1}_START"

        return m
