"""Per-fragment operations for pixel shading with memory access.

Implements the fragment pipeline with memory access:
1. Texturing (texture fetch and filtering)
2. Stencil test (read, compare, write)
3. Depth test (read, compare, write)
4. Blending (read destination, blend, write)
5. Write to framebuffer (color + depth + stencil)
"""

import amaranth_soc.wishbone.bus as wb
from amaranth import *
from amaranth.lib import data, enum, stream, wiring
from amaranth.lib.wiring import In, Out

from gpu.utils import fixed

from ..utils.layouts import (
    FragmentLayout,
    FramebufferInfoLayout,
    wb_bus_addr_width,
    wb_bus_data_width,
)
from ..utils.types import CompareOp, FixedPoint

one = fixed.Const(1.0)
zero = fixed.Const(0.0)


class StencilOp(enum.Enum, shape=unsigned(3)):
    """Stencil operations (what to do with stencil value)"""

    KEEP = 0
    ZERO = 1
    REPLACE = 2
    INCR = 3
    DECR = 4
    INVERT = 5
    INCR_WRAP = 6
    DECR_WRAP = 7


class BlendOp(enum.Enum, shape=unsigned(3)):
    """Blending operations"""

    ADD = 0
    SUBTRACT = 1
    REVERSE_SUBTRACT = 2
    MIN = 3
    MAX = 4


class BlendFactor(enum.Enum, shape=unsigned(4)):
    """Blending factors"""

    ZERO = 0
    ONE = 1
    SRC_COLOR = 2
    ONE_MINUS_SRC_COLOR = 3
    DST_COLOR = 4
    ONE_MINUS_DST_COLOR = 5
    SRC_ALPHA = 6
    ONE_MINUS_SRC_ALPHA = 7
    DST_ALPHA = 8
    ONE_MINUS_DST_ALPHA = 9


class StencilOpConfig(data.Struct):
    """Stencil operation configuration"""

    compare_op: CompareOp
    pass_op: StencilOp
    fail_op: StencilOp
    depth_fail_op: StencilOp
    _1: 4
    reference: unsigned(8)
    mask: unsigned(8)
    write_mask: unsigned(8)


class DepthTestConfig(data.Struct):
    """Depth test configuration"""

    test_enabled: 1
    write_enabled: 1
    compare_op: CompareOp
    _1: 3


class BlendConfig(data.Struct):
    """Blending configuration"""

    src_factor: BlendFactor
    dst_factor: BlendFactor
    src_a_factor: BlendFactor
    dst_a_factor: BlendFactor
    enabled: 1
    blend_op: BlendOp
    blend_a_op: BlendOp
    _1: 1
    color_write_mask: 4
    _2: 4


class Texturing(wiring.Component):
    """Texture fetch and filtering unit.

    Currently a stub that just passes through the fragment data,
    dropping texture coordinates.
    """

    def __init__(self):
        super().__init__(
            {
                "is_fragment": In(stream.Signature(FragmentLayout)),
                "os_fragment": Out(stream.Signature(FragmentLayout)),
                "ready": Out(1),
            }
        )

    def elaborate(self, platform):
        m = Module()

        m.d.comb += self.ready.eq(1)

        wiring.connect(
            m, wiring.flipped(self.is_fragment), wiring.flipped(self.os_fragment)
        )

        return m


class DepthStencilTest(wiring.Component):
    is_fragment: stream.Interface
    os_fragment: stream.Interface

    stencil_conf_front: Value
    stencil_conf_back: Value

    depth_conf: Value
    fb_info: Value

    wb_bus: wb.Interface

    def __init__(self):
        super().__init__(
            {
                "is_fragment": In(stream.Signature(FragmentLayout)),
                "os_fragment": Out(stream.Signature(FragmentLayout)),
                "stencil_conf_front": In(StencilOpConfig),
                "stencil_conf_back": In(StencilOpConfig),
                "depth_conf": In(DepthTestConfig),
                "fb_info": In(FramebufferInfoLayout),
                "wb_bus": Out(
                    wb.Signature(
                        addr_width=wb_bus_addr_width,
                        data_width=wb_bus_data_width,
                        granularity=8,
                    )
                ),
                "ready": Out(1),
            }
        )

    def elaborate(self, platform):
        m = Module()

        v = Signal.like(self.is_fragment.payload)

        stencil_addr = Signal(wb_bus_addr_width)
        stencil_offset = Signal(range(4))

        depth_addr = Signal(wb_bus_addr_width)
        depth_offset = Signal(range(4))

        stencil_value = Signal(unsigned(8))
        depth_value = Signal(unsigned(16))

        d_frag = Signal(unsigned(16))

        s_conf = Signal(StencilOpConfig)

        s_accepted = Signal()
        d_accepted = Signal()

        stencil_needs_read = Signal()
        depth_needs_read = Signal()

        m.d.comb += s_conf.eq(
            Mux(v.front_facing, self.stencil_conf_front, self.stencil_conf_back)
        )

        with m.Switch(s_conf.compare_op):
            with m.Case(CompareOp.NEVER):
                m.d.comb += stencil_needs_read.eq(
                    (s_conf.write_mask != 0)
                    & (s_conf.fail_op != StencilOp.ZERO)
                    & (s_conf.fail_op != StencilOp.REPLACE)
                )
            with m.Case(CompareOp.ALWAYS):
                m.d.comb += stencil_needs_read.eq(
                    (s_conf.write_mask != 0)
                    & (
                        (
                            (s_conf.pass_op != StencilOp.ZERO)
                            & (s_conf.pass_op != StencilOp.REPLACE)
                        )
                        | (
                            (s_conf.depth_fail_op != StencilOp.ZERO)
                            & (s_conf.depth_fail_op != StencilOp.REPLACE)
                        )
                    )
                )
            with m.Default():
                m.d.comb += stencil_needs_read.eq(1)

        with m.Switch(self.depth_conf.compare_op):
            with m.Case(CompareOp.NEVER):
                m.d.comb += depth_needs_read.eq(0)
            with m.Case(CompareOp.ALWAYS):
                m.d.comb += depth_needs_read.eq(0)
            with m.Default():
                m.d.comb += depth_needs_read.eq(self.depth_conf.test_enabled)

        def perform_compare(op, value, reference):
            """Perform comparison operation (combinational)."""
            ret = Signal()
            with m.Switch(op):
                with m.Case(CompareOp.NEVER):
                    m.d.comb += ret.eq(0)
                with m.Case(CompareOp.LESS):
                    m.d.comb += ret.eq(value < reference)
                with m.Case(CompareOp.EQUAL):
                    m.d.comb += ret.eq(value == reference)
                with m.Case(CompareOp.LESS_OR_EQUAL):
                    m.d.comb += ret.eq(value <= reference)
                with m.Case(CompareOp.GREATER):
                    m.d.comb += ret.eq(value > reference)
                with m.Case(CompareOp.NOT_EQUAL):
                    m.d.comb += ret.eq(value != reference)
                with m.Case(CompareOp.GREATER_OR_EQUAL):
                    m.d.comb += ret.eq(value >= reference)
                with m.Case(CompareOp.ALWAYS):
                    m.d.comb += ret.eq(1)
            return ret

        with m.FSM():
            with m.State("IDLE"):
                m.d.comb += [self.is_fragment.ready.eq(1), self.ready.eq(1)]
                with m.If(self.is_fragment.valid):
                    m.d.sync += v.eq(self.is_fragment.payload)

                    m.d.sync += s_accepted.eq(0)
                    m.d.sync += d_accepted.eq(0)

                    # TODO: handle minDepth and maxDepth from fb_info
                    depth_zero_one = ((self.is_fragment.p.depth + one) >> 1).clamp(
                        zero, one
                    )
                    m.d.sync += d_frag.eq(
                        ((depth_zero_one << 16) - depth_zero_one).round()
                    )

                    m.next = "PREPARE"

            with m.State("PREPARE"):
                m.d.sync += [
                    Cat(stencil_offset, stencil_addr).eq(
                        self.fb_info.stencil_address
                        + (v.coord_pos[1] * self.fb_info.stencil_pitch)
                        + v.coord_pos[0]
                    ),
                    Cat(depth_offset, depth_addr).eq(
                        self.fb_info.depth_address
                        + (v.coord_pos[1] * self.fb_info.depth_pitch)
                        + (v.coord_pos[0] * 2)
                    ),
                ]

                with m.If(stencil_needs_read):
                    m.next = "READ_STENCIL"
                with m.Elif(depth_needs_read):
                    m.next = "READ_DEPTH"
                with m.Else():
                    m.next = "OUTPUT_STENCIL"

            with m.State("READ_STENCIL"):
                m.d.comb += [
                    self.wb_bus.cyc.eq(1),
                    self.wb_bus.adr.eq(stencil_addr),
                    self.wb_bus.we.eq(0),
                    self.wb_bus.stb.eq(1),
                    self.wb_bus.sel.eq(0x1 << stencil_offset),
                ]
                with m.If(self.wb_bus.ack):
                    m.d.sync += stencil_value.eq(
                        self.wb_bus.dat_r.word_select(stencil_offset, 8)
                    )
                    with m.If(depth_needs_read):
                        m.next = "READ_DEPTH"
                    with m.Else():
                        m.next = "CHECK_DEPTH_STENCIL"

            with m.State("READ_DEPTH"):
                m.d.comb += [
                    self.wb_bus.cyc.eq(1),
                    self.wb_bus.adr.eq(depth_addr),
                    self.wb_bus.we.eq(0),
                    self.wb_bus.stb.eq(1),
                    self.wb_bus.sel.eq(0x3 << depth_offset),
                ]
                with m.If(self.wb_bus.ack):
                    m.d.sync += Print(
                        "Reading depth from address:",
                        depth_addr,
                        " got: ",
                        self.wb_bus.dat_r,
                    )

                    m.d.sync += depth_value.eq(
                        self.wb_bus.dat_r.word_select(depth_offset[1:], 16)
                    )
                    m.next = "CHECK_DEPTH_STENCIL"

            with m.State("CHECK_DEPTH_STENCIL"):
                s_passed = perform_compare(
                    s_conf.compare_op,
                    stencil_value & s_conf.mask,
                    s_conf.reference & s_conf.mask,
                )
                d_passed = perform_compare(
                    self.depth_conf.compare_op, d_frag, depth_value
                )

                m.d.sync += s_accepted.eq(s_passed)
                m.d.sync += d_accepted.eq(d_passed | ~self.depth_conf.test_enabled)
                m.d.sync += [
                    Print(
                        Format(
                            "Stencil test: value={}, ref={}, passed={}",
                            stencil_value,
                            s_conf.reference,
                            s_passed,
                        )
                    ),
                    Print(
                        Format(
                            "Depth test: value={}, frag_depth={}, passed={}",
                            depth_value,
                            d_frag,
                            d_passed,
                        )
                    ),
                ]
                m.next = "OUTPUT_STENCIL"

            with m.State("OUTPUT_STENCIL"):
                # perform depth/stencil updates if accepted

                stencil_op_to_do = Signal(StencilOp)
                new_stencil_value = Signal(unsigned(8))

                with m.If(~s_accepted):
                    m.d.comb += stencil_op_to_do.eq(s_conf.fail_op)
                with m.Elif(~d_accepted):
                    m.d.comb += stencil_op_to_do.eq(s_conf.depth_fail_op)
                with m.Else():
                    m.d.comb += stencil_op_to_do.eq(s_conf.pass_op)

                with m.Switch(stencil_op_to_do):
                    with m.Case(StencilOp.KEEP):
                        pass
                    with m.Case(StencilOp.ZERO):
                        m.d.comb += new_stencil_value.eq(0)
                    with m.Case(StencilOp.REPLACE):
                        m.d.comb += new_stencil_value.eq(s_conf.reference)
                    with m.Case(StencilOp.INCR):
                        with m.If(stencil_value != 0xFF):
                            m.d.comb += new_stencil_value.eq(stencil_value + 1)
                    with m.Case(StencilOp.DECR):
                        with m.If(stencil_value != 0x00):
                            m.d.comb += new_stencil_value.eq(stencil_value - 1)
                    with m.Case(StencilOp.INVERT):
                        m.d.comb += new_stencil_value.eq(~stencil_value)
                    with m.Case(StencilOp.INCR_WRAP):
                        m.d.comb += new_stencil_value.eq(stencil_value + 1)
                    with m.Case(StencilOp.DECR_WRAP):
                        m.d.comb += new_stencil_value.eq(stencil_value - 1)

                real_new_stencil_value = Signal(unsigned(8))
                m.d.comb += [
                    real_new_stencil_value.eq(
                        (stencil_value & ~s_conf.write_mask)
                        | (new_stencil_value & s_conf.write_mask)
                    )
                ]

                m.d.sync += Print("stencil accepted: ", s_accepted)
                m.d.sync += Print("depth accepted: ", d_accepted)
                m.d.sync += Print("Stencil op:", stencil_op_to_do)
                m.d.sync += Print(Format("Old value: {:#02x}", stencil_value))
                m.d.sync += Print(Format("New value: {:#02x}", real_new_stencil_value))

                with m.If(real_new_stencil_value != stencil_value):
                    m.d.sync += Print(
                        Format(
                            "Writing {:#02x} stencil to address {} offset {}",
                            real_new_stencil_value,
                            stencil_addr,
                            stencil_offset,
                        )
                    )
                    m.d.comb += [
                        self.wb_bus.cyc.eq(1),
                        self.wb_bus.adr.eq(stencil_addr),
                        self.wb_bus.we.eq(1),
                        self.wb_bus.stb.eq(1),
                        self.wb_bus.sel.eq(0x1 << stencil_offset),
                        self.wb_bus.dat_w.eq(
                            real_new_stencil_value << (stencil_offset * 8)
                        ),
                    ]
                    with m.If(self.wb_bus.ack):
                        with m.If(~s_accepted | ~d_accepted):
                            m.next = "IDLE"
                        with m.Elif(self.depth_conf.write_enabled):
                            m.next = "OUTPUT_DEPTH"
                        with m.Else():
                            m.next = "SEND"
                with m.Else():
                    with m.If(~s_accepted | ~d_accepted):
                        m.next = "IDLE"
                    with m.Elif(self.depth_conf.write_enabled):
                        m.next = "OUTPUT_DEPTH"
                    with m.Else():
                        m.next = "SEND"

            with m.State("OUTPUT_DEPTH"):
                m.d.sync += Print(
                    Format("Writing depth value: {}, {}", v.depth, d_frag)
                )

                m.d.sync += Print(
                    Format(
                        "Writing {} depth to address {} offset {}",
                        d_frag,
                        depth_addr,
                        depth_offset,
                    )
                )

                m.d.comb += [
                    self.wb_bus.cyc.eq(1),
                    self.wb_bus.adr.eq(depth_addr),
                    self.wb_bus.we.eq(1),
                    self.wb_bus.stb.eq(1),
                    self.wb_bus.sel.eq(0x3 << depth_offset),
                    self.wb_bus.dat_w.eq(d_frag << (depth_offset * 8)),
                ]
                with m.If(self.wb_bus.ack):
                    m.next = "SEND"

            with m.State("SEND"):
                m.d.comb += self.os_fragment.valid.eq(1)
                m.d.comb += self.os_fragment.payload.eq(v)
                with m.If(self.os_fragment.ready):
                    m.next = "IDLE"

        return m


class SwapchainOutput(wiring.Component):
    """Perform blending and write final fragment to framebuffer memory."""

    def __init__(self):
        super().__init__(
            {
                "is_fragment": In(stream.Signature(FragmentLayout)),
                "conf": In(BlendConfig),
                "fb_info": In(FramebufferInfoLayout),
                "wb_bus": Out(
                    wb.Signature(
                        addr_width=wb_bus_addr_width,
                        data_width=wb_bus_data_width,
                    )
                ),
                "ready": Out(1),
            }
        )

    def elaborate(self, platform):
        m = Module()

        color_shape = fixed.UQ(0, 16)

        src_data = Signal(data.ArrayLayout(color_shape, 4))
        dst_data = Signal(data.ArrayLayout(color_shape, 4))
        out_data = Signal(data.ArrayLayout(color_shape, 4))

        color_addr = Signal(wb_bus_addr_width)

        src_rgb = src_data[0:3]
        src_a = src_data[3]
        dst_rgb = dst_data[0:3]
        dst_a = dst_data[3]

        def factor_value(factor):
            ret = Signal(color_shape)
            with m.Switch(factor):
                with m.Case(BlendFactor.ZERO):
                    m.d.comb += ret.eq(0.0)
                with m.Case(BlendFactor.ONE):
                    m.d.comb += ret.eq(color_shape.max())
                with m.Case(BlendFactor.SRC_COLOR):
                    m.d.sync += Assert(
                        False, "Not implemented: SRC_COLOR factor in blending"
                    )
                with m.Case(BlendFactor.ONE_MINUS_SRC_COLOR):
                    m.d.sync += Assert(
                        False, "Not implemented: ONE_MINUS_SRC_COLOR factor in blending"
                    )
                with m.Case(BlendFactor.DST_COLOR):
                    m.d.sync += Assert(
                        False, "Not implemented: DST_COLOR factor in blending"
                    )
                with m.Case(BlendFactor.ONE_MINUS_DST_COLOR):
                    m.d.sync += Assert(
                        False, "Not implemented: ONE_MINUS_DST_COLOR factor in blending"
                    )
                with m.Case(BlendFactor.SRC_ALPHA):
                    m.d.comb += ret.eq(src_a)
                with m.Case(BlendFactor.ONE_MINUS_SRC_ALPHA):
                    m.d.comb += ret.eq(color_shape.max() - src_a)
                with m.Case(BlendFactor.DST_ALPHA):
                    m.d.comb += ret.eq(dst_a)
                with m.Case(BlendFactor.ONE_MINUS_DST_ALPHA):
                    m.d.comb += ret.eq(color_shape.max() - dst_a)
            return ret

        with m.FSM():
            with m.State("IDLE"):
                m.d.comb += [self.is_fragment.ready.eq(1), self.ready.eq(1)]
                with m.If(self.is_fragment.valid):
                    m.d.sync += color_addr.eq(
                        self.fb_info.color_address[2:]
                        + (
                            self.is_fragment.p.coord_pos[1]
                            * self.fb_info.color_pitch[2:]
                        )
                        + (self.is_fragment.p.coord_pos[0])
                    )
                    in_data = Signal(data.ArrayLayout(color_shape, 4))
                    m.d.comb += [
                        in_data[i].eq(
                            (self.is_fragment.payload.color[i]).saturate(color_shape)
                        )
                        for i in range(4)
                    ]
                    with m.If(self.conf.enabled):
                        m.d.sync += src_data.eq(in_data)
                        m.next = "READ_DEST"
                    with m.Else():
                        m.d.sync += out_data.eq(in_data)
                        m.next = "WRITE_OUTPUT"

            with m.State("READ_DEST"):
                m.d.comb += [
                    self.wb_bus.cyc.eq(1),
                    self.wb_bus.adr.eq(color_addr),
                    self.wb_bus.we.eq(0),
                    self.wb_bus.stb.eq(1),
                    self.wb_bus.sel.eq(~0),
                ]
                with m.If(self.wb_bus.ack):
                    m.d.sync += [
                        dst_data[i].eq(
                            (
                                self.wb_bus.dat_r.word_select(i, 8)
                                * fixed.Const(1.0 / 255.0, shape=color_shape)
                            ).saturate(color_shape)
                        )
                        for i in range(4)
                    ]
                    m.next = "BLEND_RGB"

            with m.State("BLEND_RGB"):
                src_factor = factor_value(self.conf.src_factor)
                dst_factor = factor_value(self.conf.dst_factor)

                for i in range(3):
                    src_comp = src_rgb[i]
                    dst_comp = dst_rgb[i]

                    src_scaled = Signal(FixedPoint)
                    dst_scaled = Signal(FixedPoint)

                    m.d.comb += [
                        src_scaled.eq(src_comp * src_factor),
                        dst_scaled.eq(dst_comp * dst_factor),
                    ]

                    with m.Switch(self.conf.blend_op):
                        with m.Case(BlendOp.ADD):
                            m.d.sync += out_data[i].eq(
                                (src_scaled + dst_scaled).saturate(color_shape)
                            )
                        with m.Case(BlendOp.SUBTRACT):
                            m.d.sync += out_data[i].eq(
                                (src_scaled - dst_scaled).saturate(color_shape)
                            )
                        with m.Case(BlendOp.REVERSE_SUBTRACT):
                            m.d.sync += out_data[i].eq(
                                (dst_scaled - src_scaled).saturate(color_shape)
                            )
                        with m.Case(BlendOp.MIN):
                            with m.If(src_comp < dst_comp):
                                m.d.sync += out_data[i].eq(src_comp)
                            with m.Else():
                                m.d.sync += out_data[i].eq(dst_comp)
                        with m.Case(BlendOp.MAX):
                            with m.If(src_comp > dst_comp):
                                m.d.sync += out_data[i].eq(src_comp)
                            with m.Else():
                                m.d.sync += out_data[i].eq(dst_comp)
                m.next = "BLEND_A"

            with m.State("BLEND_A"):
                src_a_factor = factor_value(self.conf.src_a_factor)
                dst_a_factor = factor_value(self.conf.dst_a_factor)

                src_scaled = Signal(FixedPoint)
                dst_scaled = Signal(FixedPoint)

                m.d.comb += [
                    src_scaled.eq(src_a * src_a_factor),
                    dst_scaled.eq(dst_a * dst_a_factor),
                ]

                with m.Switch(self.conf.blend_a_op):
                    with m.Case(BlendOp.ADD):
                        m.d.sync += out_data[3].eq(
                            (src_scaled + dst_scaled).saturate(color_shape)
                        )
                    with m.Case(BlendOp.SUBTRACT):
                        m.d.sync += out_data[3].eq(
                            (src_scaled - dst_scaled).saturate(color_shape)
                        )
                    with m.Case(BlendOp.REVERSE_SUBTRACT):
                        m.d.sync += out_data[3].eq(
                            (dst_scaled - src_scaled).saturate(color_shape)
                        )
                    with m.Case(BlendOp.MIN):
                        with m.If(src_a < dst_a):
                            m.d.sync += out_data[3].eq(src_a)
                        with m.Else():
                            m.d.sync += out_data[3].eq(dst_a)
                    with m.Case(BlendOp.MAX):
                        with m.If(src_a > dst_a):
                            m.d.sync += out_data[3].eq(src_a)
                        with m.Else():
                            m.d.sync += out_data[3].eq(dst_a)

                m.next = "WRITE_OUTPUT"

            with m.State("WRITE_OUTPUT"):
                ret_v = Signal(data.ArrayLayout(unsigned(8), 4))

                m.d.comb += [
                    # Convert from fixed-point [0,1] to [0,255]
                    ret_v[i].eq(((out_data[i] << 8) - out_data[i]).round())
                    for i in range(4)
                ]
                m.d.comb += [
                    self.wb_bus.cyc.eq(1),
                    self.wb_bus.adr.eq(color_addr),
                    self.wb_bus.we.eq(1),
                    self.wb_bus.stb.eq(1),
                    self.wb_bus.sel.eq(self.conf.color_write_mask),
                    self.wb_bus.dat_w.eq(Cat(ret_v)),
                ]
                with m.If(self.wb_bus.ack):
                    m.next = "IDLE"

        return m
