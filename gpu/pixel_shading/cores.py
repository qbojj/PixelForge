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
from amaranth.lib import data, enum, fifo, stream, wiring
from amaranth.lib.wiring import In, Out

from gpu.utils import fixed

from ..utils.layouts import (
    FragmentLayout,
    FramebufferInfoLayout,
    wb_bus_addr_width,
    wb_bus_data_width,
)
from ..utils.mem_pipelined import (
    WishbonePipelinedMaster,
    mem_read_response_layout,
)
from ..utils.types import CompareOp

BGRA_MAP = [2, 1, 0, 3]  # Mapping from RGBA to BGRA order


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
                "i": In(stream.Signature(FragmentLayout)),
                "o": Out(stream.Signature(FragmentLayout)),
                "ready": Out(1),
            }
        )

    def elaborate(self, platform):
        m = Module()

        m.d.comb += self.ready.eq(1)

        wiring.connect(m, wiring.flipped(self.i), wiring.flipped(self.o))

        return m


class DepthStencilTest(wiring.Component):
    i: stream.Interface
    o: stream.Interface

    stencil_conf_front: Value
    stencil_conf_back: Value

    depth_conf: Value
    fb_info: Value

    wb_bus: wb.Interface

    def __init__(self):
        super().__init__(
            {
                "i": In(stream.Signature(FragmentLayout)),
                "o": Out(stream.Signature(FragmentLayout)),
                "stencil_conf_front": In(StencilOpConfig),
                "stencil_conf_back": In(StencilOpConfig),
                "depth_conf": In(DepthTestConfig),
                "fb_info": In(FramebufferInfoLayout),
                "wb_bus": Out(
                    wb.Signature(
                        addr_width=wb_bus_addr_width,
                        data_width=wb_bus_data_width,
                        features={wb.Feature.STALL},
                    )
                ),
                "ready": Out(1),
            }
        )

    def elaborate(self, platform):
        m = Module()

        tag_width = 16
        req_depth = 8
        inflight_depth = 8
        lock_depth = 8
        fwd_depth = 8

        req_layout = mem_read_response_layout(wb_bus_data_width, tag_width)
        meta_layout = data.StructLayout(
            {
                "frag": FragmentLayout,
                "addr": unsigned(wb_bus_addr_width),
                "d_frag": unsigned(16),
            }
        )

        m.submodules.mem = mem = WishbonePipelinedMaster(
            addr_width=wb_bus_addr_width,
            data_width=wb_bus_data_width,
            tag_width=tag_width,
            req_depth=req_depth,
            inflight_depth=inflight_depth,
        )
        wiring.connect(m, mem.wb_bus, wiring.flipped(self.wb_bus))

        m.submodules.meta_fifo = meta_fifo = fifo.SyncFIFOBuffered(
            width=Shape.cast(meta_layout).width,
            depth=lock_depth,
        )
        m.submodules.read_req_fifo = read_req_fifo = fifo.SyncFIFOBuffered(
            width=Shape.cast(req_layout).width,
            depth=req_depth,
        )
        m.submodules.write_req_fifo = write_req_fifo = fifo.SyncFIFOBuffered(
            width=Shape.cast(req_layout).width,
            depth=req_depth,
        )
        m.submodules.out_fifo = out_fifo = fifo.SyncFIFOBuffered(
            width=Shape.cast(FragmentLayout).width,
            depth=lock_depth,
        )

        lock_valid = Array(Signal() for _ in range(lock_depth))
        lock_addr = Array(
            Signal(unsigned(wb_bus_addr_width)) for _ in range(lock_depth)
        )

        fwd_valid = Array(Signal() for _ in range(fwd_depth))
        fwd_addr = Array(Signal(unsigned(wb_bus_addr_width)) for _ in range(fwd_depth))
        fwd_data = Array(Signal(unsigned(wb_bus_data_width)) for _ in range(fwd_depth))
        fwd_seq = Array(Signal(unsigned(tag_width)) for _ in range(fwd_depth))
        write_seq_counter = Signal(unsigned(tag_width))

        def compare(op, value, reference):
            less = value < reference
            equal = value == reference
            greater = value > reference
            match_less = (op & CompareOp.LESS == CompareOp.LESS) & less
            match_equal = (op & CompareOp.EQUAL == CompareOp.EQUAL) & equal
            match_greater = (op & CompareOp.GREATER == CompareOp.GREATER) & greater
            return match_less | match_equal | match_greater

        depth_zero_one = self.i.payload.depth.clamp(0, 1)
        d_frag_calc = Signal(unsigned(16))
        m.d.comb += d_frag_calc.eq(((depth_zero_one << 16) - depth_zero_one).round())

        addr_calc = Signal(unsigned(wb_bus_addr_width))
        depthstencil_base = self.fb_info.depthstencil_address[2:]
        depthstencil_offset = self.i.payload.coord_pos[0]
        depthstencil_row = (
            self.i.payload.coord_pos[1] * self.fb_info.depthstencil_pitch[2:]
        )
        m.d.comb += addr_calc.eq(
            depthstencil_base + depthstencil_offset + depthstencil_row
        )

        lock_conflict = Signal()
        lock_free_found = Signal()
        lock_free_idx = Signal(range(lock_depth))
        m.d.comb += [
            lock_conflict.eq(0),
            lock_free_found.eq(0),
            lock_free_idx.eq(0),
        ]
        for i in range(lock_depth):
            with m.If(lock_valid[i] & (lock_addr[i] == addr_calc)):
                m.d.comb += lock_conflict.eq(1)
            with m.If(~lock_free_found & ~lock_valid[i]):
                m.d.comb += [
                    lock_free_found.eq(1),
                    lock_free_idx.eq(i),
                ]

        input_ready = (
            meta_fifo.w_rdy & read_req_fifo.w_rdy & lock_free_found & ~lock_conflict
        )
        m.d.comb += [
            self.i.ready.eq(input_ready),
            self.ready.eq(input_ready),
        ]

        meta_in = Signal(meta_layout)
        read_req_in = Signal(req_layout)
        read_seq_counter = Signal(unsigned(tag_width))

        m.d.comb += [
            meta_in.frag.eq(self.i.payload),
            meta_in.addr.eq(addr_calc),
            meta_in.d_frag.eq(d_frag_calc),
            read_req_in.addr.eq(addr_calc),
            read_req_in.data.eq(0),
            read_req_in.sel.eq(~0),
            read_req_in.we.eq(0),
            read_req_in.tag.eq(read_seq_counter),
            meta_fifo.w_en.eq(self.i.valid & input_ready),
            meta_fifo.w_data.eq(meta_in),
            read_req_fifo.w_en.eq(self.i.valid & input_ready),
            read_req_fifo.w_data.eq(read_req_in),
        ]

        with m.If(self.i.valid & input_ready):
            m.d.sync += [
                lock_valid[lock_free_idx].eq(1),
                lock_addr[lock_free_idx].eq(addr_calc),
                read_seq_counter.eq(read_seq_counter + 1),
            ]

        rr_toggle = Signal()
        read_req_out = Signal(req_layout)
        write_req_out = Signal(req_layout)
        m.d.comb += [
            read_req_out.eq(read_req_fifo.r_data),
            write_req_out.eq(write_req_fifo.r_data),
        ]

        both_reqs = read_req_fifo.r_rdy & write_req_fifo.r_rdy
        sel_write = Signal()
        m.d.comb += sel_write.eq(Mux(both_reqs, rr_toggle, write_req_fifo.r_rdy))

        mem_req_valid = read_req_fifo.r_rdy | write_req_fifo.r_rdy
        mem_req_in = Signal(req_layout)
        m.d.comb += [
            mem_req_in.addr.eq(Mux(sel_write, write_req_out.addr, read_req_out.addr)),
            mem_req_in.data.eq(Mux(sel_write, write_req_out.data, read_req_out.data)),
            mem_req_in.sel.eq(Mux(sel_write, write_req_out.sel, read_req_out.sel)),
            mem_req_in.we.eq(Mux(sel_write, write_req_out.we, read_req_out.we)),
            mem_req_in.tag.eq(Mux(sel_write, write_req_out.tag, read_req_out.tag)),
            mem.req.valid.eq(mem_req_valid),
            mem.req.payload.eq(mem_req_in),
        ]

        grant = mem.req.ready & mem_req_valid
        m.d.comb += [
            read_req_fifo.r_en.eq(grant & ~sel_write),
            write_req_fifo.r_en.eq(grant & sel_write),
        ]
        with m.If(grant & both_reqs):
            m.d.sync += rr_toggle.eq(~rr_toggle)

        meta_out = Signal(meta_layout)
        m.d.comb += meta_out.eq(meta_fifo.r_data)

        read_data = Signal(unsigned(wb_bus_data_width))
        m.d.comb += read_data.eq(mem.read_resp.payload.data)

        fwd_hit = Signal()
        fwd_data_sel = Signal(unsigned(wb_bus_data_width))
        fwd_match = Signal()
        fwd_match_idx = Signal(range(fwd_depth))
        fwd_free_found = Signal()
        fwd_free_idx = Signal(range(fwd_depth))
        m.d.comb += [
            fwd_hit.eq(0),
            fwd_data_sel.eq(0),
            fwd_match.eq(0),
            fwd_match_idx.eq(0),
            fwd_free_found.eq(0),
            fwd_free_idx.eq(0),
        ]
        for i in range(fwd_depth):
            with m.If(fwd_valid[i] & (fwd_addr[i] == meta_out.addr)):
                m.d.comb += [
                    fwd_hit.eq(1),
                    fwd_data_sel.eq(fwd_data[i]),
                    fwd_match.eq(1),
                    fwd_match_idx.eq(i),
                ]
            with m.If(~fwd_free_found & ~fwd_valid[i]):
                m.d.comb += [
                    fwd_free_found.eq(1),
                    fwd_free_idx.eq(i),
                ]

        depthstencil_data = Signal(unsigned(wb_bus_data_width))
        m.d.comb += depthstencil_data.eq(Mux(fwd_hit, fwd_data_sel, read_data))
        depth_value = depthstencil_data[0:16]
        stencil_value = depthstencil_data[24:32]

        s_conf = Signal(StencilOpConfig)
        m.d.comb += s_conf.eq(
            Mux(
                meta_out.frag.front_facing,
                self.stencil_conf_front,
                self.stencil_conf_back,
            )
        )

        s_passed = compare(
            s_conf.compare_op,
            stencil_value & s_conf.mask,
            s_conf.reference & s_conf.mask,
        )
        d_passed = compare(self.depth_conf.compare_op, meta_out.d_frag, depth_value)

        s_accepted = Signal()
        d_accepted = Signal()
        m.d.comb += [
            s_accepted.eq(s_passed),
            d_accepted.eq(d_passed | ~self.depth_conf.test_enabled),
        ]

        stencil_op_to_do = Signal(StencilOp)
        new_stencil_value = Signal(unsigned(8))
        m.d.comb += new_stencil_value.eq(stencil_value)
        with m.If(~s_accepted):
            m.d.comb += stencil_op_to_do.eq(s_conf.fail_op)
        with m.Elif(~d_accepted):
            m.d.comb += stencil_op_to_do.eq(s_conf.depth_fail_op)
        with m.Else():
            m.d.comb += stencil_op_to_do.eq(s_conf.pass_op)

        with m.Switch(stencil_op_to_do):
            with m.Case(StencilOp.KEEP):
                m.d.comb += new_stencil_value.eq(stencil_value)
            with m.Case(StencilOp.ZERO):
                m.d.comb += new_stencil_value.eq(0)
            with m.Case(StencilOp.REPLACE):
                m.d.comb += new_stencil_value.eq(s_conf.reference)
            with m.Case(StencilOp.INCR):
                m.d.comb += new_stencil_value.eq(
                    Mux(stencil_value == 0xFF, stencil_value, stencil_value + 1)
                )
            with m.Case(StencilOp.DECR):
                m.d.comb += new_stencil_value.eq(
                    Mux(stencil_value == 0x00, stencil_value, stencil_value - 1)
                )
            with m.Case(StencilOp.INVERT):
                m.d.comb += new_stencil_value.eq(~stencil_value)
            with m.Case(StencilOp.INCR_WRAP):
                m.d.comb += new_stencil_value.eq(stencil_value + 1)
            with m.Case(StencilOp.DECR_WRAP):
                m.d.comb += new_stencil_value.eq(stencil_value - 1)

        real_new_stencil_value = Signal(unsigned(8))
        for i in range(8):
            m.d.comb += real_new_stencil_value[i].eq(
                Mux(s_conf.write_mask[i], new_stencil_value[i], stencil_value[i])
            )

        new_depth_value = Signal(unsigned(16))
        m.d.comb += new_depth_value.eq(
            Mux(
                s_accepted & d_accepted & self.depth_conf.write_enabled,
                meta_out.d_frag,
                depth_value,
            )
        )

        new_depthstencil = Signal(unsigned(32))
        m.d.comb += new_depthstencil.eq(
            Cat(new_depth_value, Const(0, 8), real_new_stencil_value)
        )

        write_needed = Signal()
        m.d.comb += write_needed.eq(new_depthstencil != depthstencil_data)

        out_needed = s_accepted & d_accepted
        out_can_accept = (~out_needed) | out_fifo.w_rdy

        fwd_can_accept = fwd_match | fwd_free_found
        write_ready = ~write_needed | write_req_fifo.w_rdy
        fwd_ready = ~write_needed | fwd_can_accept
        compute_ready = out_can_accept & write_ready & fwd_ready

        compute_fire = meta_fifo.r_rdy & mem.read_resp.valid & compute_ready
        m.d.comb += [
            meta_fifo.r_en.eq(compute_fire),
            mem.read_resp.ready.eq(compute_ready & meta_fifo.r_rdy),
        ]

        write_seq = Signal(unsigned(tag_width))
        m.d.comb += write_seq.eq(write_seq_counter)

        write_req_in = Signal(req_layout)
        m.d.comb += [
            write_req_in.addr.eq(meta_out.addr),
            write_req_in.data.eq(new_depthstencil),
            write_req_in.sel.eq(~0),
            write_req_in.we.eq(1),
            write_req_in.tag.eq(write_seq),
            write_req_fifo.w_en.eq(compute_fire & write_needed),
            write_req_fifo.w_data.eq(write_req_in),
        ]

        m.d.comb += [
            out_fifo.w_en.eq(compute_fire & out_needed),
            out_fifo.w_data.eq(meta_out.frag),
        ]

        with m.If(compute_fire):
            for i in range(lock_depth):
                with m.If(lock_valid[i] & (lock_addr[i] == meta_out.addr)):
                    m.d.sync += lock_valid[i].eq(0)

            with m.If(write_needed):
                with m.If(fwd_match):
                    m.d.sync += [
                        fwd_data[fwd_match_idx].eq(new_depthstencil),
                        fwd_seq[fwd_match_idx].eq(write_seq),
                    ]
                with m.Else():
                    m.d.sync += [
                        fwd_valid[fwd_free_idx].eq(1),
                        fwd_addr[fwd_free_idx].eq(meta_out.addr),
                        fwd_data[fwd_free_idx].eq(new_depthstencil),
                        fwd_seq[fwd_free_idx].eq(write_seq),
                    ]

        with m.If(compute_fire & write_needed):
            m.d.sync += write_seq_counter.eq(write_seq + 1)

        m.d.comb += [
            mem.write_resp.ready.eq(1),
            self.o.valid.eq(out_fifo.r_rdy),
            self.o.payload.eq(out_fifo.r_data),
            out_fifo.r_en.eq(self.o.valid & self.o.ready),
        ]

        with m.If(mem.write_resp.valid):
            for i in range(fwd_depth):
                with m.If(fwd_valid[i] & (fwd_seq[i] == mem.write_resp.payload.tag)):
                    m.d.sync += fwd_valid[i].eq(0)

        return m


class SwapchainOutput(wiring.Component):
    """Perform blending and write final fragment to framebuffer memory."""

    def __init__(self):
        super().__init__(
            {
                "i": In(stream.Signature(FragmentLayout)),
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

        tag_width = 16
        req_depth = 8
        inflight_depth = 8
        lock_depth = 8
        fwd_depth = 8

        req_layout = data.StructLayout(
            {
                "addr": unsigned(wb_bus_addr_width),
                "data": unsigned(wb_bus_data_width),
                "sel": unsigned(wb_bus_data_width // 8),
                "we": unsigned(1),
                "tag": unsigned(tag_width),
            }
        )

        color_shape = fixed.UQ(0, 9)
        meta_layout = data.StructLayout(
            {
                "addr": unsigned(wb_bus_addr_width),
                "src": data.ArrayLayout(color_shape, 4),
                "need_read": unsigned(1),
            }
        )

        m.submodules.mem = mem = WishbonePipelinedMaster(
            addr_width=wb_bus_addr_width,
            data_width=wb_bus_data_width,
            tag_width=tag_width,
            req_depth=req_depth,
            inflight_depth=inflight_depth,
        )
        wiring.connect(m, mem.wb_bus, wiring.flipped(self.wb_bus))

        m.submodules.meta_fifo = meta_fifo = fifo.SyncFIFOBuffered(
            width=Shape.cast(meta_layout).width,
            depth=lock_depth,
        )
        m.submodules.read_req_fifo = read_req_fifo = fifo.SyncFIFOBuffered(
            width=Shape.cast(req_layout).width,
            depth=req_depth,
        )
        m.submodules.write_req_fifo = write_req_fifo = fifo.SyncFIFOBuffered(
            width=Shape.cast(req_layout).width,
            depth=req_depth,
        )

        lock_valid = Array(Signal() for _ in range(lock_depth))
        lock_addr = Array(
            Signal(unsigned(wb_bus_addr_width)) for _ in range(lock_depth)
        )

        fwd_valid = Array(Signal() for _ in range(fwd_depth))
        fwd_addr = Array(Signal(unsigned(wb_bus_addr_width)) for _ in range(fwd_depth))
        fwd_data = Array(Signal(unsigned(wb_bus_data_width)) for _ in range(fwd_depth))
        fwd_seq = Array(Signal(unsigned(tag_width)) for _ in range(fwd_depth))
        write_seq_counter = Signal(unsigned(tag_width))

        one = fixed.Const(1.0).saturate(color_shape)

        in_data = Signal(data.ArrayLayout(color_shape, 4))
        m.d.comb += [
            in_data[i].eq(self.i.payload.color[i].saturate(color_shape))
            for i in range(4)
        ]

        addr_calc = Signal(unsigned(wb_bus_addr_width))
        color_base = self.fb_info.color_address[2:]
        color_offset = self.i.payload.coord_pos[0]
        color_row = self.i.payload.coord_pos[1] * self.fb_info.color_pitch[2:]
        m.d.comb += addr_calc.eq(color_base + color_offset + color_row)

        lock_conflict = Signal()
        lock_free_found = Signal()
        lock_free_idx = Signal(range(lock_depth))
        m.d.comb += [
            lock_conflict.eq(0),
            lock_free_found.eq(0),
            lock_free_idx.eq(0),
        ]
        for i in range(lock_depth):
            with m.If(lock_valid[i] & (lock_addr[i] == addr_calc)):
                m.d.comb += lock_conflict.eq(1)
            with m.If(~lock_free_found & ~lock_valid[i]):
                m.d.comb += [
                    lock_free_found.eq(1),
                    lock_free_idx.eq(i),
                ]

        need_read = self.conf.enabled
        input_ready = meta_fifo.w_rdy & lock_free_found & ~lock_conflict
        input_ready = Mux(need_read, input_ready & read_req_fifo.w_rdy, input_ready)
        m.d.comb += [
            self.i.ready.eq(input_ready),
            self.ready.eq(input_ready),
        ]

        meta_in = Signal(meta_layout)
        read_req_in = Signal(req_layout)
        read_seq_counter = Signal(unsigned(tag_width))

        m.d.comb += [
            meta_in.addr.eq(addr_calc),
            meta_in.src.eq(in_data),
            meta_in.need_read.eq(need_read),
            meta_fifo.w_en.eq(self.i.valid & input_ready),
            meta_fifo.w_data.eq(meta_in),
            read_req_in.addr.eq(addr_calc),
            read_req_in.data.eq(0),
            read_req_in.sel.eq(~0),
            read_req_in.we.eq(0),
            read_req_in.tag.eq(read_seq_counter),
            read_req_fifo.w_en.eq(self.i.valid & input_ready & need_read),
            read_req_fifo.w_data.eq(read_req_in),
        ]

        with m.If(self.i.valid & input_ready):
            m.d.sync += [
                lock_valid[lock_free_idx].eq(1),
                lock_addr[lock_free_idx].eq(addr_calc),
                read_seq_counter.eq(read_seq_counter + 1),
            ]

        rr_toggle = Signal()
        read_req_out = Signal(req_layout)
        write_req_out = Signal(req_layout)
        m.d.comb += [
            read_req_out.eq(read_req_fifo.r_data),
            write_req_out.eq(write_req_fifo.r_data),
        ]

        both_reqs = read_req_fifo.r_rdy & write_req_fifo.r_rdy
        sel_write = Signal()
        m.d.comb += sel_write.eq(Mux(both_reqs, rr_toggle, write_req_fifo.r_rdy))

        mem_req_valid = read_req_fifo.r_rdy | write_req_fifo.r_rdy
        mem_req_in = Signal(req_layout)
        m.d.comb += [
            mem_req_in.addr.eq(Mux(sel_write, write_req_out.addr, read_req_out.addr)),
            mem_req_in.data.eq(Mux(sel_write, write_req_out.data, read_req_out.data)),
            mem_req_in.sel.eq(Mux(sel_write, write_req_out.sel, read_req_out.sel)),
            mem_req_in.we.eq(Mux(sel_write, write_req_out.we, read_req_out.we)),
            mem_req_in.tag.eq(Mux(sel_write, write_req_out.tag, read_req_out.tag)),
            mem.req.valid.eq(mem_req_valid),
            mem.req.payload.eq(mem_req_in),
        ]

        grant = mem.req.ready & mem_req_valid
        m.d.comb += [
            read_req_fifo.r_en.eq(grant & ~sel_write),
            write_req_fifo.r_en.eq(grant & sel_write),
        ]
        with m.If(grant & both_reqs):
            m.d.sync += rr_toggle.eq(~rr_toggle)

        meta_out = Signal(meta_layout)
        m.d.comb += meta_out.eq(meta_fifo.r_data)

        fwd_hit = Signal()
        fwd_data_sel = Signal(unsigned(wb_bus_data_width))
        fwd_match = Signal()
        fwd_match_idx = Signal(range(fwd_depth))
        fwd_free_found = Signal()
        fwd_free_idx = Signal(range(fwd_depth))
        m.d.comb += [
            fwd_hit.eq(0),
            fwd_data_sel.eq(0),
            fwd_match.eq(0),
            fwd_match_idx.eq(0),
            fwd_free_found.eq(0),
            fwd_free_idx.eq(0),
        ]
        for i in range(fwd_depth):
            with m.If(fwd_valid[i] & (fwd_addr[i] == meta_out.addr)):
                m.d.comb += [
                    fwd_hit.eq(1),
                    fwd_data_sel.eq(fwd_data[i]),
                    fwd_match.eq(1),
                    fwd_match_idx.eq(i),
                ]
            with m.If(~fwd_free_found & ~fwd_valid[i]):
                m.d.comb += [
                    fwd_free_found.eq(1),
                    fwd_free_idx.eq(i),
                ]

        read_data = Signal(unsigned(wb_bus_data_width))
        m.d.comb += read_data.eq(mem.read_resp.payload.data)

        dst_data = Signal(data.ArrayLayout(color_shape, 4))
        plain_dat = [Signal(unsigned(8)) for _ in range(4)]
        effective_read = Signal(unsigned(wb_bus_data_width))
        m.d.comb += effective_read.eq(Mux(fwd_hit, fwd_data_sel, read_data))
        m.d.comb += [
            plain_dat[i].eq(effective_read.word_select(BGRA_MAP[i], 8))
            for i in range(4)
        ]
        for i in range(4):
            m.d.comb += dst_data[i].eq(Cat(plain_dat[i][7], plain_dat[i]))

        src_data = meta_out.src
        src_rgb = src_data[0:3]
        src_a = src_data[3]
        dst_rgb = dst_data[0:3]
        dst_a = dst_data[3]

        factor_src_rgb = Signal(color_shape)
        factor_dst_rgb = Signal(color_shape)
        factor_src_a = Signal(color_shape)
        factor_dst_a = Signal(color_shape)

        def factor_value(factor, src_alpha, dst_alpha):
            ret = Signal(color_shape)
            with m.Switch(factor):
                with m.Case(BlendFactor.ZERO):
                    m.d.comb += ret.eq(0.0)
                with m.Case(BlendFactor.ONE):
                    m.d.comb += ret.eq(one)
                with m.Case(BlendFactor.SRC_COLOR):
                    m.d.comb += Assert(
                        False, "Not implemented: SRC_COLOR factor in blending"
                    )
                with m.Case(BlendFactor.ONE_MINUS_SRC_COLOR):
                    m.d.comb += Assert(
                        False, "Not implemented: ONE_MINUS_SRC_COLOR factor in blending"
                    )
                with m.Case(BlendFactor.DST_COLOR):
                    m.d.comb += Assert(
                        False, "Not implemented: DST_COLOR factor in blending"
                    )
                with m.Case(BlendFactor.ONE_MINUS_DST_COLOR):
                    m.d.comb += Assert(
                        False, "Not implemented: ONE_MINUS_DST_COLOR factor in blending"
                    )
                with m.Case(BlendFactor.SRC_ALPHA):
                    m.d.comb += ret.eq(src_alpha)
                with m.Case(BlendFactor.ONE_MINUS_SRC_ALPHA):
                    m.d.comb += ret.eq(one - src_alpha)
                with m.Case(BlendFactor.DST_ALPHA):
                    m.d.comb += ret.eq(dst_alpha)
                with m.Case(BlendFactor.ONE_MINUS_DST_ALPHA):
                    m.d.comb += ret.eq(one - dst_alpha)
            return ret

        m.d.comb += [
            factor_src_rgb.eq(factor_value(self.conf.src_factor, src_a, dst_a)),
            factor_dst_rgb.eq(factor_value(self.conf.dst_factor, src_a, dst_a)),
            factor_src_a.eq(factor_value(self.conf.src_a_factor, src_a, dst_a)),
            factor_dst_a.eq(factor_value(self.conf.dst_a_factor, src_a, dst_a)),
        ]

        mul_shape = fixed.UQ(0, 18)
        mul_a = Signal(data.ArrayLayout(color_shape, 6))
        mul_b = Signal(data.ArrayLayout(color_shape, 6))
        mul_result = Signal(data.ArrayLayout(mul_shape, 6))
        m.d.comb += [mul_result[i].eq(mul_a[i] * mul_b[i]) for i in range(6)]

        big_shape = fixed.SQ(3, 18)
        blend_out = Signal(data.ArrayLayout(big_shape, 4))

        for i in range(3):
            m.d.comb += [
                mul_a[i].eq(src_rgb[i]),
                mul_b[i].eq(factor_src_rgb),
                mul_a[i + 3].eq(dst_rgb[i]),
                mul_b[i + 3].eq(factor_dst_rgb),
            ]

            src_scaled = mul_result[i]
            dst_scaled = mul_result[i + 3]
            with m.Switch(self.conf.blend_op):
                with m.Case(BlendOp.ADD):
                    m.d.comb += blend_out[i].eq(src_scaled + dst_scaled)
                with m.Case(BlendOp.SUBTRACT):
                    m.d.comb += blend_out[i].eq(src_scaled - dst_scaled)
                with m.Case(BlendOp.REVERSE_SUBTRACT):
                    m.d.comb += blend_out[i].eq(dst_scaled - src_scaled)
                with m.Case(BlendOp.MIN):
                    m.d.comb += blend_out[i].eq(
                        Mux(src_rgb[i] < dst_rgb[i], src_scaled, dst_scaled)
                    )
                with m.Case(BlendOp.MAX):
                    m.d.comb += blend_out[i].eq(
                        Mux(src_rgb[i] > dst_rgb[i], src_scaled, dst_scaled)
                    )

        src_scaled_a = Signal(mul_shape)
        dst_scaled_a = Signal(mul_shape)
        m.d.comb += [
            mul_a[0].eq(src_a),
            mul_b[0].eq(factor_src_a),
            mul_a[3].eq(dst_a),
            mul_b[3].eq(factor_dst_a),
            src_scaled_a.eq(mul_result[0]),
            dst_scaled_a.eq(mul_result[3]),
        ]
        with m.Switch(self.conf.blend_a_op):
            with m.Case(BlendOp.ADD):
                m.d.comb += blend_out[3].eq(src_scaled_a + dst_scaled_a)
            with m.Case(BlendOp.SUBTRACT):
                m.d.comb += blend_out[3].eq(src_scaled_a - dst_scaled_a)
            with m.Case(BlendOp.REVERSE_SUBTRACT):
                m.d.comb += blend_out[3].eq(dst_scaled_a - src_scaled_a)
            with m.Case(BlendOp.MIN):
                m.d.comb += blend_out[3].eq(
                    Mux(src_a < dst_a, src_scaled_a, dst_scaled_a)
                )
            with m.Case(BlendOp.MAX):
                m.d.comb += blend_out[3].eq(
                    Mux(src_a > dst_a, src_scaled_a, dst_scaled_a)
                )

        out_data = Signal(data.ArrayLayout(big_shape, 4))
        for i in range(4):
            m.d.comb += out_data[i].eq(
                Mux(meta_out.need_read, blend_out[i], src_data[i])
            )

        out_data_clamped = Signal(data.ArrayLayout(fixed.UQ(0, 18), 4))
        m.d.comb += [
            out_data_clamped[i].eq(out_data[i].saturate(fixed.UQ(0, 18)))
            for i in range(4)
        ]

        write_mask_swizzled = Signal(unsigned(4))
        m.d.comb += write_mask_swizzled.eq(
            Cat(self.conf.color_write_mask[b] for b in BGRA_MAP)
        )

        ret_v = Signal(data.ArrayLayout(unsigned(8), 4))
        m.d.comb += [
            ret_v[BGRA_MAP[i]].eq(
                ((out_data_clamped[i] << 8) - (out_data_clamped[i] >> 3)).round()
            )
            for i in range(4)
        ]

        write_data = Signal(unsigned(wb_bus_data_width))
        m.d.comb += write_data.eq(Cat(ret_v))

        write_needed = Signal()
        m.d.comb += write_needed.eq(write_mask_swizzled != 0)

        fwd_can_accept = fwd_match | fwd_free_found
        compute_ready = (~write_needed | write_req_fifo.w_rdy) & (
            ~write_needed | fwd_can_accept
        )

        read_needed = meta_out.need_read
        compute_fire = (
            meta_fifo.r_rdy & compute_ready & (~read_needed | mem.read_resp.valid)
        )

        m.d.comb += [
            meta_fifo.r_en.eq(compute_fire),
            mem.read_resp.ready.eq(compute_ready & meta_fifo.r_rdy & read_needed),
        ]

        write_seq = Signal(unsigned(tag_width))
        m.d.comb += write_seq.eq(write_seq_counter)

        write_req_in = Signal(req_layout)
        m.d.comb += [
            write_req_in.addr.eq(meta_out.addr),
            write_req_in.data.eq(write_data),
            write_req_in.sel.eq(write_mask_swizzled),
            write_req_in.we.eq(1),
            write_req_in.tag.eq(write_seq),
            write_req_fifo.w_en.eq(compute_fire & write_needed),
            write_req_fifo.w_data.eq(write_req_in),
        ]

        with m.If(compute_fire):
            for i in range(lock_depth):
                with m.If(lock_valid[i] & (lock_addr[i] == meta_out.addr)):
                    m.d.sync += lock_valid[i].eq(0)

            with m.If(write_needed):
                with m.If(fwd_match):
                    m.d.sync += [
                        fwd_data[fwd_match_idx].eq(write_data),
                        fwd_seq[fwd_match_idx].eq(write_seq),
                    ]
                with m.Else():
                    m.d.sync += [
                        fwd_valid[fwd_free_idx].eq(1),
                        fwd_addr[fwd_free_idx].eq(meta_out.addr),
                        fwd_data[fwd_free_idx].eq(write_data),
                        fwd_seq[fwd_free_idx].eq(write_seq),
                    ]

        with m.If(compute_fire & write_needed):
            m.d.sync += write_seq_counter.eq(write_seq + 1)

        m.d.comb += mem.write_resp.ready.eq(1)
        with m.If(mem.write_resp.valid):
            for i in range(fwd_depth):
                with m.If(fwd_valid[i] & (fwd_seq[i] == mem.write_resp.payload.tag)):
                    m.d.sync += fwd_valid[i].eq(0)

        return m
