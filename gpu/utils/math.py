from amaranth import *
from amaranth.lib import stream, wiring
from amaranth.lib.memory import Memory as Mem
from amaranth.lib.wiring import In, Out

from . import fixed
from .stream import StreamToVector, VectorToStream
from .transactron_utils import sum_value


class CountLeadingZeros(wiring.Component):
    """
    Counts leading zeros in a FixedPoint number.
    """

    def __init__(self, type: Shape):
        super().__init__(
            {
                "i": In(stream.Signature(type)),
                "o": Out(stream.Signature(range(type.width + 1))),
            }
        )

    def elaborate(self, platform) -> Module:
        m = Module()
        width = self.i.p.shape().width

        m.d.comb += [
            self.i.ready.eq(self.o.ready & self.o.valid),
            self.o.valid.eq(self.i.valid),
        ]

        with m.Switch(self.i.p):
            for i in range(width):
                with m.Case("0" * i + "1" + "-" * (width - i - 1)):
                    m.d.comb += self.o.p.eq(i)
            with m.Case("0" * width):
                m.d.comb += self.o.p.eq(width)

        return m


class FixedPointInvSmallDomain(wiring.Component):
    """
    Approximate fixed-point reciprocal using Newton-Raphson method.
    Input value should be in range of [1.0, 2.0).
    Result will be in (0.5, 1.0]
    """

    def __init__(self, type: fixed.Shape, steps: int = 4, initial_guess_bits: int = 4):
        super().__init__(
            {
                "i": In(stream.Signature(type)),
                "o": Out(stream.Signature(type)),
            }
        )
        self._steps = steps
        self._type = type
        self._initial_guess_bits = min(initial_guess_bits, type.f_bits)

    def elaborate(self, platform) -> Module:
        m = Module()

        # Using Newton-Raphson method for reciprocal
        # x_{n+1}=x_n(2−value∗x_n)
        # x_{n+1}=2*x_n - value*x_n*x_n

        # Use upper 4 bits of fractional part for initial guess refinement
        initial_guess_bits = self._initial_guess_bits
        initial_guess_size = 2**initial_guess_bits
        m.submodules.rom = rom = Mem(
            shape=self._type, depth=initial_guess_size, init=[]
        )
        for i in range(initial_guess_size):
            value = 1.0 + (i + 0.5) / initial_guess_size
            rom.init[i] = fixed.Const(1.0 / value, self._type)

        initial_guess = rom.read_port()

        x = Signal(self._type)
        x2 = Signal(self._type)
        vx2 = Signal(self._type)

        mul_a = Signal(self._type)
        mul_b = Signal(self._type)
        mul_result = Signal(self._type)
        m.d.comb += mul_result.eq(mul_a * mul_b)

        p = Signal.like(self.i.p)

        iter = Signal(range(self._steps))

        with m.FSM():
            with m.State("IDLE"):
                m.d.comb += self.i.ready.eq(1)
                m.d.comb += initial_guess.addr.eq(
                    self.i.p.reshape(f_bits=initial_guess_bits)
                    .as_value()
                    .as_unsigned()[:initial_guess_bits]
                )
                with m.If(self.i.valid):
                    m.d.sync += p.eq(self.i.p)
                    m.next = "INITIAL_GUESS"
            with m.State("INITIAL_GUESS"):
                with m.If(p == fixed.Const(1.0)):
                    m.d.comb += [
                        self.o.p.eq(1.0),
                        self.o.valid.eq(1),
                    ]
                    with m.If(self.o.ready):
                        m.next = "IDLE"
                with m.Else():
                    m.d.sync += x.eq(self._type(initial_guess.data))
                    m.d.sync += iter.eq(0)
                    m.next = "STEP_0"
            with m.State("STEP_0"):
                m.d.comb += [
                    mul_a.eq(x),
                    mul_b.eq(x),
                ]
                m.d.sync += [
                    x2.eq(mul_result),
                ]
                m.next = "STEP_1"
            with m.State("STEP_1"):
                m.d.comb += [
                    mul_a.eq(p),
                    mul_b.eq(x2),
                ]
                m.d.sync += [
                    vx2.eq(mul_result),
                ]
                m.next = "STEP_2"
            with m.State("STEP_2"):
                new_x = (x << 1) - vx2
                m.d.sync += [
                    x.eq(new_x),
                    iter.eq(iter + 1),
                ]
                with m.If(iter < self._steps - 1):
                    m.next = "STEP_0"
                with m.Else():
                    m.next = "SEND_RESULT"
            with m.State("SEND_RESULT"):
                m.d.comb += [
                    self.o.p.eq(x),
                    self.o.valid.eq(1),
                ]
                with m.If(self.o.ready):
                    m.next = "IDLE"

        return m


class FixedPointInv(wiring.Component):
    """
    Approximate fixed-point reciprocal using Newton-Raphson method.
    Works for any positive or negative FixedPoint number.

    unsigned input n.m -> output unsigned m.n
    signed   input n.m -> output signed m.n
    """

    def __init__(self, type: fixed.Shape, steps: int = 4):
        if type.signed:
            output_type = fixed.SQ(max(type.f_bits, 2), type.i_bits)
        else:
            output_type = fixed.UQ(max(type.f_bits, 1), type.i_bits)

        super().__init__(
            {
                "i": In(stream.Signature(type)),
                "o": Out(stream.Signature(output_type)),
            }
        )
        self._steps = steps
        self._type = type
        self._output_type = output_type

    def elaborate(self, platform) -> Module:
        m = Module()

        data_bits = self._type.i_bits + self._type.f_bits
        # type should fit all sub computations of N-R method
        small_type = fixed.UQ(2, data_bits - 2)

        m.submodules.inv_small = inv_small = FixedPointInvSmallDomain(
            small_type, steps=self._steps
        )

        u_type = fixed.UQ(self._type.i_bits, self._type.f_bits)
        u_otype = fixed.UQ(self._output_type.i_bits, self._output_type.f_bits)

        v0 = Signal(u_type)
        sgn = Signal()

        lz = Signal(range(data_bits + 1))
        norm_value = Signal(u_otype)

        m.submodules.clz = clz = CountLeadingZeros(self._type.as_shape())

        shift_value = lz - (u_type.i_bits - small_type.i_bits) - 1

        with m.If(inv_small.i.ready):
            m.d.sync += inv_small.i.valid.eq(0)

        with m.If(self.o.ready):
            m.d.sync += self.o.valid.eq(0)

        with m.FSM():
            with m.State("IDLE"):
                with m.If(self.i.valid):
                    m.d.sync += [
                        sgn.eq(self.i.p < 0),
                        v0.eq(abs(self.i.p)),
                        clz.i.p.eq(abs(self.i.p)),
                        clz.i.valid.eq(1),
                    ]
                    m.d.comb += self.i.ready.eq(1)
                    m.next = "CLZ"
            with m.State("CLZ"):
                with m.If(clz.o.valid & (~inv_small.i.valid | inv_small.i.ready)):
                    m.d.comb += clz.o.ready.eq(1)
                    m.d.sync += [
                        lz.eq(clz.o.p),
                        inv_small.i.valid.eq(1),
                    ]

                    shift = clz.o.p - 1

                    with m.If(shift >= 0):
                        m.d.sync += inv_small.i.p.as_value().eq(
                            v0.as_value() << shift.as_unsigned()
                        )
                    with m.Else():
                        m.d.sync += inv_small.i.p.as_value().eq(
                            v0.as_value() >> (-shift).as_unsigned()
                        )

                    m.next = "INV_SMALL"
            with m.State("INV_SMALL"):
                with m.If(inv_small.o.valid & (~self.o.valid | self.o.ready)):
                    # shift back: divide by 2^shift_value
                    m.d.comb += inv_small.o.ready.eq(1)

                    with m.If(shift_value >= 0):
                        m.d.comb += norm_value.eq(
                            inv_small.o.p << shift_value.as_unsigned()
                        )
                    with m.Else():
                        m.d.comb += norm_value.eq(
                            inv_small.o.p >> (-shift_value).as_unsigned()
                        )

                    with m.If(sgn):
                        m.d.sync += self.o.p.eq(-norm_value)
                    with m.Else():
                        m.d.sync += self.o.p.eq(norm_value)

                    m.d.sync += self.o.valid.eq(1)

                    m.next = "IDLE"

        return m


class FixedPointInvSqrtSmallDomain(wiring.Component):
    """
    Fast inverse square root using Newton-Raphson method for FixedPoint numbers.
    Works in domain [1.0, 2).
    The value in should be 1.{pattern}, where {pattern} is the fractional part.
    Result will be (0.7, 1.0]
    """

    def __init__(self, type: fixed.Shape, steps: int = 2, initial_guess_bits: int = 4):
        super().__init__(
            {
                "i": In(stream.Signature(type)),
                "o": Out(stream.Signature(type)),
            }
        )
        self._steps = steps
        self._type = type
        assert not type.signed
        assert type.i_bits > 0
        self._initial_guess_bits = min(initial_guess_bits, type.f_bits)

    def elaborate(self, platform) -> Module:
        m = Module()

        # Using Newton-Raphson method for inverse square root
        # x_{n+1}=x_n(1.5−0.5∗value∗x_n*x_n)
        # x_{n+1}=(x_n + x_n/2) - half_v*x_n*x_n*x_n, where half_v = value/2

        x = Signal(self._type)

        initial_guess = fixed.Const(0.88, self._type)

        initial_guess_bits = self._initial_guess_bits
        initial_guess_size = 2**initial_guess_bits

        m.submodules.rom = rom = Mem(
            shape=self._type, depth=initial_guess_size, init=[]
        )
        for i in range(initial_guess_size):
            value = 1.0 + (i + 0.5) / initial_guess_size
            rom.init[i] = fixed.Const(1.0 / (value**0.5), self._type)
        initial_guess = rom.read_port()

        three_halfs_x = Signal(self._type)
        ax = Signal(self._type)
        x2 = Signal(self._type)

        mul_a = Signal(self._type)
        mul_b = Signal(self._type)
        mul_result = Signal(self._type)
        m.d.comb += mul_result.eq(mul_a * mul_b)

        iter = Signal(range(self._steps))

        p = Signal.like(self.i.p)

        with m.FSM():
            with m.State("IDLE"):
                m.d.comb += self.i.ready.eq(1)
                m.d.comb += initial_guess.addr.eq(
                    self.i.p.reshape(f_bits=initial_guess_bits)
                    .as_value()
                    .as_unsigned()[:initial_guess_bits]
                )
                with m.If(self.i.valid):
                    m.d.sync += p.eq(self.i.p)
                    m.next = "INITIAL_GUESS"
            with m.State("INITIAL_GUESS"):
                with m.If(p == fixed.Const(1.0)):
                    m.d.comb += [
                        self.o.p.eq(1.0),
                        self.o.valid.eq(1),
                    ]
                    with m.If(self.o.ready):
                        m.next = "IDLE"
                with m.Else():
                    m.d.sync += x.eq(self._type(initial_guess.data))
                    m.d.sync += iter.eq(0)
                    m.next = "STEP_0"
            with m.State("STEP_0"):
                m.d.comb += [
                    mul_a.eq(self.i.p >> 1),
                    mul_b.eq(x),
                ]
                m.d.sync += [
                    three_halfs_x.eq(x + (x >> 1)),
                    ax.eq(mul_result),
                ]
                m.next = "STEP_1"
            with m.State("STEP_1"):
                m.d.comb += [
                    mul_a.eq(x),
                    mul_b.eq(x),
                ]
                m.d.sync += [
                    x2.eq(mul_result),
                ]
                m.next = "STEP_2"
            with m.State("STEP_2"):
                m.d.comb += [
                    mul_a.eq(ax),
                    mul_b.eq(x2),
                ]
                new_x = three_halfs_x - mul_result
                m.d.sync += [
                    x.eq(new_x),
                    iter.eq(iter + 1),
                ]
                with m.If(iter < self._steps - 1):
                    m.next = "STEP_0"
                with m.Else():
                    m.d.comb += self.i.ready.eq(1)
                    m.next = "SEND_RESULT"
            with m.State("SEND_RESULT"):
                m.d.comb += [
                    self.o.p.eq(x),
                    self.o.valid.eq(1),
                ]
                with m.If(self.o.ready):
                    m.next = "IDLE"

        return m


class FixedPointInvSqrt(wiring.Component):
    """
    Fast inverse square root using Newton-Raphson method for FixedPoint numbers.
    Works for any positive FixedPoint number.
    """

    def __init__(self, type: fixed.Shape, steps: int = 2):
        super().__init__(
            {
                "i": In(stream.Signature(type)),
                "o": Out(stream.Signature(type)),
            }
        )
        self._steps = steps
        self._type = type

    def elaborate(self, platform) -> Module:
        m = Module()

        data_bits = self._type.i_bits + self._type.f_bits
        small_type = fixed.UQ(1, data_bits - 1)
        m.submodules.inv_sqrt_small = inv_sqrt_small = FixedPointInvSqrtSmallDomain(
            small_type, self._steps
        )
        m.submodules.clz = clz = CountLeadingZeros(self._type.as_shape())

        norm_value = Signal(self._type)
        lz = clz.o.payload
        shift_value = Signal(range(-data_bits, data_bits + 1))
        pre_shift_value = Signal(small_type)

        m.d.comb += [
            shift_value.eq(lz - (self._type.i_bits - small_type.i_bits)),
        ]

        # no pipelining for now
        m.d.comb += [
            clz.o.ready.eq(1),
            inv_sqrt_small.o.ready.eq(1),
        ]

        with m.If(clz.i.ready):
            m.d.sync += clz.i.valid.eq(0)

        with m.If(inv_sqrt_small.i.ready):
            m.d.sync += inv_sqrt_small.i.valid.eq(0)

        with m.If(self.o.ready):
            m.d.sync += self.o.valid.eq(0)

        with m.FSM():
            with m.State("IDLE"):
                with m.If(self.i.valid & (~clz.i.valid | clz.i.ready)):
                    m.d.sync += [
                        clz.i.valid.eq(1),
                        clz.i.p.eq(self.i.p),
                    ]

                    m.next = "CLZ"

            with m.State("CLZ"):
                with m.If(
                    clz.o.valid & (~inv_sqrt_small.i.valid | inv_sqrt_small.i.ready)
                ):
                    m.d.sync += [
                        inv_sqrt_small.i.p.eq(self.i.p.as_value() << lz),
                        inv_sqrt_small.i.valid.eq(1),
                    ]
                    m.d.comb += self.i.ready.eq(1)
                    m.next = "INV_SQRT_SMALL"

            with m.State("INV_SQRT_SMALL"):
                m.d.comb += inv_sqrt_small.o.ready.eq(1)
                with m.If(inv_sqrt_small.o.valid):
                    # shift back: sqrt gets half the normalization shift
                    # divide by 2^floor(shift_value/2)

                    with m.If(shift_value[0] == 1):
                        m.d.sync += pre_shift_value.eq(
                            inv_sqrt_small.o.p * fixed.Const(2**0.5)
                        )
                    with m.Else():
                        m.d.sync += pre_shift_value.eq(inv_sqrt_small.o.p)

                    m.next = "SHIFT_BACK"
            with m.State("SHIFT_BACK"):
                with m.If(self.o.ready | ~self.o.valid):
                    sv_s = shift_value >> 1

                    with m.If(sv_s >= 0):
                        m.d.comb += norm_value.eq(pre_shift_value << sv_s.as_unsigned())
                    with m.Else():
                        m.d.comb += norm_value.eq(
                            pre_shift_value >> (-sv_s).as_unsigned()
                        )

                    m.d.comb += clz.o.ready.eq(1)

                    m.d.sync += [
                        self.o.p.eq(norm_value),
                        self.o.valid.eq(1),
                    ]

                    m.next = "IDLE"

        return m


class SimpleOpModule(wiring.Component):
    def __init__(self, op, type):
        super().__init__(
            {
                "a": In(stream.Signature(type)),
                "b": In(stream.Signature(type)),
                "o": Out(stream.Signature(type)),
            }
        )
        self._op = op
        self._type = type

    def elaborate(self, platform) -> Module:
        m = Module()

        m.d.comb += [
            self.o.valid.eq(self.a.valid & self.b.valid),
            self.o.payload.eq(self._op(self.a.p, self.b.p)),
            self.a.ready.eq(self.o.ready & self.o.valid),
            self.b.ready.eq(self.o.ready & self.o.valid),
        ]

        return m


class FixedPointVecNormalize(wiring.Component):
    def __init__(self, vector_type, steps=2):
        super().__init__(
            {
                "i": In(stream.Signature(vector_type)),
                "o": Out(stream.Signature(vector_type)),
            }
        )
        self._type = vector_type
        self._steps = steps

    def elaborate(self, platform):
        m = Module()

        elem_type = self._type.elem_shape
        unsigned_elem_type = fixed.UQ(elem_type.i_bits, elem_type.f_bits)

        m.submodules.inv_sqrt = inv_sqrt = FixedPointInvSqrt(
            unsigned_elem_type, steps=self._steps
        )

        m.submodules.vec_to_stream_a = v2s_a = VectorToStream(self._type)
        m.submodules.vec_to_stream_b = v2s_b = VectorToStream(self._type)
        m.submodules.stream_to_vec = s2v = StreamToVector(self._type)
        m.submodules.mult = mult = SimpleOpModule(lambda a, b: a * b, elem_type)

        wiring.connect(m, v2s_a.o, mult.a)
        wiring.connect(m, v2s_b.o, mult.b)
        wiring.connect(m, mult.o, s2v.i)

        v = self.i.p

        with m.If(v2s_a.i.ready):
            m.d.sync += v2s_a.i.valid.eq(0)

        with m.If(v2s_b.i.ready):
            m.d.sync += v2s_b.i.valid.eq(0)

        with m.If(inv_sqrt.i.ready):
            m.d.sync += inv_sqrt.i.valid.eq(0)

        with m.If(self.o.ready):
            m.d.sync += self.o.valid.eq(0)

        with m.FSM():
            with m.State("IDLE"):
                with m.If(self.i.valid):
                    m.d.sync += [
                        v2s_a.i.p.eq(self.i.p),
                        v2s_b.i.p.eq(self.i.p),
                        v2s_a.i.valid.eq(1),
                        v2s_b.i.valid.eq(1),
                    ]
                    m.next = "COMPUTE_DOT"
            with m.State("COMPUTE_DOT"):
                with m.If(s2v.o.valid):
                    m.d.comb += s2v.o.ready.eq(1)
                    m.d.sync += [
                        inv_sqrt.i.p.eq(sum_value(*s2v.o.p)),
                        inv_sqrt.i.valid.eq(1),
                    ]
                    m.next = "INV_SQRT"
            with m.State("INV_SQRT"):
                with m.If(inv_sqrt.o.valid):
                    m.d.comb += [
                        inv_sqrt.o.ready.eq(1),
                        self.i.ready.eq(1),
                    ]
                    m.d.sync += [
                        v2s_a.i.p.eq(self.i.p),
                        v2s_b.i.p.eq(Cat([inv_sqrt.o.p for _ in range(len(v))])),
                        v2s_a.i.valid.eq(1),
                        v2s_b.i.valid.eq(1),
                    ]
                    m.next = "MULTIPLY"
            with m.State("MULTIPLY"):
                with m.If(s2v.o.valid & (self.o.ready | ~self.o.valid)):
                    m.d.comb += s2v.o.ready.eq(1)
                    m.d.sync += [
                        self.o.p.eq(s2v.o.payload),
                        self.o.valid.eq(1),
                    ]
                    m.next = "IDLE"

        return m
