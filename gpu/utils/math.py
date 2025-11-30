from amaranth import *
from amaranth.lib import stream, wiring
from amaranth.lib.wiring import In, Out

from . import fixed


class VectorToStream(wiring.Component):
    """
    Converts a vector signal into a stream signal.
    """

    def __init__(self, vector_type):
        super().__init__(
            {
                "i": In(stream.Signature(vector_type)),
                "o": Out(stream.Signature(vector_type.elem_shape)),
            }
        )

    def elaborate(self, platform) -> Module:
        m = Module()

        num_elem = self.i.p.shape().length

        def next_state(i):
            if i + 1 < num_elem:
                return f"SEND_{i + 1}"
            else:
                return "SEND_0"

        with m.If(self.o.ready):
            m.d.sync += self.o.valid.eq(0)

        with m.FSM():
            with m.State("SEND_0"):
                with m.If(self.i.valid & (self.o.ready | ~self.o.valid)):
                    m.d.sync += [
                        self.o.p.eq(self.i.p[0]),
                        self.o.valid.eq(1),
                    ]
                    m.next = next_state(0)
            for i in range(1, num_elem):
                with m.State(f"SEND_{i}"):
                    with m.If(self.o.ready):
                        m.d.sync += [
                            self.o.p.eq(self.i.p[i]),
                            self.o.valid.eq(1),
                        ]
                        m.next = next_state(i)

                        if i + 1 == num_elem:
                            m.d.comb += self.i.ready.eq(1)

        return m


class StreamToVector(wiring.Component):
    """
    Converts a stream signal into a vector signal.
    """

    def __init__(self, vector_type):
        super().__init__(
            {
                "i": In(stream.Signature(vector_type.elem_shape)),
                "o": Out(stream.Signature(vector_type)),
            }
        )

    def elaborate(self, platform) -> Module:
        m = Module()

        num_elem = self.o.p.shape().length
        received = Array(Signal.like(self.i.p) for _ in range(num_elem - 1))

        with m.If(self.o.ready):
            m.d.sync += self.o.valid.eq(0)

        with m.FSM():
            for i in range(0, num_elem - 1):
                with m.State(f"RECEIVE_{i}"):
                    m.d.comb += self.i.ready.eq(1)

                    with m.If(self.i.valid):
                        m.d.sync += received[i].eq(self.i.p)
                        m.next = f"RECEIVE_{i + 1}"
            with m.State(f"RECEIVE_{num_elem - 1}"):
                with m.If(self.o.ready | ~self.o.valid):
                    m.d.comb += self.i.ready.eq(1)

                    with m.If(self.i.valid):
                        m.d.sync += [
                            self.o.p.eq(Cat(received, self.i.p)),
                            self.o.valid.eq(1),
                        ]
                        m.d.comb += self.i.ready.eq(1)
                        m.next = "RECEIVE_0"

        return m


class FixedPointInvSqrtSmallDomain(wiring.Component):
    """
    Fast inverse square root using Newton-Raphson method for FixedPoint numbers.
    Works in domain [1.0, 2).
    The value in should be 1.{pattern}, where {pattern} is the fractional part.
    Result will be (0.7, 1.0]
    """

    def __init__(self, type: fixed.Shape, steps: int = 2):
        super().__init__(
            {
                "i": In(stream.Signature(type)),
                "o": Out(stream.Signature(type)),
            }
        )
        self.steps = steps
        self.type = type
        assert not type.signed
        assert type.i_bits > 0

    def elaborate(self, platform) -> Module:
        m = Module()

        # Using Newton-Raphson method for inverse square root
        # x_{n+1}=x_n(1.5−0.5∗value∗x_n*x_n)
        # x_{n+1}=(x_n + x_n/2) - half_v*x_n*x_n*x_n, where half_v = value/2

        x = Signal(self.type)
        half_v = Signal(self.type)

        three_halfs_x = Signal(self.type)
        ax = Signal(self.type)
        x2 = Signal(self.type)

        mul_a = Signal(self.type)
        mul_b = Signal(self.type)
        mul_result = Signal(self.type)
        m.d.comb += mul_result.eq(mul_a * mul_b)

        with m.FSM():
            with m.State("IDLE"):
                with m.If(self.i.valid):
                    m.d.comb += self.i.ready.eq(1)
                    m.d.sync += [
                        half_v.eq(self.i.p >> 1),
                        x.eq(0.88),  # Initial guess
                    ]
                    m.next = "ITERATE_0_STEP_0"
            for i in range(self.steps):
                with m.State(f"ITERATE_{i}_STEP_0"):
                    m.d.comb += [
                        mul_a.eq(half_v),
                        mul_b.eq(x),
                    ]
                    m.d.sync += [
                        three_halfs_x.eq(x + (x >> 1)),
                        ax.eq(mul_result),
                    ]
                    m.next = f"ITERATE_{i}_STEP_1"
                with m.State(f"ITERATE_{i}_STEP_1"):
                    m.d.comb += [
                        mul_a.eq(x),
                        mul_b.eq(x),
                    ]
                    m.d.sync += [
                        x2.eq(mul_result),
                    ]
                    m.next = f"ITERATE_{i}_STEP_2"
                with m.State(f"ITERATE_{i}_STEP_2"):
                    m.d.comb += [
                        mul_a.eq(ax),
                        mul_b.eq(x2),
                    ]
                    new_x = three_halfs_x - mul_result
                    m.d.sync += [x.eq(new_x)]
                    if i == self.steps - 1:
                        m.next = "SEND_RESULT"
                    else:
                        m.next = f"ITERATE_{i + 1}_STEP_0"
            with m.State("SEND_RESULT"):
                with m.If(self.o.ready | ~self.o.valid):
                    m.d.sync += [
                        self.o.p.eq(x),
                        self.o.valid.eq(1),
                    ]
                    m.next = "IDLE"

        return m


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
        self.steps = steps
        self.type = type

    def elaborate(self, platform) -> Module:
        m = Module()

        data_bits = self.type.i_bits + self.type.f_bits
        small_type = fixed.UQ(1, data_bits - 1)
        m.submodules.inv_sqrt_small = inv_sqrt_small = FixedPointInvSqrtSmallDomain(
            small_type, self.steps
        )
        m.submodules.clz = clz = CountLeadingZeros(self.type.as_shape())

        norm_value = Signal(self.type)
        lz = clz.o.payload
        shift_value = Signal(range(-data_bits, data_bits + 1))
        pre_shift_value = Signal(small_type)

        m.d.comb += [
            shift_value.eq(lz - (self.type.i_bits - small_type.i_bits)),
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
        self.op = op
        self.type = type

    def elaborate(self, platform) -> Module:
        m = Module()

        m.d.comb += [
            self.o.valid.eq(self.a.valid & self.b.valid),
            self.o.payload.eq(self.op(self.a.p, self.b.p)),
            self.a.ready.eq(self.o.ready & self.o.valid),
            self.b.ready.eq(self.o.ready & self.o.valid),
        ]

        return m


class FixedPointVecNormalize(wiring.Component):
    def __init__(self, vector_type, inv_sqrt_steps=2):
        super().__init__(
            {
                "i": In(stream.Signature(vector_type)),
                "o": Out(stream.Signature(vector_type)),
            }
        )
        self.vector_type = vector_type
        self.inv_sqrt_steps = inv_sqrt_steps

    def elaborate(self, platform):
        m = Module()

        elem_type = self.vector_type.elem_shape
        unsiged_elem_type = fixed.UQ(elem_type.i_bits, elem_type.f_bits)

        m.submodules.inv_sqrt = inv_sqrt = FixedPointInvSqrt(
            unsiged_elem_type, steps=self.inv_sqrt_steps
        )

        m.submodules.vec_to_stream_a = v2s_a = VectorToStream(self.vector_type)
        m.submodules.vec_to_stream_b = v2s_b = VectorToStream(self.vector_type)
        m.submodules.stream_to_vec = s2v = StreamToVector(self.vector_type)
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
                        inv_sqrt.i.p.eq(sum(s2v.o.p, start=fixed.Const(0.0))),
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
