import warnings
from math import ceil, log2

from amaranth import *
from amaranth.lib import data, fifo, stream, wiring
from amaranth.lib.memory import Memory as Mem
from amaranth.lib.wiring import In, Out

from . import fixed
from .stream import StreamToVector, VectorToStream
from .transactron_utils import count_leading_zeros, sum_value


class FixedPointInvSmallDomain(wiring.Component):
    """
    Approximate fixed-point reciprocal using Newton-Raphson method.
    Input value should be in range of [1.0, 2.0).
    Result will be in (0.5, 1.0]

    non-pipelined version uses 1 multiplicator
    pipelined version uses 2 * steps multiplicators

    each step doubles the number of precission digits
    """

    class InitialGuess(wiring.Component):
        """
        Initial guess for Newton-Raphson method.
        """

        i: stream.Interface
        o: stream.Interface

        def __init__(self, type: fixed.Shape, bits: int = 8):
            sig_in = stream.Signature(
                data.StructLayout(
                    {
                        "value": type,
                    }
                )
            )
            sig_out = stream.Signature(
                data.StructLayout(
                    {
                        "value": type,
                        "x": type,
                    }
                )
            )
            super().__init__(
                {
                    "i": In(sig_in),
                    "o": Out(sig_out),
                }
            )
            self._type = type
            self._bits = bits

        def elaborate(self, platform) -> Module:
            m = Module()

            initial_guess_size = 2**self._bits
            m.submodules.rom = rom = Mem(
                shape=self._type, depth=initial_guess_size, init=[]
            )
            for i in range(initial_guess_size):
                value = 1.0 + (i + 0.5) / initial_guess_size
                rom.init[i] = fixed.Const(1.0 / value, self._type)

            port = rom.read_port()
            m.d.comb += self.o.p.x.eq(port.data)

            m.d.comb += port.en.eq(0)

            with m.If(self.i.valid & (~self.o.valid | self.o.ready)):
                m.d.comb += self.i.ready.eq(1)
                m.d.comb += port.en.eq(1)
                m.d.comb += port.addr.eq(
                    self.i.p.value.reshape(f_bits=self._bits)
                    .as_value()
                    .as_unsigned()[: self._bits]
                )
                m.d.sync += self.o.valid.eq(1)
                m.d.sync += self.o.p.value.eq(self.i.p.value)
            with m.Elif(self.o.ready):
                m.d.sync += self.o.valid.eq(0)

            return m

    class NRStepPipelined(wiring.Component):
        """
        Pipelined version of Newton-Raphson method for reciprocal.
        """

        i: stream.Interface
        o: stream.Interface

        def __init__(self, type: fixed.Shape):
            sig = stream.Signature(
                data.StructLayout(
                    {
                        "value": type,
                        "x": type,
                    }
                )
            )
            super().__init__(
                {
                    "i": In(sig),
                    "o": Out(sig),
                }
            )
            self._type = type

        def elaborate(self, platform) -> Module:
            m = Module()

            # Using Newton-Raphson method for reciprocal
            # x_{n+1}=x_n(2−value∗x_n)
            # x_{n+1}=2x - value * x ^ 2

            internal = stream.Signature(
                data.StructLayout(
                    {
                        "value": self._type,
                        "x": self._type,
                        "x_squared": self._type,
                    }
                )
            ).create()

            with m.If(self.i.valid & (~internal.valid | internal.ready)):
                m.d.sync += internal.p.value.eq(self.i.p.value)
                m.d.sync += internal.p.x.eq(self.i.p.x)
                m.d.sync += internal.p.x_squared.eq(self.i.p.x * self.i.p.x)
                m.d.comb += self.i.ready.eq(1)
                m.d.sync += internal.valid.eq(1)
            with m.Elif(internal.ready):
                m.d.sync += internal.valid.eq(0)

            with m.If(internal.valid & (~self.o.valid | self.o.ready)):
                m.d.sync += self.o.p.value.eq(internal.p.value)
                x = internal.p.x
                x_sqr = internal.p.x_squared
                m.d.sync += self.o.p.x.eq((x << 1) - internal.p.value * x_sqr)
                m.d.comb += internal.ready.eq(1)
                m.d.sync += self.o.valid.eq(1)
            with m.Elif(self.o.ready):
                m.d.sync += self.o.valid.eq(0)

            return m

    i: stream.Interface
    o: stream.Interface

    def __init__(
        self,
        type: fixed.Shape,
        steps: int = 2,
        initial_guess_bits: int = 8,
        pipelined=False,
    ):
        assert not type.signed
        super().__init__(
            {
                "i": In(stream.Signature(type)),
                "o": Out(stream.Signature(type)),
            }
        )
        self._steps = steps
        self._type = type
        self._initial_guess_bits = min(initial_guess_bits, type.f_bits)
        self._pipelined = pipelined

    def elaborate(self, platform) -> Module:
        m = Module()

        # Using Newton-Raphson method for reciprocal
        # x_{n+1}=x_n(2−value∗x_n)
        # x_{n+1}=2*x_n - value*x_n*x_n

        m.submodules.initial_guess = initial_guess = self.InitialGuess(
            self._type, self._initial_guess_bits
        )
        m.d.comb += initial_guess.i.p.value.eq(self.i.p)
        m.d.comb += initial_guess.i.valid.eq(self.i.valid)
        m.d.comb += self.i.ready.eq(initial_guess.i.ready)

        if self._pipelined:
            prev = initial_guess
            for i in range(self._steps):
                m.submodules[f"nr_step_{i}"] = step = self.NRStepPipelined(self._type)
                wiring.connect(m, prev.o, step.i)
                prev = step
            m.d.comb += self.o.p.eq(prev.o.p.x)
            m.d.comb += self.o.valid.eq(prev.o.valid)
            m.d.comb += prev.o.ready.eq(self.o.ready)
            return m

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
                m.d.comb += initial_guess.o.ready.eq(1)
                with m.If(initial_guess.o.valid):
                    m.d.sync += p.eq(initial_guess.o.p.value)
                    m.d.sync += x.eq(initial_guess.o.p.x)
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

    i: stream.Interface
    o: stream.Interface

    def __init__(
        self,
        type: fixed.Shape,
        steps: int | None = None,
        pipelined: bool = False,
        initial_guess_bits: int = 8,
    ):
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

        total_output_bits = output_type.i_bits + output_type.f_bits
        suggested_steps = int(ceil(log2(total_output_bits / initial_guess_bits)))

        if steps is None:
            steps = suggested_steps

        if steps > suggested_steps:
            # The user can reduce the number of steps without losing precision
            warnings.warn(
                f"Fixed point inversion requested with {steps} steps, "
                f"but {suggested_steps} steps sufficient for {type!r}."
            )

        self._steps = steps
        self._type = type
        self._output_type = output_type
        self._pipelined = pipelined
        self._initial_guess_bits = initial_guess_bits

    def elaborate(self, platform) -> Module:
        m = Module()

        data_bits = self._type.i_bits + self._type.f_bits
        # type should fit all sub computations of N-R method
        small_type = fixed.UQ(2, data_bits - 2)

        m.submodules.inv_small = inv_small = FixedPointInvSmallDomain(
            small_type,
            steps=self._steps,
            pipelined=self._pipelined,
            initial_guess_bits=self._initial_guess_bits,
        )

        u_type = fixed.UQ(self._type.i_bits, self._type.f_bits)
        u_otype = fixed.UQ(self._output_type.i_bits, self._output_type.f_bits)

        lz = Signal(range(data_bits + 1))

        clz_payload = data.StructLayout(
            {
                "sgn": unsigned(1),
                "abs": u_type,
            }
        )
        clz_stream = stream.Signature(clz_payload).create()

        inv_payload = data.StructLayout(
            {
                "lz": lz.shape(),
                "sgn": unsigned(1),
            }
        )
        m.submodules.inv_small_fifo = inv_small_fifo = fifo.SyncFIFOBuffered(
            width=inv_payload.as_shape().width,
            depth=self._steps * 2 + 2 if self._pipelined else 1,
        )

        with m.If(self.i.valid & (~clz_stream.valid | clz_stream.ready)):
            m.d.comb += self.i.ready.eq(1)
            m.d.sync += [
                clz_stream.valid.eq(1),
                clz_stream.p.abs.eq(abs(self.i.p)),
                clz_stream.p.sgn.eq(self.i.p < 0),
            ]
        with m.Elif(clz_stream.ready):
            m.d.sync += clz_stream.valid.eq(0)

        # inv pre shifter
        abs_v = clz_stream.p.abs.as_value()
        clz = count_leading_zeros(abs_v)
        shift = clz - 1

        inv_data = Signal.like(inv_small.i.p)

        with m.If(shift >= 0):
            m.d.comb += inv_data.as_value().eq(abs_v << shift.as_unsigned())
        with m.Else():
            m.d.comb += inv_data.as_value().eq(abs_v >> 1)

        inv_small_w_data = Signal(inv_payload)
        m.d.comb += inv_small_w_data.lz.eq(clz)
        m.d.comb += inv_small_w_data.sgn.eq(clz_stream.p.sgn)

        m.d.comb += inv_small_fifo.w_data.eq(inv_small_w_data)
        with m.If(
            clz_stream.valid
            & inv_small_fifo.w_rdy
            & (~inv_small.i.valid | inv_small.i.ready)
        ):
            m.d.comb += inv_small_fifo.w_en.eq(1)
            m.d.comb += clz_stream.ready.eq(1)
            m.d.sync += inv_small.i.valid.eq(1)
            m.d.sync += inv_small.i.p.eq(inv_data)
        with m.Elif(inv_small.i.ready):
            m.d.sync += inv_small.i.valid.eq(0)

        # result preparation
        inv_small_r_data = Signal(inv_payload)
        m.d.comb += inv_small_r_data.eq(inv_small_fifo.r_data)
        lz = inv_small_r_data.lz

        shift_value = lz - (u_type.i_bits - small_type.i_bits + 1)

        norm_value = Signal(u_otype)
        with m.If(shift_value >= 0):
            m.d.comb += norm_value.eq(inv_small.o.p << shift_value.as_unsigned())
        with m.Else():
            m.d.comb += norm_value.eq(inv_small.o.p >> (-shift_value).as_unsigned())

        out_v = Signal.like(self.o.p)

        with m.If(inv_small_r_data.sgn):
            m.d.comb += out_v.eq(-norm_value)
        with m.Else():
            m.d.comb += out_v.eq(norm_value)

        with m.If(inv_small.o.valid & (~self.o.valid | self.o.ready)):
            m.d.sync += self.o.valid.eq(1)
            m.d.sync += self.o.p.eq(out_v)
            m.d.comb += inv_small_fifo.r_en.eq(1)
            m.d.comb += inv_small.o.ready.eq(1)
        with m.Elif(self.o.ready):
            m.d.sync += self.o.valid.eq(0)

        return m


class FixedPointInvSqrtSmallDomain(wiring.Component):
    """
    Fast inverse square root using Newton-Raphson method for FixedPoint numbers.
    Works in domain [1.0, 2).
    The value in should be 1.{pattern}, where {pattern} is the fractional part.
    Result will be (0.7, 1.0]
    """

    i: stream.Interface
    o: stream.Interface

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

    i: stream.Interface
    o: stream.Interface

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

        norm_value = Signal(self._type)
        shift_value = Signal(range(-data_bits, data_bits + 1))
        pre_shift_value = Signal(small_type)

        clz_in = Signal(self._type)
        clz = count_leading_zeros(clz_in.as_value())
        lz = Signal.like(clz)
        m.d.comb += [
            shift_value.eq(lz - (self._type.i_bits - small_type.i_bits)),
        ]

        # no pipelining for now
        m.d.comb += inv_sqrt_small.o.ready.eq(1)

        with m.If(inv_sqrt_small.i.ready):
            m.d.sync += inv_sqrt_small.i.valid.eq(0)

        with m.If(self.o.ready):
            m.d.sync += self.o.valid.eq(0)

        with m.FSM():
            with m.State("IDLE"):
                m.d.comb += clz_in.eq(self.i.p)
                with m.If(self.i.valid):
                    m.d.sync += lz.eq(clz)
                    m.next = "CLZ"

            with (
                m.State("CLZ"),
                m.If(~inv_sqrt_small.i.valid | inv_sqrt_small.i.ready),
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
            with m.State("SHIFT_BACK"), m.If(self.o.ready | ~self.o.valid):
                sv_s = shift_value >> 1

                with m.If(sv_s >= 0):
                    m.d.comb += norm_value.eq(pre_shift_value << sv_s.as_unsigned())
                with m.Else():
                    m.d.comb += norm_value.eq(pre_shift_value >> (-sv_s).as_unsigned())

                m.d.sync += [
                    self.o.p.eq(norm_value),
                    self.o.valid.eq(1),
                ]

                m.next = "IDLE"

        return m


class SimpleOpModule(wiring.Component):
    a: stream.Interface
    b: stream.Interface
    o: stream.Interface

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
    i: stream.Interface
    o: stream.Interface

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
            with m.State("IDLE"), m.If(self.i.valid):
                m.d.sync += [
                    v2s_a.i.p.eq(self.i.p),
                    v2s_b.i.p.eq(self.i.p),
                    v2s_a.i.valid.eq(1),
                    v2s_b.i.valid.eq(1),
                ]
                m.next = "COMPUTE_DOT"
            with m.State("COMPUTE_DOT"), m.If(s2v.o.valid):
                m.d.comb += s2v.o.ready.eq(1)
                m.d.sync += [
                    inv_sqrt.i.p.eq(sum_value(*s2v.o.p)),
                    inv_sqrt.i.valid.eq(1),
                ]
                m.next = "INV_SQRT"
            with m.State("INV_SQRT"), m.If(inv_sqrt.o.valid):
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
