from amaranth import *
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out

from .types import FixedPointLayout


class VectorizedOperation(wiring.Component):
    """
    Multi-cycle vector operation component.
    """

    def __init__(
        self, operation, input_type_left, input_type_right=None, output_type=None
    ):
        if input_type_right is None:
            input_type_right = input_type_left
        if output_type is None:
            output_type = input_type_left

        super().__init__(
            {
                "a": In(input_type_left),
                "b": In(input_type_right),
                "result": Out(output_type),
                "start": In(1),
                "ready": Out(1),
            }
        )
        self.operation = operation

    def elaborate(self, platform) -> Module:
        m = Module()

        a_v = Signal.like(self.a)
        b_v = Signal.like(self.b)

        m.submodules.op = op = self.operation()

        with m.FSM():
            with m.State("IDLE"):
                m.d.comb += self.ready.eq(1)
                with m.If(self.start):
                    m.d.sync += [
                        a_v.eq(self.a),
                        b_v.eq(self.b),
                    ]
                    m.next = "STEP_0"
            for cycle in range(len(self.a)):
                with m.State(f"STEP_{cycle}"):
                    m.d.comb += [
                        op.a.eq(a_v[cycle]),
                        op.b.eq(b_v[cycle]),
                    ]
                    m.d.sync += self.result[cycle].eq(op.result)

                    m.next = "IDLE" if cycle == len(self.a) - 1 else f"STEP_{cycle + 1}"

        return m


class VectorizedOperationMC(wiring.Component):
    """
    Multi-cycle vector operation component for multi-cycle operations.
    """

    def __init__(
        self, operation, input_type_left, input_type_right=None, output_type=None
    ):
        if input_type_right is None:
            input_type_right = input_type_left
        if output_type is None:
            output_type = input_type_left

        super().__init__(
            {
                "a": In(input_type_left),
                "b": In(input_type_right),
                "result": Out(output_type),
                "start": In(1),
                "ready": Out(1),
            }
        )
        self.operation = operation

    def elaborate(self, platform) -> Module:
        m = Module()

        a_v = Signal.like(self.a)
        b_v = Signal.like(self.b)

        m.submodules.op = op = self.operation()

        with m.FSM():
            with m.State("IDLE"):
                m.d.comb += self.ready.eq(1)
                with m.If(self.start):
                    m.d.sync += [
                        a_v.eq(self.a),
                        b_v.eq(self.b),
                    ]
                    m.d.comb += [
                        op.a.eq(self.a[0]),
                        op.b.eq(self.b[0]),
                        op.start.eq(1),
                    ]
                    m.next = "STEP_0"
            for cycle in range(len(self.a)):
                with m.State(f"STEP_{cycle}"):
                    with m.If(op.ready):
                        m.d.sync += self.result[cycle].eq(op.result)

                        if cycle < len(self.a) - 1:
                            m.d.comb += [
                                op.a.eq(a_v[cycle + 1]),
                                op.b.eq(b_v[cycle + 1]),
                                op.start.eq(1),
                            ]
                            m.next = f"STEP_{cycle + 1}"
                        else:
                            m.next = "IDLE"

        return m


def count_leading_zeros(value: Value) -> Value:
    width = len(value)

    ret = C(width)
    for i in range(width):
        ret = Mux(value[i] == 0, ret, C(width - i - 1))

    return ret


class FixedPointInvSqrtSmallDomain(wiring.Component):
    """
    Fast inverse square root using Newton-Raphson method for FixedPoint numbers.
    Works in domain [1.0, 2).
    The value in should be 1.{pattern}, where {pattern} is the fractional part.
    """

    def __init__(self, type: FixedPointLayout, steps: int = 2):
        super().__init__(
            {
                "value": In(type),
                "result": Out(type),
                "start": In(1),
                "ready": Out(1),
            }
        )
        self.steps = steps
        self.type = type

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
                m.d.comb += [self.ready.eq(1), self.result.eq(x)]
                with m.If(self.start):
                    m.d.sync += [
                        # Assert(self.value.int.as_unsigned() > 0),
                        half_v.data.eq(self.value.data.as_unsigned() >> 1),
                        x.eq(self.type.from_float(0.88)),
                    ]
                    m.next = "ITERATE_0_STEP_0"
            for i in range(self.steps):
                with m.State(f"ITERATE_{i}_STEP_0"):
                    m.d.comb += [
                        mul_a.eq(half_v),
                        mul_b.eq(x),
                    ]
                    m.d.sync += [
                        three_halfs_x.eq(x + self.type(x.data.as_unsigned() >> 1)),
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
                        m.next = "IDLE"
                    else:
                        with m.If(x == new_x):
                            m.next = "IDLE"
                        with m.Else():
                            m.next = f"ITERATE_{i + 1}_STEP_0"

        return m


class FixedPointInvSqrt(wiring.Component):
    """
    Fast inverse square root using Newton-Raphson method for FixedPoint numbers.
    Works for any positive FixedPoint number.
    """

    def __init__(self, type: FixedPointLayout, steps: int = 2):
        super().__init__(
            {
                "value": In(type),
                "result": Out(type),
                "start": In(1),
                "ready": Out(1),
            }
        )
        self.steps = steps
        self.type = type

    def elaborate(self, platform) -> Module:
        m = Module()

        data_bits = len(self.value.data)
        small_type = FixedPointLayout(data_bits - 2, 2)
        m.submodules.inv_sqrt_small = inv_sqrt_small = FixedPointInvSqrtSmallDomain(
            small_type, self.steps
        )

        v = Signal(self.type)
        norm_value = Signal(self.type)
        lz = Signal(range(data_bits))
        shift_value = Signal(range(-data_bits, data_bits + 1))
        inv_sqrt_data_in = Signal(small_type)
        inv_sqrt_data_out = Signal(small_type)
        pre_shift_value = Signal(small_type)

        m.d.comb += [
            inv_sqrt_small.value.eq(inv_sqrt_data_in),
            inv_sqrt_data_out.eq(inv_sqrt_small.result),
            shift_value.eq(lz - (self.type.hi_bits - small_type.hi_bits) - 1),
        ]

        def do_shift_by(value: Value, shift: Value) -> Value:
            """Shift value by shift (can be negative)."""
            return Mux(
                shift >= 0,
                (value << shift.as_unsigned())[: len(value)],
                value >> (-shift).as_unsigned(),
            )

        with m.FSM():
            with m.State("IDLE"):
                m.d.comb += [self.ready.eq(1), self.result.eq(norm_value)]
                with m.If(self.start):
                    with m.If(self.value.data <= 0):
                        m.d.sync += [
                            # Return 0 for non-positive inputs
                            norm_value.eq(self.type.from_float(0.0)),
                        ]
                        m.next = "IDLE"
                    with m.Else():
                        m.d.sync += v.eq(self.value)
                        m.next = "CLZ"
            with m.State("CLZ"):
                m.d.sync += [
                    lz.eq(count_leading_zeros(v.data.as_unsigned())),
                ]
                m.next = "NORMALIZE"
            with m.State("NORMALIZE"):
                m.d.comb += [
                    inv_sqrt_data_in.eq(do_shift_by(v.data, lz - 1)),
                    inv_sqrt_small.start.eq(1),
                ]
                m.next = "INV_SQRT_SMALL"
            with m.State("INV_SQRT_SMALL"):
                with m.If(inv_sqrt_small.ready):
                    # shift back: sqrt gets half the normalization shift
                    # divide by 2^floor(shift_value/2)
                    m.d.sync += pre_shift_value.eq(inv_sqrt_data_out)
                    with m.If(shift_value[0] == 1):
                        m.next = "POST_MULT"
                    with m.Else():
                        m.next = "SHIFT_BACK"
            with m.State("POST_MULT"):
                m.d.sync += [
                    pre_shift_value.eq(pre_shift_value * small_type.from_float(2**0.5)),
                ]
                m.next = "SHIFT_BACK"
            with m.State("SHIFT_BACK"):
                shift_offset = self.type.lo_bits - small_type.lo_bits
                m.d.sync += norm_value.eq(
                    do_shift_by(
                        pre_shift_value.data.as_unsigned(),
                        shift_value[1:].as_signed() + shift_offset,
                    )
                )
                m.next = "IDLE"

        return m


class SimpleOpModule(wiring.Component):
    def __init__(self, op, type):
        super().__init__(
            {
                "a": In(type),
                "b": In(type),
                "result": Out(type),
            }
        )
        self.op = op
        self.type = type

    def elaborate(self, platform) -> Module:
        m = Module()
        m.d.comb += self.result.eq(self.op(self.a, self.b))
        return m


class FixedPointVecNormalize(wiring.Component):
    def __init__(self, vector_type, inv_sqrt_steps=2):
        super().__init__(
            {
                "value": In(vector_type),
                "result": Out(vector_type),
                "start": In(1),
                "ready": Out(1),
            }
        )
        self.vector_type = vector_type
        self.inv_sqrt_steps = inv_sqrt_steps

    def elaborate(self, platform):
        m = Module()

        elem_type = self.vector_type.elem_shape

        m.submodules.inv_sqrt = inv_sqrt = FixedPointInvSqrt(
            elem_type, steps=self.inv_sqrt_steps
        )
        m.submodules.mult = mult = VectorizedOperation(
            lambda: SimpleOpModule(lambda a, b: a * b, elem_type), self.vector_type
        )

        v = Signal.like(self.value)
        dot_v = Signal.like(v[0])
        inv_len_v = Signal.like(v[0])

        m.d.comb += [
            dot_v.eq(sum(mult.result, start=elem_type.from_float(0.0))),
            inv_len_v.eq(inv_sqrt.result),
        ]

        with m.FSM():
            with m.State("IDLE"):
                m.d.comb += [self.ready.eq(1), self.result.eq(mult.result)]
                with m.If(self.start):
                    m.d.sync += v.eq(self.value)
                    m.d.comb += [
                        mult.a.eq(self.value),
                        mult.b.eq(self.value),
                        mult.start.eq(1),
                    ]
                    m.next = "COMPUTE_DOT"
            with m.State("COMPUTE_DOT"):
                with m.If(mult.ready):
                    m.d.comb += [
                        inv_sqrt.value.eq(dot_v),
                        inv_sqrt.start.eq(1),
                    ]
                    m.next = "INV_SQRT"
            with m.State("INV_SQRT"):
                with m.If(inv_sqrt.ready):
                    m.d.comb += [
                        mult.a.eq(v),
                        mult.b.eq(Cat([inv_len_v for _ in range(len(v))])),
                        mult.start.eq(1),
                    ]
                    m.next = "MULTIPLY"
            with m.State("MULTIPLY"):
                with m.If(mult.ready):
                    m.d.comb += [self.ready.eq(1), self.result.eq(mult.result)]
                    m.next = "IDLE"

        return m
