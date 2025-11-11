from amaranth import *
from amaranth.lib import data, wiring
from amaranth.lib.wiring import In, Out


class FixedPoint(data.StructLayout):
    """
    Fixed point in format upper.lower
    """

    def __init__(self, upper, lower):
        super().__init__(
            {
                "lower": unsigned(lower),
                "upper": unsigned(upper),
            }
        )

    def __call__(self, value):
        return FixedPointView(self, value)

    def cast_to(self, value: Value):
        lo = self["lower"].width
        hi = self["upper"].width
        width = lo + hi

        if isinstance(value.shape(), FixedPoint):
            v_lo = lower_bits(value)
            v_hi = upper_bits(value)

            if v_lo < lo:
                value = Cat(value.as_value(), C(0, lo - v_lo))
            elif v_lo > lo:
                value = value.as_value()[:width]

            if v_hi < hi:
                value = Cat(C(0, hi - v_hi), value)
            elif v_hi > hi:
                value = value.as_value()[v_lo : v_lo + hi]

            return FixedPointView(self, value)
        else:
            # assume integer
            v_width = value.shape().width
            if v_width < hi:
                value = Cat(value, C(0, hi - v_width))
            elif v_width > hi:
                value = value[:hi]

            return FixedPointView(self, Cat(C(0, lo), value))


def lower_bits(v: Value) -> int:
    if isinstance(v.shape(), FixedPoint):
        return v.shape()["lower"].width
    else:
        raise 0


def upper_bits(v: Value) -> int:
    if isinstance(v.shape(), FixedPoint):
        return v.shape()["upper"].width
    else:
        raise v.shape().width


class FixedPointView(data.View):
    def cast_like(self, o: Value) -> "FixedPointView":
        return self.shape().cast_to(o)

    def make_common(a: Value, b: Value) -> tuple["FixedPointView", "FixedPointView"]:
        a_lower = lower_bits(a)
        b_lower = lower_bits(b)
        common_lower = max(a_lower, b_lower)

        a_upper = upper_bits(a)
        b_upper = upper_bits(b)
        common_upper = max(a_upper, b_upper)

        common_shape = FixedPoint(common_upper, common_lower)
        return common_shape.cast_to(a), common_shape.cast_to(b)

    def __add__(self, o: Value) -> "FixedPointView":
        o_casted = self.cast_like(o)
        v1, v2 = FixedPointView.make_common(self, o_casted)
        new_shape = FixedPoint(v1.shape()["upper"].width + 1, v1.shape()["lower"].width)
        return FixedPointView(new_shape, v1.as_value() + v2.as_value())
