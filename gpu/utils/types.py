from math import ceil, log10

from amaranth import *
from amaranth.lib import data


class FixedPointLayout(data.StructLayout):
    def __init__(self, lo_bits: int = 16, hi_bits: int = 16):
        super().__init__({"data": signed(lo_bits + hi_bits)})
        self.lo_bits = lo_bits
        self.hi_bits = hi_bits
        self.total_bits = lo_bits + hi_bits

    def __call__(self, value: Value) -> "FixedPointView":
        return FixedPointView(self, value)

    def from_float(self, value: float) -> "FixedPointView":
        return self(self.from_float_const(value))

    def from_float_const(self, value: float) -> data.Const:
        fixed_value = int(value * (1 << self.lo_bits))
        return self.const({"data": fixed_value})

    def change_radix(self, value: "FixedPointView") -> "FixedPointView":
        # Cast a FixedPointView to this layout (align the integer bits)
        shift = self.lo_bits - value.shape().lo_bits
        if self.lo_bits >= value.shape().lo_bits:
            new_data = value.data.shift_left(shift)
        else:
            new_data = value.data.shift_right(-shift)

        return self(Cat(new_data, C(0, self.total_bits - len(new_data))))


class FixedPointView(data.View):
    @property
    def fract(self):
        return self.data[: self.shape().lo_bits].as_unsigned()

    @property
    def int(self):
        return self.data[-self.shape().hi_bits :].as_signed()

    def __add__(self, o: "FixedPointView") -> "FixedPointView":
        assert self.shape() == o.shape(), "Mismatched fixed point formats"
        s = self.shape()
        return s((self.data + o.data)[: s.total_bits])

    def __sub__(self, o: "FixedPointView") -> "FixedPointView":
        assert self.shape() == o.shape(), "Mismatched fixed point formats"
        s = self.shape()
        return s((self.data - o.data)[: s.total_bits])

    def __mul__(self, o: "FixedPointView") -> "FixedPointView":
        assert self.shape() == o.shape(), "Mismatched fixed point formats"
        s = self.shape()
        return s((self.data * o.data)[s.lo_bits : s.lo_bits + s.total_bits])

    def format(self, format_spec):
        fract_bits = len(self.fract)

        # format as standard fixed_point
        decimal_length = ceil(log10(2) * fract_bits)
        len_pow = 10**decimal_length

        if format_spec == "b":
            # format as binary
            return Format(
                "{value.int:b}.{value.fract:0{fract_bits}b}}",
                value=self,
                fract_bits=fract_bits,
            )
        elif format_spec == "x":
            # format as hex
            if fract_bits % 4 != 0:
                raise ValueError(
                    "Hex format requires number of fractional bits to be multiple of 4"
                )
            return Format(
                "{value.int:x}.{value.fract:0{fract_bits}x}}",
                value=self,
                fract_bits=fract_bits // 4,
            )
        elif format_spec == "u":
            # format as unsigned float
            v2 = (self.data.as_unsigned() * len_pow // (1 << fract_bits)) % len_pow
            return Format("{:d}.{:0{}d}", self.int.as_unsigned(), v2, decimal_length)
        elif format_spec != "":
            raise ValueError(
                f"Format specifier {format_spec!r} is not supported for layouts"
            )

        v1 = Mux((self.fract > 0) & (self.int < 0), self.int + 1, self.int)
        v2 = (self.data.as_signed() * len_pow // (1 << fract_bits)) % len_pow
        v3 = Mux((self.int < 0) & (self.fract > 0), len_pow - v2, v2)
        return Format("{:-d}.{:0{}d}", v1, v3, decimal_length)


# DE1-SoC has DSP blocks that can do 27x27 multiplications, so we use 13.13 format by default
FixedPoint = FixedPointLayout(lo_bits=13, hi_bits=13)
Vector2 = data.ArrayLayout(FixedPoint, 2)
Vector3 = data.ArrayLayout(FixedPoint, 3)
Vector4 = data.ArrayLayout(FixedPoint, 4)
