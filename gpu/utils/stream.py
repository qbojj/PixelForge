from collections.abc import Callable, Iterable
from functools import reduce

from amaranth import *
from amaranth import ShapeCastable, ValueCastable
from amaranth.lib import data, stream, wiring
from amaranth.lib.wiring import In, Out


class VectorToStream(wiring.Component):
    """
    Converts a vector signal into a stream signal.
    """

    def __init__(self, vector_type: ShapeCastable):
        super().__init__(
            {
                "i": In(stream.Signature(vector_type)),
                "o": Out(stream.Signature(vector_type.elem_shape)),
            }
        )

    def elaborate(self, platform) -> Module:
        m = Module()

        num_elem = len(self.i.p)

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

    def __init__(self, vector_type: ShapeCastable):
        super().__init__(
            {
                "i": In(stream.Signature(vector_type.elem_shape)),
                "o": Out(stream.Signature(vector_type)),
            }
        )

    def elaborate(self, platform) -> Module:
        m = Module()

        num_elem = len(self.o.p)
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


class StreamReshape(wiring.Component):
    def __init__(
        self,
        shapes_in: Iterable[ShapeCastable],
        shapes_out: Iterable[ShapeCastable],
        reshaper: Callable[[list[ValueCastable]], list[ValueCastable]],
    ):
        attrs = {}

        for i, s in enumerate(shapes_in):
            attrs[f"i__{i}"] = In(stream.Signature(s))

        for i, s in enumerate(shapes_out):
            attrs[f"o__{i}"] = Out(stream.Signature(s))

        super().__init__(attrs)
        self._reshaper = reshaper
        self._i = Array(getattr(self, f"i__{i}") for i in range(len(shapes_in)))
        self._o = Array(getattr(self, f"o__{i}") for i in range(len(shapes_out)))

    def elaborate(self, platform) -> Module:
        m = Module()

        in_valid = Signal()
        out_ready_for_next = Signal()
        m.d.comb += in_valid.eq(reduce(lambda a, b: a & b, (s.valid for s in self._i)))
        m.d.comb += out_ready_for_next.eq(
            reduce(lambda a, b: a & b, (s.ready | ~s.valid for s in self._o))
        )

        for out_s in self._o:
            with m.If(out_s.ready):
                m.d.sync += out_s.valid.eq(0)

        with m.If(in_valid & out_ready_for_next):
            m.d.comb += [s.ready.eq(1) for s in self._i]

            reshaped = self._reshaper([s.p for s in self._i])
            for i, out_s in enumerate(self._o):
                m.d.sync += [
                    out_s.p.eq(reshaped[i]),
                    out_s.valid.eq(1),
                ]

        return m


class ValueDuplicator(wiring.Component):
    """
    Duplicate input stream n times
    """

    def __init__(self, shape: ShapeCastable, max_n: int):
        super().__init__(
            {
                "i": In(stream.Signature(shape)),
                "n": In(stream.Signature(range(max_n + 1))),
                "o": Out(stream.Signature(shape)),
            }
        )

    def elaborate(self, platform) -> Module:
        m = Module()

        count = Signal.like(self.n.p)

        with m.FSM():
            with m.State("IDLE"):
                with m.If(self.i.valid & self.n.valid):
                    m.d.comb += [
                        self.i.ready.eq(1),
                        self.n.ready.eq(1),
                    ]
                    with m.If(self.n.p > 0):
                        m.d.sync += [
                            count.eq(self.n.p - 1),
                            self.o.p.eq(self.i.p),
                            self.o.valid.eq(1),
                        ]
                        m.next = "DUPLICATE"
            with m.State("DUPLICATE"):
                with m.If(self.o.ready):
                    with m.If(count > 0):
                        m.d.sync += count.eq(count - 1)
                    with m.Else():
                        m.d.sync += self.o.valid.eq(0)
                        m.next = "IDLE"

        return m


class RoundRobinDistributor(wiring.Component):
    """
    Distributes input stream to multiple output streams in round-robin fashion
    """

    def __init__(self, shape: ShapeCastable, num_outputs: int):
        super().__init__(
            {
                "i": In(stream.Signature(shape)),
                "o": Out(stream.Signature(shape)).array(num_outputs),
            }
        )
        self.num_outputs = num_outputs

    def elaborate(self, platform) -> Module:
        m = Module()

        current_output = Signal(range(self.num_outputs))
        outs = Array(self.o)

        for out_s in outs:
            with m.If(out_s.ready):
                m.d.sync += out_s.valid.eq(0)

        with m.If(
            self.i.valid & (outs[current_output].ready | ~outs[current_output].valid)
        ):
            m.d.sync += [
                outs[current_output].p.eq(self.i.p),
                outs[current_output].valid.eq(1),
            ]
            m.d.comb += self.i.ready.eq(1)
            m.d.sync += current_output.eq(
                Mux(current_output == self.num_outputs - 1, 0, current_output + 1)
            )

        return m


class RoundRobinRecombiner(wiring.Component):
    """
    If values were distributed using RoundRobinDistributor and assuming no elements were dropped
    or added, this component recombines them back into a single stream in the original order.
    """

    def __init__(self, shape: ShapeCastable, num_inputs: int):
        super().__init__(
            {
                "i": In(stream.Signature(shape)).array(num_inputs),
                "o": Out(stream.Signature(shape)),
            }
        )
        self.num_inputs = num_inputs

    def elaborate(self, platform) -> Module:
        m = Module()

        current_input = Signal(range(self.num_inputs))
        ins = Array(self.i)

        m.d.comb += [
            self.o.valid.eq(ins[current_input].valid),
            self.o.p.eq(ins[current_input].p),
            ins[current_input].ready.eq(self.o.ready),
        ]

        with m.If(self.o.valid & self.o.ready):
            m.d.sync += current_input.eq(
                Mux(current_input == self.num_inputs - 1, 0, current_input + 1)
            )

        return m


class AnyDistributor(wiring.Component):
    """
    Distributes input stream to multiple output streams, taking data from any output that is ready.
    If the subsequent modules are constant latency, you should use RoundRobinDistributor instead.
    """

    def __init__(self, shape: ShapeCastable, num_outputs: int):
        super().__init__(
            {
                "i": In(stream.Signature(shape)),
                "o": Out(stream.Signature(shape)).array(num_outputs),
            }
        )
        self.num_outputs = num_outputs

    def elaborate(self, platform) -> Module:
        m = Module()

        for out_s in self.o:
            with m.If(out_s.ready):
                m.d.sync += out_s.valid.eq(0)

        sel = Signal(range(self.num_outputs + 1), reset=self.num_outputs)

        for i, o in enumerate(self.o):
            with m.If(~o.valid | o.ready):
                m.d.comb += sel.eq(i)

        m.d.comb += self.i.ready.eq(sel != self.num_outputs)
        with m.If(self.i.valid):
            for i in range(self.num_outputs):
                with m.If(sel == i):
                    m.d.sync += [
                        self.o[i].p.eq(self.i.p),
                        self.o[i].valid.eq(1),
                    ]

        return m


class AnyRecombiner(wiring.Component):
    """
    Recombines multiple input streams into a single output stream, taking data from any input that is valid.
    """

    def __init__(self, shape: ShapeCastable, num_inputs: int):
        super().__init__(
            {
                "i": In(stream.Signature(shape)).array(num_inputs),
                "o": Out(stream.Signature(shape)),
            }
        )
        self.num_inputs = num_inputs

    def elaborate(self, platform) -> Module:
        m = Module()

        with m.If(self.o.ready):
            m.d.sync += self.o.valid.eq(0)

        sel = Signal(range(self.num_inputs + 1), reset=self.num_inputs)
        for i, inp in enumerate(self.i):
            with m.If(inp.valid):
                m.d.comb += sel.eq(i)

        with m.If(~self.o.valid | self.o.ready):
            for i in range(self.num_inputs):
                with m.If(sel == i):
                    m.d.sync += [
                        self.o.p.eq(self.i[i].p),
                        self.o.valid.eq(1),
                    ]
                    m.d.comb += self.i[i].ready.eq(1)

        return m


class WideStreamOutput(wiring.Component):
    """
    Splits wide output stream into multiple cycles.
    """

    def __init__(self, shape: ShapeCastable, max_width: int):
        sig = data.StructLayout(
            {
                "data": data.ArrayLayout(shape, max_width),
                "n": range(max_width + 1),
            }
        )
        super().__init__(
            {
                "i": In(stream.Signature(sig)),
                "o": Out(stream.Signature(shape)),
            }
        )

    def elaborate(self, platform):
        m = Module()

        n = Signal.like(self.i.p.n)
        i = Signal.like(self.i.p.n)
        p = Signal.like(self.i.p.data)

        with m.FSM():
            with m.State("IDLE"):
                m.d.comb += self.i.ready.eq(1)
                with m.If(self.i.valid):
                    with m.If(self.i.p.n > 0):
                        m.d.sync += [
                            n.eq(self.i.p.n),
                            i.eq(0),
                            p.eq(self.i.p.data),
                            self.o.p.eq(self.i.p.data[0]),
                            self.o.valid.eq(1),
                        ]
                        m.next = "SEND"
            with m.State("SEND"):
                with m.If(self.o.ready):
                    with m.If(i + 1 < n):
                        m.d.sync += [
                            i.eq(i + 1),
                            self.o.p.eq(p[i + 1]),
                        ]
                    with m.Else():
                        m.d.sync += self.o.valid.eq(0)
                        m.next = "IDLE"

        return m
