from amaranth import *
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out


class SevenSegmentDisplay(wiring.Component):
    i: In(4)

    def __init__(self, seg_index: int = 0):
        super().__init__()
        self.seg_index = seg_index

    def elaborate(self, platform) -> Module:
        m = Module()

        names = ["a", "b", "c", "d", "e", "f", "g"]
        resource = platform.request("display_7seg", self.seg_index)
        segments = Cat(getattr(resource, name).o for name in names)

        with m.Switch(self.i):
            with m.Case(0):
                m.d.comb += segments.eq(0b0111111)  # Display 0
            with m.Case(1):
                m.d.comb += segments.eq(0b0000110)  # Display 1
            with m.Case(2):
                m.d.comb += segments.eq(0b1011011)  # Display 2
            with m.Case(3):
                m.d.comb += segments.eq(0b1001111)  # Display 3
            with m.Case(4):
                m.d.comb += segments.eq(0b1100110)  # Display 4
            with m.Case(5):
                m.d.comb += segments.eq(0b1101101)  # Display 5
            with m.Case(6):
                m.d.comb += segments.eq(0b1111101)  # Display 6
            with m.Case(7):
                m.d.comb += segments.eq(0b0000111)  # Display 7
            with m.Case(8):
                m.d.comb += segments.eq(0b1111111)  # Display 8
            with m.Case(9):
                m.d.comb += segments.eq(0b1101111)  # Display 9
            with m.Case(10):
                m.d.comb += segments.eq(0b1110111)  # Display A
            with m.Case(11):
                m.d.comb += segments.eq(0b1111100)  # Display b
            with m.Case(12):
                m.d.comb += segments.eq(0b0111001)  # Display C
            with m.Case(13):
                m.d.comb += segments.eq(0b1011110)  # Display d
            with m.Case(14):
                m.d.comb += segments.eq(0b1111001)  # Display E
            with m.Case(15):
                m.d.comb += segments.eq(0b1110001)  # Display F

        return m


class Synchronizer(wiring.Component):
    i: In(1)
    o: Out(1)

    def elaborate(self, platform) -> Module:
        m = Module()

        sync_reg1 = Signal()
        sync_reg2 = Signal()

        m.d.sync += sync_reg1.eq(self.i)
        m.d.sync += sync_reg2.eq(sync_reg1)

        m.d.comb += self.o.eq(sync_reg2)

        return m


class EdgeDetector(wiring.Component):
    i: In(1)
    rising_edge: Out(1)
    falling_edge: Out(1)

    def elaborate(self, platform) -> Module:
        m = Module()

        prev_signal = Signal()

        m.d.sync += prev_signal.eq(self.i)

        m.d.comb += self.rising_edge.eq(self.i & ~prev_signal)
        m.d.comb += self.falling_edge.eq(~self.i & prev_signal)

        return m


class Debouncer(wiring.Component):
    i: In(1)
    o: Out(1)

    def __init__(self, width: int = 16):
        super().__init__()
        self.width = width

    def elaborate(self, platform) -> Module:
        m = Module()

        history = Signal(self.width)

        m.d.sync += history.eq(Cat(self.i, history[:-1]))
        with m.If(history == 0):
            m.d.sync += self.o.eq(0)
        with m.Elif(history == (2**self.width - 1)):
            m.d.sync += self.o.eq(1)

        return m


class EdgeDetector_DB(wiring.Component):
    i: In(1)
    rising_edge: Out(1)
    falling_edge: Out(1)

    def __init__(self, width: int = 16):
        super().__init__()
        self.width = width

    def elaborate(self, platform) -> Module:
        m = Module()

        debouncer = m.submodules.debouncer = Debouncer(self.width)
        edge_detector = m.submodules.edge_detector = EdgeDetector()

        m.d.comb += [
            debouncer.i.eq(self.i),
            edge_detector.i.eq(debouncer.o),
            self.rising_edge.eq(edge_detector.rising_edge),
            self.falling_edge.eq(edge_detector.falling_edge),
        ]

        return m
