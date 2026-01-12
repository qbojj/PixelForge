from amaranth.lib import data, enum

from ..utils.types import Vector4_mem, address_shape, stride_shape

__all__ = ["InputMode", "InputData"]


class InputMode(enum.Enum, shape=1):
    CONSTANT = 0
    PER_VERTEX = 1


class PerVertexData(data.Struct):
    address: address_shape
    stride: stride_shape
    _pad0: 16
    _pad1: 32
    # for now assume format (FixedPoint 16.16) for all attributes
    # with appropriate number of components


class InputData(data.Union):
    constant_value: Vector4_mem
    per_vertex: PerVertexData
