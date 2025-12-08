from amaranth import *
from amaranth.lib import data, enum

from . import fixed

# DE1-SoC has DSP blocks that can do 27x27 multiplications, so we use 13.13 format by default
FixedPoint = fixed.SQ(13, 13)
Vector2 = data.ArrayLayout(FixedPoint, 2)
Vector3 = data.ArrayLayout(FixedPoint, 3)
Vector4 = data.ArrayLayout(FixedPoint, 4)

# fixed point as it is stored in memory
FixedPoint_mem = fixed.SQ(16, 16)
Vector2_mem = data.ArrayLayout(FixedPoint_mem, 2)
Vector3_mem = data.ArrayLayout(FixedPoint_mem, 3)
Vector4_mem = data.ArrayLayout(FixedPoint_mem, 4)

# collumn-major matrices
Matrix4_mem = data.ArrayLayout(FixedPoint_mem, 16)
Matrix3_mem = data.ArrayLayout(FixedPoint_mem, 9)


texture_coord_shape = unsigned(12)  # Max 4kx4k textures
address_shape = unsigned(32)

# The FPGA has hardware multiplier of max 27x27 bits, so we don't want stride at 32bits
# Using 16 bits should be enough as we don't expect very wide input buffer strides or textures
stride_shape = unsigned(16)
index_shape = unsigned(32)


class IndexKind(enum.Enum, shape=2):
    NOT_INDEXED = 0
    U8 = 1
    U16 = 2
    U32 = 3


class InputTopology(enum.Enum, shape=4):
    """
    Input primitive topology types
    VkPrimitiveTopology equivalent
    """

    POINT_LIST = 0
    LINE_LIST = 1
    LINE_STRIP = 2
    TRIANGLE_LIST = 3
    TRIANGLE_STRIP = 4
    TRIANGLE_FAN = 5
    LINE_LIST_WITH_ADJACENCY = 6
    LINE_STRIP_WITH_ADJACENCY = 7
    TRIANGLE_LIST_WITH_ADJACENCY = 8
    TRIANGLE_STRIP_WITH_ADJACENCY = 9
    PATCH_LIST = 10


class PrimitiveType(enum.Enum, shape=2):
    POINTS = 0
    LINES = 1
    TRIANGLES = 2


class ScalingType(enum.Enum, shape=3):
    """Scaling format type

    Component format changes

    UNORM: value / (2^n - 1)
    SNORM: clamp(value / (2^(n-1) - 1), 1.0, -1.0)
    UINT: value as unsigned integer
    SINT: value as signed integer
    FIXED: value as FixedPoint (16.16)
    FLOAT: value as IEEE 754 floating point
    """

    UNORM = 0
    SNORM = 1
    UINT = 2
    SINT = 3
    FIXED = 4
    FLOAT = 5


class CullFace(enum.Flag, shape=2):
    """
    VkCullModeFlagBits equivalent
    """

    FRONT = 1
    BACK = 2

    NONE = 0
    FRONT_AND_BACK = FRONT | BACK


class FrontFace(enum.Enum, shape=1):
    """
    VkFrontFace equivalent
    """

    CCW = 0
    CW = 1


class CompareOp(enum.Flag, shape=3):
    """
    VkCompareOp equivalent
    """

    NEVER = 0
    LESS = 1
    EQUAL = 2
    LESS_OR_EQUAL = LESS | EQUAL
    GREATER = 4
    NOT_EQUAL = LESS | GREATER
    GREATER_OR_EQUAL = EQUAL | GREATER
    ALWAYS = LESS | EQUAL | GREATER
