from amaranth.lib import data

from ..utils.types import CullFace, FrontFace, PrimitiveType


class PrimitiveAssemblyConfigLayout(data.Struct):
    """Primitive assembly configuration"""

    type: PrimitiveType
    cull: CullFace
    winding: FrontFace
