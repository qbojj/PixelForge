"""LiteX wrappers for GPU pipeline components."""

from .build import generate_all_verilog
from .build import generate_verilog as generate_single_verilog
from .litex_clipper import LiteXPrimitiveClipper
from .litex_depthstencil import LiteXDepthStencilTest
from .litex_gpu import LiteXGPU, generate_verilog
from .litex_index_generator import LiteXIndexGenerator
from .litex_input_assembly import LiteXInputAssembly
from .litex_input_topology import LiteXInputTopologyProcessor
from .litex_primitive_assembly import LiteXPrimitiveAssembly
from .litex_rasterizer import LiteXTriangleRasterizer
from .litex_swapchain import LiteXSwapchainOutput
from .litex_texturing import LiteXTexturing
from .litex_vertex_shading import LiteXVertexShading
from .litex_vertex_transform import LiteXVertexTransform

__all__ = [
    "LiteXGPU",
    "generate_verilog",
    "LiteXIndexGenerator",
    "LiteXInputTopologyProcessor",
    "LiteXInputAssembly",
    "LiteXVertexTransform",
    "LiteXVertexShading",
    "LiteXPrimitiveAssembly",
    "LiteXPrimitiveClipper",
    "LiteXTriangleRasterizer",
    "LiteXTexturing",
    "LiteXDepthStencilTest",
    "LiteXSwapchainOutput",
    "generate_all_verilog",
    "generate_single_verilog",
]
