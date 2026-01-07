from amaranth import *
from amaranth.lib import data

from .types import (
    FixedPoint,
    FixedPoint_depth,
    FixedPoint_fb,
    Vector3,
    Vector4,
    address_shape,
    stride_shape,
    texture_coord_shape,
)

# Number of supported textures and lights
num_textures = 0
num_lights = 1

texture_coords = data.ArrayLayout(Vector4, num_textures)
texture_position = data.ArrayLayout(texture_coord_shape, 2)

# Wishbone bus parameters for GPU memory access
# granularity 1 byte, data width 4 bytes, 32-bit addresses
wb_bus_data_width = 32
wb_bus_granularity = 8
wb_bus_addr_width = 30  # Addresses are per data width (4 bytes)


class VertexLayout(data.Struct):
    position: Vector4
    normal: Vector3
    texcoords: texture_coords
    color: Vector4


class ShadingVertexLayout(data.Struct):
    position_view: Vector4
    position_proj: Vector4
    normal_view: Vector3
    texcoords: texture_coords  # After transforms
    color: Vector4


class PrimitiveAssemblyLayout(data.Struct):
    position_ndc: Vector4
    texcoords: texture_coords
    color: Vector4
    color_back: Vector4


class RasterizerLayout(data.Struct):
    position_ndc: Vector4  # In normalized device coordinates
    texcoords: texture_coords
    color: Vector4
    front_facing: unsigned(1)


class FragmentLayout(data.Struct):
    depth: FixedPoint
    texcoords: texture_coords
    color: Vector4  # rgba in linear space
    coord_pos: texture_position
    front_facing: unsigned(1)


class FramebufferInfoLayout(data.Struct):
    width: texture_coord_shape
    height: texture_coord_shape

    # Viewport transform (NDC to screen space) - stored as standard FixedPoint_mem (16.16) in CSRs
    # Internal pipeline converts to FixedPoint_fb (12.4) or FixedPoint_depth (UQ(1,15)) as needed
    viewport_x: FixedPoint_fb
    viewport_y: FixedPoint_fb
    viewport_width: FixedPoint_fb
    viewport_height: FixedPoint_fb
    viewport_min_depth: FixedPoint_depth
    viewport_max_depth: FixedPoint_depth

    # Scissor rectangle (rasterization clip region) - integer coordinates
    scissor_offset_x: signed(32)
    scissor_offset_y: signed(32)
    scissor_width: unsigned(32)
    scissor_height: unsigned(32)

    color_address: address_shape  # assume R8G8B8A8
    color_pitch: stride_shape  # in bytes
    depth_address: address_shape  # assume D16
    depth_pitch: stride_shape  # in bytes
    stencil_address: address_shape  # S8
    stencil_pitch: stride_shape  # in bytes
