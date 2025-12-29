import amaranth_soc.wishbone.bus as wb
from amaranth import *
from amaranth.lib import data, wiring
from amaranth.lib.wiring import In, Out

from .input_assembly.cores import (
    IndexGenerator,
    InputAssembly,
    InputAssemblyAttrConfigLayout,
    InputTopologyProcessor,
)
from .pixel_shading.cores import (
    BlendConfig,
    DepthStencilTest,
    DepthTestConfig,
    StencilOpConfig,
    SwapchainOutput,
    Texturing,
)
from .primitive_assembly.cores import (
    PrimitiveAssembly,
    PrimitiveAssemblyConfigLayout,
)
from .rasterizer.cores import PrimitiveClipper
from .rasterizer.rasterizer import TriangleRasterizer
from .utils.layouts import (
    FramebufferInfoLayout,
    num_lights,
    num_textures,
    wb_bus_addr_width,
    wb_bus_data_width,
)
from .utils.types import (
    FixedPoint_mem,
    IndexKind,
    InputTopology,
    address_shape,
)
from .vertex_shading.cores import (
    LightPropertyLayout,
    MaterialPropertyLayout,
    VertexShading,
)
from .vertex_transform.cores import (
    VertexTransform,
    VertexTransformEnablementLayout,
)


class GraphicsPipeline(wiring.Component):
    """End-to-end graphics pipeline wiring.

    Stages (streams):
      IndexGenerator → InputTopologyProcessor → InputAssembly → VertexTransform →
      VertexShading → PrimitiveAssembly → PrimitiveClipper → TriangleRasterizer →
      Texturing → DepthStencilTest → SwapchainOutput

    Exposes separate Wishbone buses for vertex fetch, depth/stencil and color.
    """

    # Draw/index generation
    c_index_address: In(address_shape)
    c_index_count: In(unsigned(32))
    c_index_kind: In(IndexKind)
    start: In(1)

    # Input topology
    c_input_topology: In(InputTopology)
    c_primitive_restart_enable: In(unsigned(1))
    c_primitive_restart_index: In(unsigned(32))
    c_base_vertex: In(unsigned(32))

    # Input assembly attributes
    c_pos: In(InputAssemblyAttrConfigLayout)
    c_norm: In(InputAssemblyAttrConfigLayout)
    c_tex: In(InputAssemblyAttrConfigLayout).array(num_textures)
    c_col: In(InputAssemblyAttrConfigLayout)

    # Vertex transform configuration
    vt_enabled: In(VertexTransformEnablementLayout)
    position_mv: In(data.ArrayLayout(FixedPoint_mem, 16))
    position_p: In(data.ArrayLayout(FixedPoint_mem, 16))
    normal_mv_inv_t: In(data.ArrayLayout(FixedPoint_mem, 9))
    texture_transforms: In(data.ArrayLayout(FixedPoint_mem, 16)).array(num_textures)

    # Shading configuration
    material: In(MaterialPropertyLayout)
    lights: In(LightPropertyLayout).array(num_lights)

    # Primitive assembly configuration
    pa_conf: In(PrimitiveAssemblyConfigLayout)

    # Framebuffer and per-fragment
    fb_info: In(FramebufferInfoLayout)
    stencil_conf_front: In(StencilOpConfig)
    stencil_conf_back: In(StencilOpConfig)
    depth_conf: In(DepthTestConfig)
    blend_conf: In(BlendConfig)

    # Wishbone buses (separate for simplicity)
    wb_index: Out(
        wb.Signature(
            addr_width=wb_bus_addr_width, data_width=wb_bus_data_width, granularity=8
        )
    )
    wb_vertex: Out(
        wb.Signature(addr_width=wb_bus_addr_width, data_width=wb_bus_data_width)
    )
    wb_depthstencil: Out(
        wb.Signature(
            addr_width=wb_bus_addr_width, data_width=wb_bus_data_width, granularity=8
        )
    )
    wb_color: Out(
        wb.Signature(addr_width=wb_bus_addr_width, data_width=wb_bus_data_width)
    )

    # Backpressure/ready (reflect index generator readiness)
    ready: Out(1)

    def elaborate(self, platform):
        m = Module()

        # Submodules
        m.submodules.idx = idx = IndexGenerator()
        m.submodules.topo = topo = InputTopologyProcessor()
        m.submodules.ia = ia = InputAssembly()

        m.submodules.vtx_xf = vtx_xf = VertexTransform()
        m.submodules.vtx_sh = vtx_sh = VertexShading()

        m.submodules.pa = pa = PrimitiveAssembly()
        m.submodules.clip = clip = PrimitiveClipper()
        m.submodules.rast = rast = TriangleRasterizer()

        m.submodules.tex = tex = Texturing()
        m.submodules.ds = ds = DepthStencilTest()
        m.submodules.sc = sc = SwapchainOutput()

        # Streams wiring: IA chain
        wiring.connect(m, idx.os_index, topo.is_index)
        wiring.connect(m, topo.os_index, ia.is_index)
        wiring.connect(m, ia.os_vertex, vtx_xf.is_vertex)
        wiring.connect(m, vtx_xf.os_vertex, vtx_sh.is_vertex)
        wiring.connect(m, vtx_sh.os_vertex, pa.is_vertex)
        wiring.connect(m, pa.os_primitive, clip.is_vertex)
        wiring.connect(m, clip.os_vertex, rast.is_vertex)
        wiring.connect(m, rast.os_fragment, tex.is_fragment)
        wiring.connect(m, tex.os_fragment, ds.is_fragment)
        wiring.connect(m, ds.os_fragment, sc.is_fragment)

        # Wishbone buses wiring
        wiring.connect(m, idx.bus, wiring.flipped(self.wb_index))
        wiring.connect(m, ia.bus, wiring.flipped(self.wb_vertex))
        wiring.connect(m, ds.wb_bus, wiring.flipped(self.wb_depthstencil))
        wiring.connect(m, sc.wb_bus, wiring.flipped(self.wb_color))

        # Pipeline-wide ready: AND of all sub-module ready signals only
        m.d.comb += self.ready.eq(
            idx.ready
            & topo.ready
            & ia.ready
            & vtx_xf.ready
            & vtx_sh.ready
            & pa.ready
            & clip.ready
            & rast.ready
            & tex.ready
            & ds.ready
            & sc.ready
        )

        # IndexGenerator configuration
        m.d.comb += [
            idx.c_address.eq(self.c_index_address),
            idx.c_count.eq(self.c_index_count),
            idx.c_kind.eq(self.c_index_kind),
            idx.start.eq(self.start),
        ]

        # Topology configuration
        m.d.comb += [
            topo.c_input_topology.eq(self.c_input_topology),
            topo.c_primitive_restart_enable.eq(self.c_primitive_restart_enable),
            topo.c_primitive_restart_index.eq(self.c_primitive_restart_index),
            topo.c_base_vertex.eq(self.c_base_vertex),
        ]

        # Input Assembly configuration
        m.d.comb += [
            ia.c_pos.eq(self.c_pos),
            ia.c_norm.eq(self.c_norm),
            *[ia.c_tex[i].eq(self.c_tex[i]) for i in range(num_textures)],
            ia.c_col.eq(self.c_col),
        ]

        # Vertex transform configuration
        m.d.comb += [
            vtx_xf.enabled.eq(self.vt_enabled),
            vtx_xf.position_mv.eq(self.position_mv),
            vtx_xf.position_p.eq(self.position_p),
            vtx_xf.normal_mv_inv_t.eq(self.normal_mv_inv_t),
            *[
                vtx_xf.texture_transforms[i].eq(self.texture_transforms[i])
                for i in range(num_textures)
            ],
        ]

        # Vertex shading configuration
        m.d.comb += [
            vtx_sh.material.eq(self.material),
            *[vtx_sh.lights[i].eq(self.lights[i]) for i in range(num_lights)],
        ]

        # Primitive assembly and clipper configuration
        m.d.comb += [
            pa.config.eq(self.pa_conf),
            clip.prim_type.eq(self.pa_conf.type),
        ]

        # Framebuffer and per-fragment config
        m.d.comb += [
            rast.fb_info.eq(self.fb_info),
            ds.fb_info.eq(self.fb_info),
            sc.fb_info.eq(self.fb_info),
            ds.stencil_conf_back.eq(self.stencil_conf_back),
            ds.stencil_conf_front.eq(self.stencil_conf_front),
            ds.depth_conf.eq(self.depth_conf),
            sc.conf.eq(self.blend_conf),
        ]

        return m


__all__ = ["GraphicsPipeline"]
