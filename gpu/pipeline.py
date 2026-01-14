import amaranth_soc.wishbone.bus as wb
from amaranth import *
from amaranth.lib import data, fifo, wiring
from amaranth.lib.wiring import In, Out
from amaranth_soc import csr
from amaranth_soc.csr.wishbone import WishboneCSRBridge
from amaranth_soc.memory import MemoryMap

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
from .rasterizer.rasterizer import PerspectiveDivide, TriangleRasterizer
from .utils import avalon as avl
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

__all__ = ["GraphicsPipeline", "GraphicsPipelineCSR", "GraphicsPipelineAvalonCSR"]


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
        wb.Signature(addr_width=wb_bus_addr_width, data_width=wb_bus_data_width)
    )
    wb_vertex: Out(
        wb.Signature(addr_width=wb_bus_addr_width, data_width=wb_bus_data_width)
    )
    wb_depthstencil: Out(
        wb.Signature(addr_width=wb_bus_addr_width, data_width=wb_bus_data_width)
    )
    wb_color: Out(
        wb.Signature(addr_width=wb_bus_addr_width, data_width=wb_bus_data_width)
    )

    # ready (reflect index generator readiness)
    ready: Out(1)
    ready_components: Out(
        4
    )  # [input assembly, vertex transform, rasterizer, pixel pipeline]
    ready_vec: Out(32)

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
        m.submodules.div = div = PerspectiveDivide()
        m.submodules.rast = rast = TriangleRasterizer(num_generators=5)

        m.submodules.tex = tex = Texturing()
        m.submodules.ds = ds = DepthStencilTest()
        m.submodules.sc = sc = SwapchainOutput()

        fifo_size_default = 256

        # FIFO buffers between stages
        m.submodules.idx_to_topo_fifo = fifo_idx_topo = fifo.SyncFIFOBuffered(
            width=Shape.cast(topo.is_index.p.shape()).width, depth=fifo_size_default
        )
        m.submodules.topo_to_ia_fifo = fifo_topo_ia = fifo.SyncFIFOBuffered(
            width=Shape.cast(ia.is_index.p.shape()).width, depth=fifo_size_default
        )
        m.submodules.ia_to_vtx_xf_fifo = fifo_ia_vtx_xf = fifo.SyncFIFOBuffered(
            width=Shape.cast(vtx_xf.is_vertex.p.shape()).width, depth=fifo_size_default
        )
        m.submodules.vtx_xf_to_vtx_sh_fifo = fifo_vtx_xf_vtx_sh = fifo.SyncFIFOBuffered(
            width=Shape.cast(vtx_sh.is_vertex.p.shape()).width, depth=fifo_size_default
        )
        m.submodules.vtx_sh_to_pa_fifo = fifo_vtx_sh_pa = fifo.SyncFIFOBuffered(
            width=Shape.cast(pa.is_vertex.p.shape()).width, depth=fifo_size_default
        )
        m.submodules.pa_to_clip_fifo = fifo_pa_clip = fifo.SyncFIFOBuffered(
            width=Shape.cast(clip.is_vertex.p.shape()).width, depth=fifo_size_default
        )
        m.submodules.clip_to_div_fifo = fifo_clip_div = fifo.SyncFIFOBuffered(
            width=Shape.cast(div.i_vertex.p.shape()).width, depth=fifo_size_default
        )
        m.submodules.div_to_rast_fifo = fifo_div_rast = fifo.SyncFIFOBuffered(
            width=Shape.cast(rast.is_vertex.p.shape()).width, depth=fifo_size_default
        )
        m.submodules.rast_to_tex_fifo = fifo_rast_tex = fifo.SyncFIFOBuffered(
            width=Shape.cast(tex.is_fragment.p.shape()).width, depth=fifo_size_default
        )
        m.submodules.tex_to_ds_fifo = fifo_tex_ds = fifo.SyncFIFOBuffered(
            width=Shape.cast(ds.is_fragment.p.shape()).width, depth=fifo_size_default
        )
        m.submodules.ds_to_sc_fifo = fifo_ds_sc = fifo.SyncFIFOBuffered(
            width=Shape.cast(sc.is_fragment.p.shape()).width, depth=fifo_size_default
        )

        # Streams wiring: IA chain
        wiring.connect(m, idx.os_index, fifo_idx_topo.w_stream)

        wiring.connect(m, fifo_idx_topo.r_stream, topo.is_index)
        wiring.connect(m, topo.os_index, fifo_topo_ia.w_stream)

        wiring.connect(m, fifo_topo_ia.r_stream, ia.is_index)
        wiring.connect(m, ia.os_vertex, fifo_ia_vtx_xf.w_stream)

        wiring.connect(m, fifo_ia_vtx_xf.r_stream, vtx_xf.is_vertex)
        wiring.connect(m, vtx_xf.os_vertex, fifo_vtx_xf_vtx_sh.w_stream)

        wiring.connect(m, fifo_vtx_xf_vtx_sh.r_stream, vtx_sh.is_vertex)
        wiring.connect(m, vtx_sh.os_vertex, fifo_vtx_sh_pa.w_stream)

        wiring.connect(m, fifo_vtx_sh_pa.r_stream, pa.is_vertex)
        wiring.connect(m, pa.os_primitive, fifo_pa_clip.w_stream)

        wiring.connect(m, fifo_pa_clip.r_stream, clip.is_vertex)
        wiring.connect(m, clip.os_vertex, fifo_clip_div.w_stream)

        wiring.connect(m, fifo_clip_div.r_stream, div.i_vertex)
        wiring.connect(m, div.o_vertex, fifo_div_rast.w_stream)

        wiring.connect(m, fifo_div_rast.r_stream, rast.is_vertex)
        wiring.connect(m, rast.os_fragment, fifo_rast_tex.w_stream)

        wiring.connect(m, fifo_rast_tex.r_stream, tex.is_fragment)
        wiring.connect(m, tex.os_fragment, fifo_tex_ds.w_stream)

        wiring.connect(m, fifo_tex_ds.r_stream, ds.is_fragment)
        wiring.connect(m, ds.os_fragment, fifo_ds_sc.w_stream)

        wiring.connect(m, fifo_ds_sc.r_stream, sc.is_fragment)

        # Wishbone buses wiring
        wiring.connect(m, idx.bus, wiring.flipped(self.wb_index))
        wiring.connect(m, ia.bus, wiring.flipped(self.wb_vertex))
        wiring.connect(m, ds.wb_bus, wiring.flipped(self.wb_depthstencil))
        wiring.connect(m, sc.wb_bus, wiring.flipped(self.wb_color))

        input_assembly_ready_ = [
            idx.ready,
            (fifo_idx_topo.level == 0) & ~fifo_idx_topo.w_en,
            topo.ready,
            (fifo_topo_ia.level == 0) & ~fifo_topo_ia.w_en,
            ia.ready,
        ]

        vertex_transform_ready_ = [
            (fifo_ia_vtx_xf.level == 0) & ~fifo_ia_vtx_xf.w_en,
            vtx_xf.ready,
            (fifo_vtx_xf_vtx_sh.level == 0) & ~fifo_vtx_xf_vtx_sh.w_en,
            vtx_sh.ready,
        ]

        raster_ready_ = [
            (fifo_vtx_sh_pa.level == 0) & ~fifo_vtx_sh_pa.w_en,
            pa.ready,
            (fifo_pa_clip.level == 0) & ~fifo_pa_clip.w_en,
            clip.ready,
            (fifo_clip_div.level == 0) & ~fifo_clip_div.w_en,
            div.ready,
            (fifo_div_rast.level == 0) & ~fifo_div_rast.w_en,
            rast.ready,
        ]

        fragment_processing_ready_ = [
            (fifo_rast_tex.level == 0) & ~fifo_rast_tex.w_en,
            tex.ready,
            (fifo_tex_ds.level == 0) & ~fifo_tex_ds.w_en,
            ds.ready,
            (fifo_ds_sc.level == 0) & ~fifo_ds_sc.w_en,
            sc.ready,
        ]

        input_assembly_ready = Signal(len(input_assembly_ready_))
        m.d.comb += input_assembly_ready.eq(Cat(input_assembly_ready_))

        vertex_transform_ready = Signal(len(vertex_transform_ready_))
        m.d.comb += vertex_transform_ready.eq(Cat(vertex_transform_ready_))

        raster_ready = Signal(len(raster_ready_))
        m.d.comb += raster_ready.eq(Cat(raster_ready_))

        fragment_processing_ready = Signal(len(fragment_processing_ready_))
        m.d.comb += fragment_processing_ready.eq(Cat(fragment_processing_ready_))

        m.d.comb += self.ready_vec.eq(
            Cat(
                input_assembly_ready,
                vertex_transform_ready,
                raster_ready,
                fragment_processing_ready,
            )
        )

        m.d.comb += self.ready_components.eq(
            Cat(
                input_assembly_ready.all(),
                vertex_transform_ready.all(),
                raster_ready.all(),
                fragment_processing_ready.all(),
            )
        )

        m.d.comb += self.ready.eq(self.ready_components.all())

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


class GraphicsPipelineCSR(wiring.Component):
    """Graphics pipeline with CSR interface exposing configuration registers."""

    ready: Out(1)

    wb_index: Out(
        wb.Signature(addr_width=wb_bus_addr_width, data_width=wb_bus_data_width)
    )
    wb_vertex: Out(
        wb.Signature(addr_width=wb_bus_addr_width, data_width=wb_bus_data_width)
    )
    wb_depthstencil: Out(
        wb.Signature(addr_width=wb_bus_addr_width, data_width=wb_bus_data_width)
    )
    wb_color: Out(
        wb.Signature(addr_width=wb_bus_addr_width, data_width=wb_bus_data_width)
    )

    wb_csr: In(wb.Signature(addr_width=12, data_width=32, granularity=32))

    def elaborate(self, platform):
        m = Module()

        m.submodules.pipeline = pipeline = GraphicsPipeline()

        bld = csr.Builder(addr_width=12, data_width=32)

        class RWReg(csr.Register):
            def __init__(self, field_shape):
                super().__init__(
                    csr.Field(csr.action.RW, Shape.cast(field_shape)), "rw"
                )

        with bld.Cluster("idx"):
            idx_addr = bld.add("address", RWReg(unsigned(32)))
            idx_count = bld.add("count", RWReg(unsigned(32)))
            idx_kind = bld.add(
                "kind",
                RWReg(pipeline.c_index_kind.shape()),
            )
            idx_start = bld.add("start", csr.Register(csr.Field(csr.action.W, 1), "w"))

            m.d.comb += [
                pipeline.c_index_address.eq(idx_addr.f.data),
                pipeline.c_index_count.eq(idx_count.f.data),
                pipeline.c_index_kind.eq(idx_kind.f.data),
                pipeline.start.eq(idx_start.f.w_data & idx_start.f.w_stb),
            ]

        with bld.Cluster("topo"):
            topo_topology = bld.add(
                "input_topology",
                RWReg(pipeline.c_input_topology.shape()),
            )
            topo_prim_restart_en = bld.add(
                "primitive_restart_enable",
                RWReg(pipeline.c_primitive_restart_enable.shape()),
            )
            topo_prim_restart_idx = bld.add(
                "primitive_restart_index",
                RWReg(pipeline.c_primitive_restart_index.shape()),
            )
            topo_base_vertex = bld.add(
                "base_vertex", RWReg(pipeline.c_base_vertex.shape())
            )

            m.d.comb += [
                pipeline.c_input_topology.eq(topo_topology.f.data),
                pipeline.c_primitive_restart_enable.eq(topo_prim_restart_en.f.data),
                pipeline.c_primitive_restart_index.eq(topo_prim_restart_idx.f.data),
                pipeline.c_base_vertex.eq(topo_base_vertex.f.data),
            ]

        with bld.Cluster("ia"):

            def add_attr_cluster(name, cfg):
                with bld.Cluster(name):
                    mode = bld.add("mode", RWReg(cfg.mode.shape()))
                    info = bld.add("info", RWReg(cfg.info.shape()))
                return mode, info

            ia_pos_mode, ia_pos_info = add_attr_cluster("pos", pipeline.c_pos)
            ia_norm_mode, ia_norm_info = add_attr_cluster("norm", pipeline.c_norm)
            ia_tex_mode = []
            ia_tex_info = []
            for i in range(num_textures):
                with bld.Index(i):
                    mode, info = add_attr_cluster("tex", pipeline.c_tex[i])
                    ia_tex_mode.append(mode)
                    ia_tex_info.append(info)
            ia_col_mode, ia_col_info = add_attr_cluster("col", pipeline.c_col)

            m.d.comb += [
                pipeline.c_pos.mode.eq(ia_pos_mode.f.data),
                pipeline.c_pos.info.eq(ia_pos_info.f.data),
                pipeline.c_norm.mode.eq(ia_norm_mode.f.data),
                pipeline.c_norm.info.eq(ia_norm_info.f.data),
                pipeline.c_col.mode.eq(ia_col_mode.f.data),
                pipeline.c_col.info.eq(ia_col_info.f.data),
            ]
            for i in range(num_textures):
                m.d.comb += [
                    pipeline.c_tex[i].mode.eq(ia_tex_mode[i].f.data),
                    pipeline.c_tex[i].info.eq(ia_tex_info[i].f.data),
                ]

        with bld.Cluster("vtx_xf"):
            vt_enabled = bld.add("enabled", RWReg(pipeline.vt_enabled.shape()))

            pos_mv_regs = bld.add(
                "position_mv",
                RWReg(pipeline.position_mv.shape()),
            )
            pos_p_regs = bld.add(
                "position_p",
                RWReg(pipeline.position_p.shape()),
            )
            norm_mv_regs = bld.add(
                "normal_mv_inv_t",
                RWReg(pipeline.normal_mv_inv_t.shape()),
            )
            tex_xf_regs = []
            for t in range(num_textures):
                for i in range(len(pipeline.texture_transforms[t])):
                    tex_xf_regs.append(
                        bld.add(
                            "texture_transform",
                            RWReg(pipeline.texture_transforms[t].shape()),
                        )
                    )
            m.d.comb += [
                pipeline.vt_enabled.eq(vt_enabled.f.data),
                pipeline.position_mv.eq(pos_mv_regs.f.data),
                pipeline.position_p.eq(pos_p_regs.f.data),
                pipeline.normal_mv_inv_t.eq(norm_mv_regs.f.data),
            ]
            for t in range(num_textures):
                m.d.comb += pipeline.texture_transforms[t].eq(tex_xf_regs[t].f.data)

        with bld.Cluster("vtx_sh"):
            with bld.Cluster("material"):
                mat_amb = bld.add(
                    "ambient",
                    RWReg(pipeline.material.ambient.shape()),
                )
                mat_dif = bld.add(
                    "diffuse",
                    RWReg(pipeline.material.diffuse.shape()),
                )
                mat_spe = bld.add(
                    "specular",
                    RWReg(pipeline.material.specular.shape()),
                )
                mat_shine = bld.add("shininess", RWReg(FixedPoint_mem))

            light_regs = []
            for l_idx in range(num_lights):
                with bld.Cluster(f"{l_idx}"):
                    with bld.Cluster("light"):
                        light = pipeline.lights[l_idx]
                        pos = bld.add(
                            "position",
                            RWReg(light.position.shape()),
                        )
                        amb = bld.add(
                            "ambient",
                            RWReg(light.ambient.shape()),
                        )
                        dif = bld.add(
                            "diffuse",
                            RWReg(light.diffuse.shape()),
                        )
                        spe = bld.add(
                            "specular",
                            RWReg(light.specular.shape()),
                        )
                        light_regs.append((pos, amb, dif, spe))

            m.d.comb += [
                pipeline.material.ambient.eq(mat_amb.f.data),
                pipeline.material.diffuse.eq(mat_dif.f.data),
                pipeline.material.specular.eq(mat_spe.f.data),
                pipeline.material.shininess.eq(mat_shine.f.data),
            ]
            for idx, (pos, amb, dif, spe) in enumerate(light_regs):
                m.d.comb += [
                    pipeline.lights[idx].position.eq(pos.f.data),
                    pipeline.lights[idx].ambient.eq(amb.f.data),
                    pipeline.lights[idx].diffuse.eq(dif.f.data),
                    pipeline.lights[idx].specular.eq(spe.f.data),
                ]

        with bld.Cluster("prim"):
            prim_type = bld.add("type", RWReg(pipeline.pa_conf.type.shape()))
            prim_cull = bld.add("cull", RWReg(pipeline.pa_conf.cull.shape()))
            prim_wind = bld.add("winding", RWReg(pipeline.pa_conf.winding.shape()))

            m.d.comb += [
                pipeline.pa_conf.type.eq(prim_type.f.data),
                pipeline.pa_conf.cull.eq(prim_cull.f.data),
                pipeline.pa_conf.winding.eq(prim_wind.f.data),
            ]

        with bld.Cluster("fb"):
            fb_width = bld.add("width", RWReg(pipeline.fb_info.width.shape()))
            fb_height = bld.add("height", RWReg(pipeline.fb_info.height.shape()))
            fb_vx = bld.add("viewport_x", RWReg(FixedPoint_mem))
            fb_vy = bld.add("viewport_y", RWReg(FixedPoint_mem))
            fb_vw = bld.add("viewport_width", RWReg(FixedPoint_mem))
            fb_vh = bld.add("viewport_height", RWReg(FixedPoint_mem))
            fb_min_d = bld.add("viewport_min_depth", RWReg(FixedPoint_mem))
            fb_max_d = bld.add("viewport_max_depth", RWReg(FixedPoint_mem))
            fb_sc_x = bld.add(
                "scissor_offset_x", RWReg(pipeline.fb_info.scissor_offset_x.shape())
            )
            fb_sc_y = bld.add(
                "scissor_offset_y", RWReg(pipeline.fb_info.scissor_offset_y.shape())
            )
            fb_sc_w = bld.add(
                "scissor_width", RWReg(pipeline.fb_info.scissor_width.shape())
            )
            fb_sc_h = bld.add(
                "scissor_height", RWReg(pipeline.fb_info.scissor_height.shape())
            )
            fb_color_addr = bld.add(
                "color_address", RWReg(pipeline.fb_info.color_address.shape())
            )
            fb_color_pitch = bld.add(
                "color_pitch", RWReg(pipeline.fb_info.color_pitch.shape())
            )
            fb_depthstencil_addr = bld.add(
                "depthstencil_address",
                RWReg(pipeline.fb_info.depthstencil_address.shape()),
            )
            fb_depthstencil_pitch = bld.add(
                "depthstencil_pitch", RWReg(pipeline.fb_info.depthstencil_pitch.shape())
            )

            m.d.comb += [
                pipeline.fb_info.width.eq(fb_width.f.data),
                pipeline.fb_info.height.eq(fb_height.f.data),
                pipeline.fb_info.viewport_x.eq(
                    FixedPoint_mem(fb_vx.f.data).saturate(
                        pipeline.fb_info.viewport_x.shape()
                    )
                ),
                pipeline.fb_info.viewport_y.eq(
                    FixedPoint_mem(fb_vy.f.data).saturate(
                        pipeline.fb_info.viewport_y.shape()
                    )
                ),
                pipeline.fb_info.viewport_width.eq(
                    FixedPoint_mem(fb_vw.f.data).saturate(
                        pipeline.fb_info.viewport_width.shape()
                    )
                ),
                pipeline.fb_info.viewport_height.eq(
                    FixedPoint_mem(fb_vh.f.data).saturate(
                        pipeline.fb_info.viewport_height.shape()
                    )
                ),
                pipeline.fb_info.viewport_min_depth.eq(
                    FixedPoint_mem(fb_min_d.f.data).saturate(
                        pipeline.fb_info.viewport_min_depth.shape()
                    )
                ),
                pipeline.fb_info.viewport_max_depth.eq(
                    FixedPoint_mem(fb_max_d.f.data).saturate(
                        pipeline.fb_info.viewport_max_depth.shape()
                    )
                ),
                pipeline.fb_info.scissor_offset_x.eq(fb_sc_x.f.data),
                pipeline.fb_info.scissor_offset_y.eq(fb_sc_y.f.data),
                pipeline.fb_info.scissor_width.eq(fb_sc_w.f.data),
                pipeline.fb_info.scissor_height.eq(fb_sc_h.f.data),
                pipeline.fb_info.color_address.eq(fb_color_addr.f.data),
                pipeline.fb_info.color_pitch.eq(fb_color_pitch.f.data),
                pipeline.fb_info.depthstencil_address.eq(fb_depthstencil_addr.f.data),
                pipeline.fb_info.depthstencil_pitch.eq(fb_depthstencil_pitch.f.data),
            ]

        with bld.Cluster("ds"):
            stencil_front = bld.add(
                "stencil_front",
                RWReg(pipeline.stencil_conf_front.shape()),
            )
            stencil_back = bld.add(
                "stencil_back",
                RWReg(pipeline.stencil_conf_back.shape()),
            )
            depth_cfg = bld.add(
                "depth",
                RWReg(pipeline.depth_conf.shape()),
            )

            m.d.comb += [
                pipeline.stencil_conf_front.eq(stencil_front.f.data),
                pipeline.stencil_conf_back.eq(stencil_back.f.data),
                pipeline.depth_conf.eq(depth_cfg.f.data),
            ]

        with bld.Cluster("blend"):
            blend_cfg = bld.add("config", RWReg(pipeline.blend_conf.shape()))
            m.d.comb += pipeline.blend_conf.eq(blend_cfg.f.data)

        # Status register exposing pipeline readiness
        ready_reg = bld.add(
            "ready", csr.Register(csr.Field(csr.action.R, unsigned(1)), "r")
        )
        m.d.comb += ready_reg.f.r_data.eq(pipeline.ready)

        ready_components = bld.add(
            "ready_components", csr.Register(csr.Field(csr.action.R, unsigned(4)), "r")
        )
        m.d.comb += ready_components.f.r_data.eq(pipeline.ready_components)

        ready_vec = bld.add(
            "ready_vec", csr.Register(csr.Field(csr.action.R, unsigned(32)), "r")
        )
        m.d.comb += ready_vec.f.r_data.eq(pipeline.ready_vec)

        m.submodules.csr_bus = csr_bus = csr.Bridge(bld.as_memory_map())
        m.submodules.csr_bridge = csr_bridge = WishboneCSRBridge(
            csr_bus.bus, data_width=32
        )

        wiring.connect(m, wiring.flipped(self.wb_csr), csr_bridge.wb_bus)
        self.wb_csr.memory_map = csr_bridge.wb_bus.memory_map

        wiring.connect(m, wiring.flipped(self.wb_index), pipeline.wb_index)
        wiring.connect(m, wiring.flipped(self.wb_vertex), pipeline.wb_vertex)
        wiring.connect(
            m, wiring.flipped(self.wb_depthstencil), pipeline.wb_depthstencil
        )
        wiring.connect(m, wiring.flipped(self.wb_color), pipeline.wb_color)
        m.d.comb += self.ready.eq(pipeline.ready)

        return m


class GraphicsPipelineAvalonCSR(wiring.Component):
    """Graphics pipeline with CSR and Avalon-MM interfaces.

    Wraps GraphicsPipelineCSR and bridges all Wishbone buses (memory + CSR) to Avalon-MM.
    """

    def __init__(self):
        self._pipeline_csr = GraphicsPipelineCSR()

        self._bridge_index = avl.WishboneMasterToAvalonBridge(
            self._pipeline_csr.wb_index
        )
        self._bridge_vertex = avl.WishboneMasterToAvalonBridge(
            self._pipeline_csr.wb_vertex
        )
        self._bridge_depthstencil = avl.WishboneMasterToAvalonBridge(
            self._pipeline_csr.wb_depthstencil
        )
        self._bridge_color = avl.WishboneMasterToAvalonBridge(
            self._pipeline_csr.wb_color
        )

        self._bridge_csr = avl.WishboneSlaveToAvalonBridge(self._pipeline_csr.wb_csr)

        super().__init__(
            {
                "ready": Out(1),
                "avl_index": Out(self._bridge_index.avl_bus.signature),
                "avl_vertex": Out(self._bridge_vertex.avl_bus.signature),
                "avl_depthstencil": Out(self._bridge_depthstencil.avl_bus.signature),
                "avl_color": Out(self._bridge_color.avl_bus.signature),
                "avl_csr": Out(self._bridge_csr.avl_bus.signature),
            }
        )

    def elaborate(self, platform):
        m = Module()

        m.submodules.pipeline = pipeline = self._pipeline_csr

        m.submodules.bridge_index = bridge_index = self._bridge_index
        m.submodules.bridge_vertex = bridge_vertex = self._bridge_vertex
        m.submodules.bridge_depthstencil = bridge_depthstencil = (
            self._bridge_depthstencil
        )
        m.submodules.bridge_color = bridge_color = self._bridge_color
        m.submodules.bridge_csr = bridge_csr = self._bridge_csr

        # Connect memory buses
        wiring.connect(m, bridge_index.avl_bus, wiring.flipped(self.avl_index))
        wiring.connect(m, bridge_vertex.avl_bus, wiring.flipped(self.avl_vertex))
        wiring.connect(
            m, bridge_depthstencil.avl_bus, wiring.flipped(self.avl_depthstencil)
        )
        wiring.connect(m, bridge_color.avl_bus, wiring.flipped(self.avl_color))

        # Connect CSR bus
        wiring.connect(m, bridge_csr.avl_bus, wiring.flipped(self.avl_csr))

        # Connect ready signal
        m.d.comb += self.ready.eq(pipeline.ready)

        return m


__all__ = ["GraphicsPipeline", "GraphicsPipelineCSR", "GraphicsPipelineAvalonCSR"]

if __name__ == "__main__":
    import json

    from amaranth.back import verilog

    # Generate Verilog
    gp_avl = GraphicsPipelineAvalonCSR()
    with open("graphics_pipeline_avalon_csr.sv", "w") as f:
        f.write(verilog.convert(gp_avl))

    mem_map: MemoryMap = gp_avl._pipeline_csr.wb_csr.memory_map

    regs_dict = {}

    for ri in mem_map.all_resources():
        ratio = ri.width // 8
        byte_addr = ri.start * ratio
        byte_size = (ri.end - ri.start) * ratio

        path_splat = []
        for p in ri.path:
            path_splat += [*p]

        d = regs_dict
        for p in path_splat:
            if p not in d:
                d[p] = {}
            d = d[p]
        d.update(
            {
                "address": byte_addr,
                "size": byte_size,
            }
        )

    # Convert memory map to JSON
    csr_map = {
        "address_width": mem_map.addr_width,
        "data_width": 32,
        "granularity": 8,
        "registers": regs_dict,
    }

    with open("graphics_pipeline_csr_map.json", "w") as f:
        json.dump(csr_map, f, indent=2)
