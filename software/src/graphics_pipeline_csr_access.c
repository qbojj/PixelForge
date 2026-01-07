#include "graphics_pipeline_csr_access.h"

/* =============================
 * Index Generator
 * ============================= */
void pf_csr_set_idx(volatile uint8_t *base, const pixelforge_idx_config_t *cfg) {
    pf_csr_write32(base, PIXELFORGE_CSR_IDX_ADDRESS, cfg->address);
    pf_csr_write32(base, PIXELFORGE_CSR_IDX_COUNT, cfg->count);
    pf_csr_write32(base, PIXELFORGE_CSR_IDX_KIND, (uint32_t)cfg->kind);
}

void pf_csr_get_idx(volatile uint8_t *base, pixelforge_idx_config_t *cfg) {
    cfg->address = pf_csr_read32(base, PIXELFORGE_CSR_IDX_ADDRESS);
    cfg->count = pf_csr_read32(base, PIXELFORGE_CSR_IDX_COUNT);
    cfg->kind = (uint8_t)pf_csr_read32(base, PIXELFORGE_CSR_IDX_KIND);
}

void pf_csr_start(volatile uint8_t *base) {
    pf_csr_write32(base, PIXELFORGE_CSR_IDX_START, 1u);
}

/* =============================
 * Topology
 * ============================= */
void pf_csr_set_topology(volatile uint8_t *base, const pixelforge_topo_config_t *cfg) {
    pf_csr_write32(base, PIXELFORGE_CSR_TOPO_INPUT_TOPOLOGY, (uint32_t)cfg->input_topology);
    pf_csr_write32(base, PIXELFORGE_CSR_TOPO_PRIMITIVE_RESTART_ENABLE, (uint32_t)cfg->primitive_restart_enable);
    pf_csr_write32(base, PIXELFORGE_CSR_TOPO_PRIMITIVE_RESTART_INDEX, cfg->primitive_restart_index);
    pf_csr_write32(base, PIXELFORGE_CSR_TOPO_BASE_VERTEX, cfg->base_vertex);
}

void pf_csr_get_topology(volatile uint8_t *base, pixelforge_topo_config_t *cfg) {
    cfg->input_topology = (uint8_t)pf_csr_read32(base, PIXELFORGE_CSR_TOPO_INPUT_TOPOLOGY);
    cfg->primitive_restart_enable = (uint8_t)pf_csr_read32(base, PIXELFORGE_CSR_TOPO_PRIMITIVE_RESTART_ENABLE);
    cfg->primitive_restart_index = pf_csr_read32(base, PIXELFORGE_CSR_TOPO_PRIMITIVE_RESTART_INDEX);
    cfg->base_vertex = pf_csr_read32(base, PIXELFORGE_CSR_TOPO_BASE_VERTEX);
}

/* Helpers for attribute unions */
static inline void pf_write_attr_constant(volatile uint8_t *base, uint32_t base_off, const pixelforge_input_attr_t *attr) {
    pf_csr_write32(base, base_off,     (uint32_t)attr->info.constant_value.value[0]);
    pf_csr_write32(base, base_off+4,   (uint32_t)attr->info.constant_value.value[1]);
    pf_csr_write32(base, base_off+8,   (uint32_t)attr->info.constant_value.value[2]);
    pf_csr_write32(base, base_off+12,  (uint32_t)attr->info.constant_value.value[3]);
}
static inline void pf_read_attr_constant(volatile uint8_t *base, uint32_t base_off, pixelforge_input_attr_t *attr) {
    attr->info.constant_value.value[0] = (int32_t)pf_csr_read32(base, base_off);
    attr->info.constant_value.value[1] = (int32_t)pf_csr_read32(base, base_off+4);
    attr->info.constant_value.value[2] = (int32_t)pf_csr_read32(base, base_off+8);
    attr->info.constant_value.value[3] = (int32_t)pf_csr_read32(base, base_off+12);
}
static inline void pf_write_attr_per_vertex(volatile uint8_t *base, uint32_t base_off, const pixelforge_input_attr_t *attr) {
    pf_csr_write32(base, base_off,     attr->info.per_vertex.address);
    pf_csr_write32(base, base_off+4,   (uint32_t)attr->info.per_vertex.stride);
}
static inline void pf_read_attr_per_vertex(volatile uint8_t *base, uint32_t base_off, pixelforge_input_attr_t *attr) {
    attr->info.per_vertex.address = pf_csr_read32(base, base_off);
    attr->info.per_vertex.stride = (uint16_t)pf_csr_read32(base, base_off+4);
}

/* =============================
 * Input Attributes
 * ============================= */
void pf_csr_set_attr_position(volatile uint8_t *base, const pixelforge_input_attr_t *attr) {
    pf_csr_write32(base, PIXELFORGE_CSR_IA_POS_MODE, (uint32_t)attr->mode);
    if (attr->mode == PIXELFORGE_ATTR_CONSTANT) {
        pf_write_attr_constant(base, PIXELFORGE_CSR_IA_POS_INFO, attr);
    } else {
        pf_write_attr_per_vertex(base, PIXELFORGE_CSR_IA_POS_INFO, attr);
    }
}
void pf_csr_get_attr_position(volatile uint8_t *base, pixelforge_input_attr_t *attr) {
    attr->mode = pf_csr_read32(base, PIXELFORGE_CSR_IA_POS_MODE);
    if (attr->mode == PIXELFORGE_ATTR_CONSTANT) {
        pf_read_attr_constant(base, PIXELFORGE_CSR_IA_POS_INFO, attr);
    } else {
        pf_read_attr_per_vertex(base, PIXELFORGE_CSR_IA_POS_INFO, attr);
    }
}

void pf_csr_set_attr_normal(volatile uint8_t *base, const pixelforge_input_attr_t *attr) {
    pf_csr_write32(base, PIXELFORGE_CSR_IA_NORM_MODE, (uint32_t)attr->mode);
    if (attr->mode == PIXELFORGE_ATTR_CONSTANT) {
        pf_write_attr_constant(base, PIXELFORGE_CSR_IA_NORM_INFO, attr);
    } else {
        pf_write_attr_per_vertex(base, PIXELFORGE_CSR_IA_NORM_INFO, attr);
    }
}
void pf_csr_get_attr_normal(volatile uint8_t *base, pixelforge_input_attr_t *attr) {
    attr->mode = pf_csr_read32(base, PIXELFORGE_CSR_IA_NORM_MODE);
    if (attr->mode == PIXELFORGE_ATTR_CONSTANT) {
        pf_read_attr_constant(base, PIXELFORGE_CSR_IA_NORM_INFO, attr);
    } else {
        pf_read_attr_per_vertex(base, PIXELFORGE_CSR_IA_NORM_INFO, attr);
    }
}

void pf_csr_set_attr_color(volatile uint8_t *base, const pixelforge_input_attr_t *attr) {
    pf_csr_write32(base, PIXELFORGE_CSR_IA_COL_MODE, (uint32_t)attr->mode);
    if (attr->mode == PIXELFORGE_ATTR_CONSTANT) {
        pf_write_attr_constant(base, PIXELFORGE_CSR_IA_COL_INFO, attr);
    } else {
        pf_write_attr_per_vertex(base, PIXELFORGE_CSR_IA_COL_INFO, attr);
    }
}
void pf_csr_get_attr_color(volatile uint8_t *base, pixelforge_input_attr_t *attr) {
    attr->mode = pf_csr_read32(base, PIXELFORGE_CSR_IA_COL_MODE);
    if (attr->mode == PIXELFORGE_ATTR_CONSTANT) {
        pf_read_attr_constant(base, PIXELFORGE_CSR_IA_COL_INFO, attr);
    } else {
        pf_read_attr_per_vertex(base, PIXELFORGE_CSR_IA_COL_INFO, attr);
    }
}

/* =============================
 * Vertex Transform
 * ============================= */
void pf_csr_set_vtx_xf(volatile uint8_t *base, const pixelforge_vtx_xf_config_t *cfg) {
    pf_csr_write32(base, PIXELFORGE_CSR_VTX_XF_ENABLED_NORMAL, (uint32_t)cfg->enabled.normal_enable);
    for (int i = 0; i < 16; ++i) pf_csr_write32(base, PIXELFORGE_CSR_VTX_XF_POSITION_MV + i*4, (uint32_t)cfg->position_mv[i]);
    for (int i = 0; i < 16; ++i) pf_csr_write32(base, PIXELFORGE_CSR_VTX_XF_POSITION_P  + i*4, (uint32_t)cfg->position_p[i]);
    for (int i = 0; i < 9; ++i) pf_csr_write32(base, PIXELFORGE_CSR_VTX_XF_NORMAL_MV_INV_T + i*4, (uint32_t)cfg->normal_mv_inv_t[i]);
}

void pf_csr_get_vtx_xf(volatile uint8_t *base, pixelforge_vtx_xf_config_t *cfg) {
    cfg->enabled.normal_enable = (uint8_t)pf_csr_read32(base, PIXELFORGE_CSR_VTX_XF_ENABLED_NORMAL);
    for (int i = 0; i < 16; ++i) cfg->position_mv[i] = (int32_t)pf_csr_read32(base, PIXELFORGE_CSR_VTX_XF_POSITION_MV + i*4);
    for (int i = 0; i < 16; ++i) cfg->position_p[i]  = (int32_t)pf_csr_read32(base, PIXELFORGE_CSR_VTX_XF_POSITION_P  + i*4);
    for (int i = 0; i < 9; ++i) cfg->normal_mv_inv_t[i] = (int32_t)pf_csr_read32(base, PIXELFORGE_CSR_VTX_XF_NORMAL_MV_INV_T + i*4);
}

/* =============================
 * Material & Light0
 * ============================= */
void pf_csr_set_material(volatile uint8_t *base, const pixelforge_material_t *mat) {
    for (int i = 0; i < 3; ++i) pf_csr_write32(base, PIXELFORGE_CSR_VTX_SH_MATERIAL_AMBIENT + i*4, (uint32_t)mat->ambient[i]);
    for (int i = 0; i < 3; ++i) pf_csr_write32(base, PIXELFORGE_CSR_VTX_SH_MATERIAL_DIFFUSE + i*4, (uint32_t)mat->diffuse[i]);
    for (int i = 0; i < 3; ++i) pf_csr_write32(base, PIXELFORGE_CSR_VTX_SH_MATERIAL_SPECULAR + i*4, (uint32_t)mat->specular[i]);
    pf_csr_write32(base, PIXELFORGE_CSR_VTX_SH_MATERIAL_SHININESS, (uint32_t)mat->shininess);
}
void pf_csr_get_material(volatile uint8_t *base, pixelforge_material_t *mat) {
    for (int i = 0; i < 3; ++i) mat->ambient[i]  = (int32_t)pf_csr_read32(base, PIXELFORGE_CSR_VTX_SH_MATERIAL_AMBIENT + i*4);
    for (int i = 0; i < 3; ++i) mat->diffuse[i]  = (int32_t)pf_csr_read32(base, PIXELFORGE_CSR_VTX_SH_MATERIAL_DIFFUSE + i*4);
    for (int i = 0; i < 3; ++i) mat->specular[i] = (int32_t)pf_csr_read32(base, PIXELFORGE_CSR_VTX_SH_MATERIAL_SPECULAR + i*4);
    mat->shininess = (int32_t)pf_csr_read32(base, PIXELFORGE_CSR_VTX_SH_MATERIAL_SHININESS);
}

void pf_csr_set_light0(volatile uint8_t *base, const pixelforge_light_t *lit) {
    for (int i = 0; i < 4; ++i) pf_csr_write32(base, PIXELFORGE_CSR_VTX_SH_0_LIGHT_POSITION + i*4, (uint32_t)lit->position[i]);
    for (int i = 0; i < 3; ++i) pf_csr_write32(base, PIXELFORGE_CSR_VTX_SH_0_LIGHT_AMBIENT  + i*4, (uint32_t)lit->ambient[i]);
    for (int i = 0; i < 3; ++i) pf_csr_write32(base, PIXELFORGE_CSR_VTX_SH_0_LIGHT_DIFFUSE  + i*4, (uint32_t)lit->diffuse[i]);
    for (int i = 0; i < 3; ++i) pf_csr_write32(base, PIXELFORGE_CSR_VTX_SH_0_LIGHT_SPECULAR + i*4, (uint32_t)lit->specular[i]);
}
void pf_csr_get_light0(volatile uint8_t *base, pixelforge_light_t *lit) {
    for (int i = 0; i < 4; ++i) lit->position[i] = (int32_t)pf_csr_read32(base, PIXELFORGE_CSR_VTX_SH_0_LIGHT_POSITION + i*4);
    for (int i = 0; i < 3; ++i) lit->ambient[i]  = (int32_t)pf_csr_read32(base, PIXELFORGE_CSR_VTX_SH_0_LIGHT_AMBIENT  + i*4);
    for (int i = 0; i < 3; ++i) lit->diffuse[i]  = (int32_t)pf_csr_read32(base, PIXELFORGE_CSR_VTX_SH_0_LIGHT_DIFFUSE  + i*4);
    for (int i = 0; i < 3; ++i) lit->specular[i] = (int32_t)pf_csr_read32(base, PIXELFORGE_CSR_VTX_SH_0_LIGHT_SPECULAR + i*4);
}

/* =============================
 * Primitive Assembly
 * ============================= */
void pf_csr_set_prim(volatile uint8_t *base, const pixelforge_prim_config_t *cfg) {
    pf_csr_write32(base, PIXELFORGE_CSR_PRIM_TYPE, (uint32_t)cfg->type);
    pf_csr_write32(base, PIXELFORGE_CSR_PRIM_CULL, (uint32_t)cfg->cull);
    pf_csr_write32(base, PIXELFORGE_CSR_PRIM_WINDING, (uint32_t)cfg->winding);
}
void pf_csr_get_prim(volatile uint8_t *base, pixelforge_prim_config_t *cfg) {
    cfg->type = (uint8_t)pf_csr_read32(base, PIXELFORGE_CSR_PRIM_TYPE);
    cfg->cull = (uint8_t)pf_csr_read32(base, PIXELFORGE_CSR_PRIM_CULL);
    cfg->winding = (uint8_t)pf_csr_read32(base, PIXELFORGE_CSR_PRIM_WINDING);
}

/* =============================
 * Framebuffer
 * ============================= */
void pf_csr_set_fb(volatile uint8_t *base, const pixelforge_framebuffer_config_t *cfg) {
    pf_csr_write32(base, PIXELFORGE_CSR_FB_WIDTH,  (uint32_t)cfg->width);
    pf_csr_write32(base, PIXELFORGE_CSR_FB_HEIGHT, (uint32_t)cfg->height);
    pf_csr_write32(base, PIXELFORGE_CSR_FB_VIEWPORT_X,        (uint32_t)cfg->viewport_x);
    pf_csr_write32(base, PIXELFORGE_CSR_FB_VIEWPORT_Y,        (uint32_t)cfg->viewport_y);
    pf_csr_write32(base, PIXELFORGE_CSR_FB_VIEWPORT_WIDTH,    (uint32_t)cfg->viewport_width);
    pf_csr_write32(base, PIXELFORGE_CSR_FB_VIEWPORT_HEIGHT,   (uint32_t)cfg->viewport_height);
    pf_csr_write32(base, PIXELFORGE_CSR_FB_VIEWPORT_MIN_DEPTH,(uint32_t)cfg->viewport_min_depth);
    pf_csr_write32(base, PIXELFORGE_CSR_FB_VIEWPORT_MAX_DEPTH,(uint32_t)cfg->viewport_max_depth);
    pf_csr_write32(base, PIXELFORGE_CSR_FB_SCISSOR_OFFSET_X,  (uint32_t)cfg->scissor_offset_x);
    pf_csr_write32(base, PIXELFORGE_CSR_FB_SCISSOR_OFFSET_Y,  (uint32_t)cfg->scissor_offset_y);
    pf_csr_write32(base, PIXELFORGE_CSR_FB_SCISSOR_WIDTH,     (uint32_t)cfg->scissor_width);
    pf_csr_write32(base, PIXELFORGE_CSR_FB_SCISSOR_HEIGHT,    (uint32_t)cfg->scissor_height);
    pf_csr_write32(base, PIXELFORGE_CSR_FB_COLOR_ADDRESS, cfg->color_address);
    pf_csr_write32(base, PIXELFORGE_CSR_FB_COLOR_PITCH,   (uint32_t)cfg->color_pitch);
    pf_csr_write32(base, PIXELFORGE_CSR_FB_DEPTH_ADDRESS, cfg->depth_address);
    pf_csr_write32(base, PIXELFORGE_CSR_FB_DEPTH_PITCH,   (uint32_t)cfg->depth_pitch);
    pf_csr_write32(base, PIXELFORGE_CSR_FB_STENCIL_ADDRESS, cfg->stencil_address);
    pf_csr_write32(base, PIXELFORGE_CSR_FB_STENCIL_PITCH,   (uint32_t)cfg->stencil_pitch);
}
void pf_csr_get_fb(volatile uint8_t *base, pixelforge_framebuffer_config_t *cfg) {
    cfg->width  = (uint16_t)pf_csr_read32(base, PIXELFORGE_CSR_FB_WIDTH);
    cfg->height = (uint16_t)pf_csr_read32(base, PIXELFORGE_CSR_FB_HEIGHT);
    cfg->viewport_x        = (int32_t)pf_csr_read32(base, PIXELFORGE_CSR_FB_VIEWPORT_X);
    cfg->viewport_y        = (int32_t)pf_csr_read32(base, PIXELFORGE_CSR_FB_VIEWPORT_Y);
    cfg->viewport_width    = (int32_t)pf_csr_read32(base, PIXELFORGE_CSR_FB_VIEWPORT_WIDTH);
    cfg->viewport_height   = (int32_t)pf_csr_read32(base, PIXELFORGE_CSR_FB_VIEWPORT_HEIGHT);
    cfg->viewport_min_depth= (int32_t)pf_csr_read32(base, PIXELFORGE_CSR_FB_VIEWPORT_MIN_DEPTH);
    cfg->viewport_max_depth= (int32_t)pf_csr_read32(base, PIXELFORGE_CSR_FB_VIEWPORT_MAX_DEPTH);
    cfg->scissor_offset_x  = (int32_t)pf_csr_read32(base, PIXELFORGE_CSR_FB_SCISSOR_OFFSET_X);
    cfg->scissor_offset_y  = (int32_t)pf_csr_read32(base, PIXELFORGE_CSR_FB_SCISSOR_OFFSET_Y);
    cfg->scissor_width     = (uint32_t)pf_csr_read32(base, PIXELFORGE_CSR_FB_SCISSOR_WIDTH);
    cfg->scissor_height    = (uint32_t)pf_csr_read32(base, PIXELFORGE_CSR_FB_SCISSOR_HEIGHT);
    cfg->color_address     = pf_csr_read32(base, PIXELFORGE_CSR_FB_COLOR_ADDRESS);
    cfg->color_pitch       = (uint16_t)pf_csr_read32(base, PIXELFORGE_CSR_FB_COLOR_PITCH);
    cfg->depth_address     = pf_csr_read32(base, PIXELFORGE_CSR_FB_DEPTH_ADDRESS);
    cfg->depth_pitch       = (uint16_t)pf_csr_read32(base, PIXELFORGE_CSR_FB_DEPTH_PITCH);
    cfg->stencil_address   = pf_csr_read32(base, PIXELFORGE_CSR_FB_STENCIL_ADDRESS);
    cfg->stencil_pitch     = (uint16_t)pf_csr_read32(base, PIXELFORGE_CSR_FB_STENCIL_PITCH);
}

/* =============================
 * Depth/Stencil & Blend helpers
 * ============================= */
static uint32_t pf_pack_stencil_ops(const pixelforge_stencil_op_config_t *c) {
    uint32_t w = 0;
    w |= ((uint32_t)c->compare_op & 0x7) << 0;
    w |= ((uint32_t)c->pass_op    & 0x7) << 3;
    w |= ((uint32_t)c->fail_op    & 0x7) << 6;
    w |= ((uint32_t)c->depth_fail_op & 0x7) << 9;
    return w;
}
static uint32_t pf_pack_stencil_masks(const pixelforge_stencil_op_config_t *c) {
    uint32_t w = 0;
    w |= ((uint32_t)c->reference & 0xFF) << 16;
    w |= ((uint32_t)c->mask      & 0xFF) << 24;
    /* write_mask in another 32-bit reg if needed; here we assume single 32-bit */
    return w;
}
static void pf_unpack_stencil_ops(uint32_t w, pixelforge_stencil_op_config_t *c) {
    c->compare_op    = (uint8_t)((w >> 0) & 0x7);
    c->pass_op       = (uint8_t)((w >> 3) & 0x7);
    c->fail_op       = (uint8_t)((w >> 6) & 0x7);
    c->depth_fail_op = (uint8_t)((w >> 9) & 0x7);
}
static void pf_unpack_stencil_masks(uint32_t w, pixelforge_stencil_op_config_t *c) {
    c->reference = (uint8_t)((w >> 16) & 0xFF);
    c->mask      = (uint8_t)((w >> 24) & 0xFF);
}

static uint32_t pf_pack_depth_test(const pixelforge_depth_test_config_t *c) {
    uint32_t w = 0;
    w |= ((uint32_t)c->test_enabled  & 0x1) << 0;
    w |= ((uint32_t)c->write_enabled & 0x1) << 1;
    w |= ((uint32_t)c->compare_op    & 0x7) << 2;
    return w;
}
static void pf_unpack_depth_test(uint32_t w, pixelforge_depth_test_config_t *c) {
    c->test_enabled  = (uint8_t)((w >> 0) & 0x1);
    c->write_enabled = (uint8_t)((w >> 1) & 0x1);
    c->compare_op    = (uint8_t)((w >> 2) & 0x7);
}

static uint32_t pf_pack_blend_config(const pixelforge_blend_config_t *c) {
    uint32_t w = 0;
    w |= ((uint32_t)c->src_factor   & 0xF) << 0;
    w |= ((uint32_t)c->dst_factor   & 0xF) << 4;
    w |= ((uint32_t)c->src_a_factor & 0xF) << 8;
    w |= ((uint32_t)c->dst_a_factor & 0xF) << 12;
    w |= ((uint32_t)c->enabled      & 0x1) << 16;
    w |= ((uint32_t)c->blend_op     & 0x7) << 17;
    w |= ((uint32_t)c->blend_a_op   & 0x7) << 20;
    w |= ((uint32_t)c->color_write_mask & 0xF) << 24;
    return w;
}
static void pf_unpack_blend_config(uint32_t w, pixelforge_blend_config_t *c) {
    c->src_factor   = (uint32_t)((w >> 0) & 0xF);
    c->dst_factor   = (uint32_t)((w >> 4) & 0xF);
    c->src_a_factor = (uint32_t)((w >> 8) & 0xF);
    c->dst_a_factor = (uint32_t)((w >> 12) & 0xF);
    c->enabled      = (uint32_t)((w >> 16) & 0x1);
    c->blend_op     = (uint32_t)((w >> 17) & 0x7);
    c->blend_a_op   = (uint32_t)((w >> 20) & 0x7);
    c->color_write_mask = (uint32_t)((w >> 24) & 0xF);
}

void pf_csr_set_stencil_front(volatile uint8_t *base, const pixelforge_stencil_op_config_t *c) {
    uint32_t ops = pf_pack_stencil_ops(c);
    uint32_t masks = pf_pack_stencil_masks(c);
    pf_csr_write32(base, PIXELFORGE_CSR_DS_STENCIL_FRONT, ops);
    pf_csr_write32(base, PIXELFORGE_CSR_DS_STENCIL_FRONT + 4, masks);
}
void pf_csr_get_stencil_front(volatile uint8_t *base, pixelforge_stencil_op_config_t *c) {
    uint32_t ops   = pf_csr_read32(base, PIXELFORGE_CSR_DS_STENCIL_FRONT);
    uint32_t masks = pf_csr_read32(base, PIXELFORGE_CSR_DS_STENCIL_FRONT + 4);
    pf_unpack_stencil_ops(ops, c);
    pf_unpack_stencil_masks(masks, c);
}

void pf_csr_set_stencil_back(volatile uint8_t *base, const pixelforge_stencil_op_config_t *c) {
    uint32_t ops = pf_pack_stencil_ops(c);
    uint32_t masks = pf_pack_stencil_masks(c);
    pf_csr_write32(base, PIXELFORGE_CSR_DS_STENCIL_BACK, ops);
    pf_csr_write32(base, PIXELFORGE_CSR_DS_STENCIL_BACK + 4, masks);
}
void pf_csr_get_stencil_back(volatile uint8_t *base, pixelforge_stencil_op_config_t *c) {
    uint32_t ops   = pf_csr_read32(base, PIXELFORGE_CSR_DS_STENCIL_BACK);
    uint32_t masks = pf_csr_read32(base, PIXELFORGE_CSR_DS_STENCIL_BACK + 4);
    pf_unpack_stencil_ops(ops, c);
    pf_unpack_stencil_masks(masks, c);
}

void pf_csr_set_depth(volatile uint8_t *base, const pixelforge_depth_test_config_t *c) {
    uint32_t w = pf_pack_depth_test(c);
    pf_csr_write32(base, PIXELFORGE_CSR_DS_DEPTH, w);
}
void pf_csr_get_depth(volatile uint8_t *base, pixelforge_depth_test_config_t *c) {
    uint32_t w = pf_csr_read32(base, PIXELFORGE_CSR_DS_DEPTH);
    pf_unpack_depth_test(w, c);
}

void pf_csr_set_blend(volatile uint8_t *base, const pixelforge_blend_config_t *c) {
    uint32_t w = pf_pack_blend_config(c);
    pf_csr_write32(base, PIXELFORGE_CSR_BLEND_CONFIG, w);
}
void pf_csr_get_blend(volatile uint8_t *base, pixelforge_blend_config_t *c) {
    uint32_t w = pf_csr_read32(base, PIXELFORGE_CSR_BLEND_CONFIG);
    pf_unpack_blend_config(w, c);
}
