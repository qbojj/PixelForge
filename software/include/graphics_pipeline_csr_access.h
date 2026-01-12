#ifndef PIXELFORGE_CSR_ACCESS_H
#define PIXELFORGE_CSR_ACCESS_H

#include <stdint.h>
#include <stddef.h>

#include "graphics_pipeline_csr.h"
#include "graphics_pipeline_formats.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Low-level MMIO helpers */
static inline void pf_csr_write32(volatile uint8_t *base, uint32_t offset, uint32_t value) {
    *(volatile uint32_t *)(base + offset) = value;
}

static inline uint32_t pf_csr_read32(volatile uint8_t *base, uint32_t offset) {
    return *(volatile uint32_t *)(base + offset);
}

/* =============================
 * Index Generator
 * ============================= */
void pf_csr_set_idx(volatile uint8_t *base, const pixelforge_idx_config_t *cfg);
void pf_csr_get_idx(volatile uint8_t *base, pixelforge_idx_config_t *cfg);
void pf_csr_start(volatile uint8_t *base);

/* =============================
 * Topology
 * ============================= */
void pf_csr_set_topology(volatile uint8_t *base, const pixelforge_topo_config_t *cfg);
void pf_csr_get_topology(volatile uint8_t *base, pixelforge_topo_config_t *cfg);

/* =============================
 * Input Attributes (position/normal/color)
 * ============================= */
void pf_csr_set_attr_position(volatile uint8_t *base, const pixelforge_input_attr_t *attr);
void pf_csr_get_attr_position(volatile uint8_t *base, pixelforge_input_attr_t *attr);

void pf_csr_set_attr_normal(volatile uint8_t *base, const pixelforge_input_attr_t *attr);
void pf_csr_get_attr_normal(volatile uint8_t *base, pixelforge_input_attr_t *attr);

void pf_csr_set_attr_color(volatile uint8_t *base, const pixelforge_input_attr_t *attr);
void pf_csr_get_attr_color(volatile uint8_t *base, pixelforge_input_attr_t *attr);

/* =============================
 * Vertex Transform
 * ============================= */
void pf_csr_set_vtx_xf(volatile uint8_t *base, const pixelforge_vtx_xf_config_t *cfg);
void pf_csr_get_vtx_xf(volatile uint8_t *base, pixelforge_vtx_xf_config_t *cfg);

/* =============================
 * Material & Light 0
 * ============================= */
void pf_csr_set_material(volatile uint8_t *base, const pixelforge_material_t *mat);
void pf_csr_get_material(volatile uint8_t *base, pixelforge_material_t *mat);

void pf_csr_set_light0(volatile uint8_t *base, const pixelforge_light_t *lit);
void pf_csr_get_light0(volatile uint8_t *base, pixelforge_light_t *lit);

/* =============================
 * Primitive Assembly
 * ============================= */
void pf_csr_set_prim(volatile uint8_t *base, const pixelforge_prim_config_t *cfg);
void pf_csr_get_prim(volatile uint8_t *base, pixelforge_prim_config_t *cfg);

/* =============================
 * Framebuffer Info
 * ============================= */
void pf_csr_set_fb(volatile uint8_t *base, const pixelforge_framebuffer_config_t *cfg);
void pf_csr_get_fb(volatile uint8_t *base, pixelforge_framebuffer_config_t *cfg);

/* =============================
 * Depth/Stencil & Blend
 * ============================= */
void pf_csr_set_stencil_front(volatile uint8_t *base, const pixelforge_stencil_op_config_t *c);
void pf_csr_get_stencil_front(volatile uint8_t *base, pixelforge_stencil_op_config_t *c);

void pf_csr_set_stencil_back(volatile uint8_t *base, const pixelforge_stencil_op_config_t *c);
void pf_csr_get_stencil_back(volatile uint8_t *base, pixelforge_stencil_op_config_t *c);

void pf_csr_set_depth(volatile uint8_t *base, const pixelforge_depth_test_config_t *c);
void pf_csr_get_depth(volatile uint8_t *base, pixelforge_depth_test_config_t *c);

void pf_csr_set_blend(volatile uint8_t *base, const pixelforge_blend_config_t *c);
void pf_csr_get_blend(volatile uint8_t *base, pixelforge_blend_config_t *c);

/* =============================
 * Status
 * ============================= */
uint32_t pf_csr_get_ready(volatile uint8_t *base);
uint32_t pf_csr_get_ready_components(volatile uint8_t *base);
uint32_t pf_csr_get_ready_vec(volatile uint8_t *base);

#ifdef __cplusplus
}
#endif

#endif /* PIXELFORGE_CSR_ACCESS_H */
