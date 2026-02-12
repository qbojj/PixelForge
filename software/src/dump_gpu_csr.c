#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "graphics_pipeline_csr_access.h"

#define PAGE_SIZE       4096u
#define PAGE_ALIGN_DOWN(a) ((a) & ~(PAGE_SIZE - 1u))
#define PAGE_OFFSET(a)  ((a) & (PAGE_SIZE - 1u))
#define MAP_ALIGN(v)    (((v) + PAGE_SIZE - 1u) & ~(PAGE_SIZE - 1u))

/* Physical base addresses */
#define PF_CSR_BASE_PHYS     0xFF200000u
#define PF_CSR_MAP_SIZE      0x4000u

static void* map_physical(int memfd, uint32_t phys, size_t length) {
    uint32_t aligned_phys = PAGE_ALIGN_DOWN(phys);
    uint32_t offset = PAGE_OFFSET(phys);
    size_t aligned_length = MAP_ALIGN(length + offset);

    uint8_t *mapped = mmap(NULL, aligned_length, PROT_READ | PROT_WRITE,
                          MAP_SHARED, memfd, aligned_phys);
    if (mapped == MAP_FAILED) {
        perror("mmap");
        return NULL;
    }
    return mapped + offset;
}

/* Fixed-point to float conversion (16.16 format) */
static float fp16_16_to_float(int32_t fp) {
    return (float)fp / 65536.0f;
}

static void dump_idx_config(volatile uint8_t *csr) {
    pixelforge_idx_config_t cfg;
    pf_csr_get_idx(csr, &cfg);

    printf("\n[INDEX]\n");
    printf("  address:  0x%08x\n", cfg.address);
    printf("  count:    %u\n", cfg.count);
    printf("  kind:     %u ", cfg.kind);
    switch (cfg.kind) {
        case PIXELFORGE_INDEX_NOT_INDEXED: printf("(NOT_INDEXED)\n"); break;
        case PIXELFORGE_INDEX_U8:  printf("(U8)\n"); break;
        case PIXELFORGE_INDEX_U16: printf("(U16)\n"); break;
        case PIXELFORGE_INDEX_U32: printf("(U32)\n"); break;
        default: printf("(unknown)\n");
    }
}

static void dump_topology_config(volatile uint8_t *csr) {
    pixelforge_topo_config_t cfg;
    pf_csr_get_topology(csr, &cfg);

    printf("\n[TOPOLOGY]\n");
    printf("  input_topology:           %u ", cfg.input_topology);
    switch (cfg.input_topology) {
        case PIXELFORGE_TOPOLOGY_POINT_LIST: printf("(POINT_LIST)\n"); break;
        case PIXELFORGE_TOPOLOGY_LINE_LIST: printf("(LINE_LIST)\n"); break;
        case PIXELFORGE_TOPOLOGY_LINE_STRIP: printf("(LINE_STRIP)\n"); break;
        case PIXELFORGE_TOPOLOGY_TRIANGLE_LIST: printf("(TRIANGLE_LIST)\n"); break;
        case PIXELFORGE_TOPOLOGY_TRIANGLE_STRIP: printf("(TRIANGLE_STRIP)\n"); break;
        case PIXELFORGE_TOPOLOGY_TRIANGLE_FAN: printf("(TRIANGLE_FAN)\n"); break;
        case PIXELFORGE_TOPOLOGY_LINE_LIST_ADJACENCY: printf("(LINE_LIST_ADJACENCY)\n"); break;
        case PIXELFORGE_TOPOLOGY_LINE_STRIP_ADJACENCY: printf("(LINE_STRIP_ADJACENCY)\n"); break;
        case PIXELFORGE_TOPOLOGY_TRIANGLE_LIST_ADJACENCY: printf("(TRIANGLE_LIST_ADJACENCY)\n"); break;
        case PIXELFORGE_TOPOLOGY_TRIANGLE_STRIP_ADJACENCY: printf("(TRIANGLE_STRIP_ADJACENCY)\n"); break;
        case PIXELFORGE_TOPOLOGY_PATCH_LIST: printf("(PATCH_LIST)\n"); break;
        default: printf("(unknown)\n");
    }
    printf("  primitive_restart_enable: %u\n", cfg.primitive_restart_enable);
    printf("  primitive_restart_index:  0x%08x\n", cfg.primitive_restart_index);
    printf("  base_vertex:              %u\n", cfg.base_vertex);
}

static const char *attr_mode_str(uint32_t mode) {
    switch (mode) {
        case PIXELFORGE_ATTR_CONSTANT: return "CONSTANT";
        case PIXELFORGE_ATTR_PER_VERTEX: return "PER_VERTEX";
        default: return "unknown";
    }
}

static void dump_input_attr(volatile uint8_t *csr, const char *name,
                            void (*getter)(volatile uint8_t *, pixelforge_input_attr_t *)) {
    pixelforge_input_attr_t attr;
    getter(csr, &attr);

    printf("  [%s]\n", name);
    printf("    mode:   %u (%s)\n", attr.mode, attr_mode_str(attr.mode));

    if (attr.mode == PIXELFORGE_ATTR_CONSTANT) {
        printf("    constant_value: [%.4f, %.4f, %.4f, %.4f]\n",
            fp16_16_to_float(attr.info.constant_value.value[0]),
            fp16_16_to_float(attr.info.constant_value.value[1]),
            fp16_16_to_float(attr.info.constant_value.value[2]),
            fp16_16_to_float(attr.info.constant_value.value[3]));
    } else if (attr.mode == PIXELFORGE_ATTR_PER_VERTEX) {
        printf("    per_vertex:\n");
        printf("      address: 0x%08x\n", attr.info.per_vertex.address);
        printf("      stride:  %u\n", attr.info.per_vertex.stride);
    }
}

static void dump_input_assembly(volatile uint8_t *csr) {
    printf("\n[INPUT ASSEMBLY]\n");
    dump_input_attr(csr, "POSITION", pf_csr_get_attr_position);
    dump_input_attr(csr, "NORMAL", pf_csr_get_attr_normal);
    dump_input_attr(csr, "COLOR", pf_csr_get_attr_color);
}

static void dump_vertex_transform(volatile uint8_t *csr) {
    pixelforge_vtx_xf_config_t cfg;
    pf_csr_get_vtx_xf(csr, &cfg);

    printf("\n[VERTEX TRANSFORM]\n");
    printf("  enabled:\n");
    printf("    normal:       %u\n", cfg.enabled.normal_enable);

    printf("  position_mv (4x4):\n");
    for (int i = 0; i < 4; ++i) {
        printf("    [%d] ", i);
        for (int j = 0; j < 4; ++j) {
            printf("%10.4f ", fp16_16_to_float(cfg.position_mv[i * 4 + j]));
        }
        printf("\n");
    }

    printf("  position_p (4x4):\n");
    for (int i = 0; i < 4; ++i) {
        printf("    [%d] ", i);
        for (int j = 0; j < 4; ++j) {
            printf("%10.4f ", fp16_16_to_float(cfg.position_p[i * 4 + j]));
        }
        printf("\n");
    }

    printf("  normal_mv_inv_t (3x3):\n");
    for (int i = 0; i < 3; ++i) {
        printf("    [%d] ", i);
        for (int j = 0; j < 3; ++j) {
            printf("%10.4f ", fp16_16_to_float(cfg.normal_mv_inv_t[i * 3 + j]));
        }
        printf("\n");
    }
}

static void dump_material(volatile uint8_t *csr) {
    pixelforge_material_t mat;
    pf_csr_get_material(csr, &mat);

    printf("\n[MATERIAL]\n");
    printf("  ambient:  [%.4f, %.4f, %.4f]\n",
        fp16_16_to_float(mat.ambient[0]),
        fp16_16_to_float(mat.ambient[1]),
        fp16_16_to_float(mat.ambient[2]));
    printf("  diffuse:  [%.4f, %.4f, %.4f]\n",
        fp16_16_to_float(mat.diffuse[0]),
        fp16_16_to_float(mat.diffuse[1]),
        fp16_16_to_float(mat.diffuse[2]));
    printf("  specular: [%.4f, %.4f, %.4f]\n",
        fp16_16_to_float(mat.specular[0]),
        fp16_16_to_float(mat.specular[1]),
        fp16_16_to_float(mat.specular[2]));
    printf("  shininess: %.4f\n", fp16_16_to_float(mat.shininess));
}

static void dump_light(volatile uint8_t *csr, int index) {
    pixelforge_light_t light;
    if (index == 0) {
        pf_csr_get_light(csr, 0, &light);
    } else {
        printf("  Light %d: not supported in getter functions\n", index);
        return;
    }

    printf("  [LIGHT %d]\n", index);
    printf("    position: [%.4f, %.4f, %.4f, %.4f]\n",
        fp16_16_to_float(light.position[0]),
        fp16_16_to_float(light.position[1]),
        fp16_16_to_float(light.position[2]),
        fp16_16_to_float(light.position[3]));
    printf("    ambient:  [%.4f, %.4f, %.4f]\n",
        fp16_16_to_float(light.ambient[0]),
        fp16_16_to_float(light.ambient[1]),
        fp16_16_to_float(light.ambient[2]));
    printf("    diffuse:  [%.4f, %.4f, %.4f]\n",
        fp16_16_to_float(light.diffuse[0]),
        fp16_16_to_float(light.diffuse[1]),
        fp16_16_to_float(light.diffuse[2]));
    printf("    specular: [%.4f, %.4f, %.4f]\n",
        fp16_16_to_float(light.specular[0]),
        fp16_16_to_float(light.specular[1]),
        fp16_16_to_float(light.specular[2]));
}

static void dump_vertex_shading(volatile uint8_t *csr) {
    printf("\n[VERTEX SHADING]\n");
    dump_material(csr);
    printf("\n  [LIGHTING]\n");
    dump_light(csr, 0);
}

static const char *prim_type_str(uint32_t type) {
    switch (type) {
        case PIXELFORGE_PRIM_POINTS: return "POINTS";
        case PIXELFORGE_PRIM_LINES: return "LINES";
        case PIXELFORGE_PRIM_TRIANGLES: return "TRIANGLES";
        default: return "unknown";
    }
}

static const char *cull_mode_str(uint32_t cull) {
    switch (cull) {
        case PIXELFORGE_CULL_NONE: return "NONE";
        case PIXELFORGE_CULL_FRONT: return "FRONT";
        case PIXELFORGE_CULL_BACK: return "BACK";
        case PIXELFORGE_CULL_FRONT_AND_BACK: return "FRONT_AND_BACK";
        default: return "unknown";
    }
}

static const char *winding_str(uint32_t winding) {
    switch (winding) {
        case PIXELFORGE_WINDING_CCW: return "CCW";
        case PIXELFORGE_WINDING_CW: return "CW";
        default: return "unknown";
    }
}

static void dump_primitive_assembly(volatile uint8_t *csr) {
    pixelforge_prim_config_t cfg;
    pf_csr_get_prim(csr, &cfg);

    printf("\n[PRIMITIVE ASSEMBLY]\n");
    printf("  type:    %u (%s)\n", cfg.type, prim_type_str(cfg.type));
    printf("  cull:    %u (%s)\n", cfg.cull, cull_mode_str(cfg.cull));
    printf("  winding: %u (%s)\n", cfg.winding, winding_str(cfg.winding));
}

static void dump_framebuffer(volatile uint8_t *csr) {
    pixelforge_framebuffer_config_t cfg;
    pf_csr_get_fb(csr, &cfg);

    printf("\n[FRAMEBUFFER]\n");
    printf("  dimensions:\n");
    printf("    width:  %u\n", cfg.width);
    printf("    height: %u\n", cfg.height);

    printf("  viewport:\n");
    printf("    x:           %.4f\n", fp16_16_to_float(cfg.viewport_x));
    printf("    y:           %.4f\n", fp16_16_to_float(cfg.viewport_y));
    printf("    width:       %.4f\n", fp16_16_to_float(cfg.viewport_width));
    printf("    height:      %.4f\n", fp16_16_to_float(cfg.viewport_height));
    printf("    min_depth:   %.4f\n", fp16_16_to_float(cfg.viewport_min_depth));
    printf("    max_depth:   %.4f\n", fp16_16_to_float(cfg.viewport_max_depth));

    printf("  scissor:\n");
    printf("    offset_x: %u\n", cfg.scissor_offset_x);
    printf("    offset_y: %u\n", cfg.scissor_offset_y);
    printf("    width:    %u\n", cfg.scissor_width);
    printf("    height:   %u\n", cfg.scissor_height);

    printf("  color buffer:\n");
    printf("    address: 0x%08x\n", cfg.color_address);
    printf("    pitch:   %u bytes/line\n", cfg.color_pitch);

    printf("  depth/stencil buffer:\n");
    printf("    address: 0x%08x\n", cfg.depthstencil_address);
    printf("    pitch:   %u bytes/line\n", cfg.depthstencil_pitch);
}

static const char *cmp_op_str(uint32_t op) {
    switch (op) {
        case PIXELFORGE_CMP_NEVER: return "NEVER";
        case PIXELFORGE_CMP_LESS: return "LESS";
        case PIXELFORGE_CMP_EQUAL: return "EQUAL";
        case PIXELFORGE_CMP_LESS_OR_EQUAL: return "LEQUAL";
        case PIXELFORGE_CMP_GREATER: return "GREATER";
        case PIXELFORGE_CMP_NOT_EQUAL: return "NOTEQUAL";
        case PIXELFORGE_CMP_GREATER_OR_EQUAL: return "GEQUAL";
        case PIXELFORGE_CMP_ALWAYS: return "ALWAYS";
        default: return "unknown";
    }
}

static void dump_depth_test(volatile uint8_t *csr) {
    pixelforge_depth_test_config_t cfg;
    pf_csr_get_depth(csr, &cfg);

    printf("\n[DEPTH TEST]\n");
    printf("  test_enabled:  %u\n", cfg.test_enabled);
    printf("  write_enabled: %u\n", cfg.write_enabled);
    printf("  compare_op:    %u (%s)\n", cfg.compare_op, cmp_op_str(cfg.compare_op));
}

static const char *stencil_op_str(uint32_t op) {
    switch (op) {
        case PIXELFORGE_STENCIL_KEEP: return "KEEP";
        case PIXELFORGE_STENCIL_ZERO: return "ZERO";
        case PIXELFORGE_STENCIL_REPLACE: return "REPLACE";
        case PIXELFORGE_STENCIL_INCR: return "INCR";
        case PIXELFORGE_STENCIL_INCR_WRAP: return "INCR_WRAP";
        case PIXELFORGE_STENCIL_DECR: return "DECR";
        case PIXELFORGE_STENCIL_DECR_WRAP: return "DECR_WRAP";
        case PIXELFORGE_STENCIL_INVERT: return "INVERT";
        default: return "unknown";
    }
}

static void dump_stencil_config(volatile uint8_t *csr, int back) {
    pixelforge_stencil_op_config_t cfg;
    if (back) {
        pf_csr_get_stencil_back(csr, &cfg);
    } else {
        pf_csr_get_stencil_front(csr, &cfg);
    }

    printf("  [%s]\n", back ? "BACK" : "FRONT");
    printf("    compare_op:       %u (%s)\n", cfg.compare_op, cmp_op_str(cfg.compare_op));
    printf("    reference:        0x%02x\n", cfg.reference);
    printf("    mask:             0x%02x\n", cfg.mask);
    printf("    write_mask:       0x%02x\n", cfg.write_mask);
    printf("    fail_op:          %u (%s)\n", cfg.fail_op, stencil_op_str(cfg.fail_op));
    printf("    depth_fail_op:    %u (%s)\n", cfg.depth_fail_op, stencil_op_str(cfg.depth_fail_op));
    printf("    pass_op:          %u (%s)\n", cfg.pass_op, stencil_op_str(cfg.pass_op));
}

static void dump_output_merger(volatile uint8_t *csr) {
    printf("\n[OUTPUT MERGER]\n");
    printf("  Depth/Stencil:\n");
    dump_depth_test(csr);
    printf("\n  Stencil Operations:\n");
    dump_stencil_config(csr, 0);
    dump_stencil_config(csr, 1);
}

static const char *blend_factor_str(uint32_t factor) {
    switch (factor) {
        case PIXELFORGE_BLEND_ZERO: return "ZERO";
        case PIXELFORGE_BLEND_ONE: return "ONE";
        case PIXELFORGE_BLEND_SRC_COLOR: return "SRC_COLOR";
        case PIXELFORGE_BLEND_ONE_MINUS_SRC_COLOR: return "ONE_MINUS_SRC_COLOR";
        case PIXELFORGE_BLEND_DST_COLOR: return "DST_COLOR";
        case PIXELFORGE_BLEND_ONE_MINUS_DST_COLOR: return "ONE_MINUS_DST_COLOR";
        case PIXELFORGE_BLEND_SRC_ALPHA: return "SRC_ALPHA";
        case PIXELFORGE_BLEND_ONE_MINUS_SRC_ALPHA: return "ONE_MINUS_SRC_ALPHA";
        case PIXELFORGE_BLEND_DST_ALPHA: return "DST_ALPHA";
        case PIXELFORGE_BLEND_ONE_MINUS_DST_ALPHA: return "ONE_MINUS_DST_ALPHA";
        default: return "unknown";
    }
}

static const char *blend_op_str(uint32_t op) {
    switch (op) {
        case PIXELFORGE_BLEND_ADD: return "ADD";
        case PIXELFORGE_BLEND_SUBTRACT: return "SUBTRACT";
        case PIXELFORGE_BLEND_REVERSE_SUBTRACT: return "REV_SUBTRACT";
        case PIXELFORGE_BLEND_MIN: return "MIN";
        case PIXELFORGE_BLEND_MAX: return "MAX";
        default: return "unknown";
    }
}

static void dump_blending(volatile uint8_t *csr) {
    pixelforge_blend_config_t cfg;
    pf_csr_get_blend(csr, &cfg);

    printf("\n[BLENDING]\n");
    printf("  enabled:         %u\n", cfg.enabled);
    printf("  src_factor:      %u (%s)\n", cfg.src_factor, blend_factor_str(cfg.src_factor));
    printf("  dst_factor:      %u (%s)\n", cfg.dst_factor, blend_factor_str(cfg.dst_factor));
    printf("  src_a_factor:    %u (%s)\n", cfg.src_a_factor, blend_factor_str(cfg.src_a_factor));
    printf("  dst_a_factor:    %u (%s)\n", cfg.dst_a_factor, blend_factor_str(cfg.dst_a_factor));
    printf("  blend_op:        %u (%s)\n", cfg.blend_op, blend_op_str(cfg.blend_op));
    printf("  blend_a_op:      %u (%s)\n", cfg.blend_a_op, blend_op_str(cfg.blend_a_op));
    printf("  color_write_mask: 0x%x\n", cfg.color_write_mask);
}

static void dump_pixel_shading(volatile uint8_t *csr) {
    printf("\n[PIXEL SHADING]\n");
    dump_blending(csr);
    dump_output_merger(csr);
}

static void dump_status(volatile uint8_t *csr) {
    uint32_t ready = pf_csr_get_ready(csr);
    uint32_t ready_components = pf_csr_get_ready_components(csr);
    uint32_t ready_vec = pf_csr_get_ready_vec(csr);

    printf("\n[STATUS]\n");
    printf("  ready: %u (%s)\n", ready & 1, ready & 1 ? "ready" : "busy");
    printf("  ia:      (%s)\n", ready_components & 1 ? "ready" : "busy");
    printf("  vt:      (%s)\n", ready_components & 2 ? "ready" : "busy");
    printf("  rast:    (%s)\n", ready_components & 4 ? "ready" : "busy");
    printf("  pix:     (%s)\n", ready_components & 8 ? "ready" : "busy");

    printf("  ready vector:  %0b\n", ready_vec);
}

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    int memfd = open("/dev/mem", O_RDWR | O_SYNC);
    if (memfd < 0) {
        perror("open /dev/mem");
        return 1;
    }

    volatile uint8_t *csr = map_physical(memfd, PF_CSR_BASE_PHYS, PF_CSR_MAP_SIZE);
    if (!csr) {
        close(memfd);
        return 1;
    }

    printf("================================================================================\n");
    printf("GPU PIPELINE CSR DUMP\n");
    printf("================================================================================\n");

    dump_idx_config(csr);
    dump_topology_config(csr);
    dump_input_assembly(csr);
    dump_vertex_transform(csr);
    dump_vertex_shading(csr);
    dump_primitive_assembly(csr);
    dump_framebuffer(csr);
    dump_pixel_shading(csr);
    dump_status(csr);

    printf("\n================================================================================\n");

    munmap((void*)csr, MAP_ALIGN(PF_CSR_MAP_SIZE + PAGE_OFFSET(PF_CSR_BASE_PHYS)));
    close(memfd);

    return 0;
}
