/*
 * PixelForge Demo: Stencil Buffer - Object Outline/Glow Effect
 *
 * This demo showcases:
 * - Stencil buffer operations (write, test, masking)
 * - Two-pass rendering technique
 * - Creating outline/glow effect around an object
 *
 * Technique:
 * Pass 1: Draw object with stencil write (reference=1, op=REPLACE)
 * Pass 2: Draw slightly enlarged version with stencil test (compare==1, invert)
 *         This creates border only where original object wasn't
 */

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <stddef.h>

#include "graphics_pipeline_csr_access.h"
#include "vram_alloc.h"
#include "pixelforge_utils.h"
#include "demo_utils.h"

#define PAGE_SIZE       4096u
#define VB_REGION_SIZE  0x00010000u

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static volatile bool keep_running = true;
static bool g_verbose = false;

#define DBG(fmt, ...) do { if (g_verbose) fprintf(stderr, "[dbg] " fmt "\n", ##__VA_ARGS__); } while (0)

static void handle_sigint(int sig) {
    (void)sig;
    keep_running = false;
}

static int32_t fp16_16(float v) {
    return (int32_t)(v * 65536.0f);
}

struct vertex {
    int32_t pos[4];
    int32_t norm[3];
    int32_t col[4];
};

static int wait_for_gpu_ready(pixelforge_dev *dev) {
    for (int i = 0; i < 10000000 && keep_running; ++i) {
        uint32_t ready = pf_csr_get_ready(dev->csr_base);
        if (ready & 0x1) return 0;
        usleep(50);
    }
    return -1;
}

/* Create cube with colorful faces (same as demo_cube) */
static void create_cube(struct vertex *vertices, uint16_t *indices, uint32_t *idx_count) {
    float vtx_pos[24][3] = {
        {-0.5f, -0.5f,  0.5f}, {0.5f, -0.5f,  0.5f}, {0.5f,  0.5f,  0.5f}, {-0.5f,  0.5f,  0.5f},
        { 0.5f, -0.5f, -0.5f}, {-0.5f, -0.5f, -0.5f}, {-0.5f,  0.5f, -0.5f}, { 0.5f,  0.5f, -0.5f},
        {-0.5f, -0.5f, -0.5f}, {-0.5f, -0.5f,  0.5f}, {-0.5f,  0.5f,  0.5f}, {-0.5f,  0.5f, -0.5f},
        { 0.5f, -0.5f,  0.5f}, { 0.5f, -0.5f, -0.5f}, { 0.5f,  0.5f, -0.5f}, { 0.5f,  0.5f,  0.5f},
        {-0.5f,  0.5f,  0.5f}, { 0.5f,  0.5f,  0.5f}, { 0.5f,  0.5f, -0.5f}, {-0.5f,  0.5f, -0.5f},
        {-0.5f, -0.5f, -0.5f}, { 0.5f, -0.5f, -0.5f}, { 0.5f, -0.5f,  0.5f}, {-0.5f, -0.5f,  0.5f},
    };

    float vtx_norm[24][3] = {
        {0,0,1}, {0,0,1}, {0,0,1}, {0,0,1},
        {0,0,-1}, {0,0,-1}, {0,0,-1}, {0,0,-1},
        {-1,0,0}, {-1,0,0}, {-1,0,0}, {-1,0,0},
        {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0},
        {0,1,0}, {0,1,0}, {0,1,0}, {0,1,0},
        {0,-1,0}, {0,-1,0}, {0,-1,0}, {0,-1,0},
    };

    float vtx_color[24][3] = {
        {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0},
        {0,1,0}, {0,1,0}, {0,1,0}, {0,1,0},
        {0,0,1}, {0,0,1}, {0,0,1}, {0,0,1},
        {1,1,0}, {1,1,0}, {1,1,0}, {1,1,0},
        {0,1,1}, {0,1,1}, {0,1,1}, {0,1,1},
        {1,0,1}, {1,0,1}, {1,0,1}, {1,0,1},
    };

    int idx[36] = {
        0,1,2, 0,2,3,
        4,5,6, 4,6,7,
        8,9,10, 8,10,11,
        12,13,14, 12,14,15,
        16,17,18, 16,18,19,
        20,21,22, 20,22,23
    };

    for (int i = 0; i < 24; i++) {
        vertices[i].pos[0] = fp16_16(vtx_pos[i][0]);
        vertices[i].pos[1] = fp16_16(vtx_pos[i][1]);
        vertices[i].pos[2] = fp16_16(vtx_pos[i][2]);
        vertices[i].pos[3] = fp16_16(1.0f);
        vertices[i].norm[0] = fp16_16(vtx_norm[i][0]);
        vertices[i].norm[1] = fp16_16(vtx_norm[i][1]);
        vertices[i].norm[2] = fp16_16(vtx_norm[i][2]);
        vertices[i].col[0] = fp16_16(vtx_color[i][0]);
        vertices[i].col[1] = fp16_16(vtx_color[i][1]);
        vertices[i].col[2] = fp16_16(vtx_color[i][2]);
        vertices[i].col[3] = fp16_16(1.0f);
    }

    for (int i = 0; i < 36; i++) {
        indices[i] = (uint16_t)idx[i];
    }

    *idx_count = 36;
}

static void configure_gpu_base(pixelforge_dev *dev, uint32_t idx_addr, uint32_t idx_count,
                               uint32_t pos_addr, uint32_t norm_addr, uint32_t col_addr,
                               uint16_t stride, uint32_t color_addr, uint32_t ds_addr,
                               const float mv[16], const float p[16]) {
    volatile uint8_t *csr = dev->csr_base;

    pixelforge_idx_config_t idx_cfg = {
        .address = idx_addr,
        .count = idx_count,
        .kind = PIXELFORGE_INDEX_U16,
    };
    pf_csr_set_idx(csr, &idx_cfg);

    pixelforge_topo_config_t topo = {
        .input_topology = PIXELFORGE_TOPOLOGY_TRIANGLE_LIST,
        .primitive_restart_enable = false,
        .primitive_restart_index = 0,
        .base_vertex = 0,
    };
    pf_csr_set_topology(csr, &topo);

    pixelforge_input_attr_t attr = {
        .mode = PIXELFORGE_ATTR_PER_VERTEX,
        .info.per_vertex = { .address = pos_addr, .stride = stride },
    };
    pf_csr_set_attr_position(csr, &attr);
    attr.info.per_vertex.address = norm_addr;
    pf_csr_set_attr_normal(csr, &attr);
    attr.info.per_vertex.address = col_addr;
    pf_csr_set_attr_color(csr, &attr);

    pixelforge_vtx_xf_config_t xf = {0};
    xf.enabled.normal_enable = true;
    for (int i = 0; i < 16; i++) {
        xf.position_mv[i] = fp16_16(mv[i]);
        xf.position_p[i] = fp16_16(p[i]);
    }

    float nm[9];
    mat3_from_mat4(nm, mv);
    for (int i = 0; i < 9; i++) {
        xf.normal_mv_inv_t[i] = fp16_16(nm[i]);
    }
    pf_csr_set_vtx_xf(csr, &xf);

    pixelforge_material_t mat = {0};
    mat.ambient[0] = fp16_16(1.0f);
    mat.ambient[1] = fp16_16(1.0f);
    mat.ambient[2] = fp16_16(1.0f);
    mat.diffuse[0] = fp16_16(0.0f);
    mat.diffuse[1] = fp16_16(0.0f);
    mat.diffuse[2] = fp16_16(0.0f);
    mat.specular[0] = fp16_16(0.0f);
    mat.specular[1] = fp16_16(0.0f);
    mat.specular[2] = fp16_16(0.0f);
    mat.shininess = fp16_16(1.0f);
    pf_csr_set_material(csr, &mat);

    pixelforge_light_t light = {0};
    light.position[0] = fp16_16(0.0f);
    light.position[1] = fp16_16(0.0f);
    light.position[2] = fp16_16(1.0f);
    light.position[3] = fp16_16(0.0f);
    light.ambient[0] = fp16_16(1.0f);
    light.ambient[1] = fp16_16(1.0f);
    light.ambient[2] = fp16_16(1.0f);
    light.diffuse[0] = fp16_16(0.0f);
    light.diffuse[1] = fp16_16(0.0f);
    light.diffuse[2] = fp16_16(0.0f);
    light.specular[0] = fp16_16(0.0f);
    light.specular[1] = fp16_16(0.0f);
    light.specular[2] = fp16_16(0.0f);
    pf_csr_set_light0(csr, &light);

    pixelforge_prim_config_t prim = {
        .type = PIXELFORGE_PRIM_TRIANGLES,
        .cull = PIXELFORGE_CULL_BACK,
        .winding = PIXELFORGE_WINDING_CCW,
    };
    pf_csr_set_prim(csr, &prim);

    pixelforge_framebuffer_config_t fb = {0};
    fb.width = dev->x_resolution;
    fb.height = dev->y_resolution;
    fb.viewport_x = fp16_16(0.0f);
    fb.viewport_y = fp16_16(0.0f);
    fb.viewport_width = fp16_16((float)dev->x_resolution);
    fb.viewport_height = fp16_16((float)dev->y_resolution);
    fb.viewport_min_depth = fp16_16(0.0f);
    fb.viewport_max_depth = fp16_16(1.0f);
    fb.scissor_offset_x = 0;
    fb.scissor_offset_y = 0;
    fb.scissor_width = dev->x_resolution;
    fb.scissor_height = dev->y_resolution;
    fb.color_address = color_addr;
    fb.color_pitch = dev->buffer_stride;
    fb.depthstencil_address = ds_addr;
    fb.depthstencil_pitch = dev->x_resolution * 4;
    pf_csr_set_fb(csr, &fb);

    pixelforge_depth_test_config_t depth = {
        .test_enabled = true,
        .write_enabled = true,
        .compare_op = PIXELFORGE_CMP_LESS,
    };
    pf_csr_set_depth(csr, &depth);

    pixelforge_blend_config_t blend = {
        .src_factor = PIXELFORGE_BLEND_ONE,
        .dst_factor = PIXELFORGE_BLEND_ZERO,
        .src_a_factor = PIXELFORGE_BLEND_ONE,
        .dst_a_factor = PIXELFORGE_BLEND_ZERO,
        .enabled = false,
        .blend_op = PIXELFORGE_BLEND_ADD,
        .blend_a_op = PIXELFORGE_BLEND_ADD,
        .color_write_mask = 0xF,
    };
    pf_csr_set_blend(csr, &blend);
}

static void set_stencil_write_mode(pixelforge_dev *dev) {
    volatile uint8_t *csr = dev->csr_base;

    /* Pass 1: Write stencil where object is drawn */
    pixelforge_stencil_op_config_t stencil = {0};
    stencil.compare_op = PIXELFORGE_CMP_ALWAYS;
    stencil.reference = 1;
    stencil.mask = 0xFF;
    stencil.write_mask = 0xFF;
    stencil.fail_op = PIXELFORGE_STENCIL_KEEP;
    stencil.depth_fail_op = PIXELFORGE_STENCIL_KEEP;
    stencil.pass_op = PIXELFORGE_STENCIL_REPLACE; /* Write reference value */
    pf_csr_set_stencil_front(csr, &stencil);
    pf_csr_set_stencil_back(csr, &stencil);
}

static void set_stencil_outline_mode(pixelforge_dev *dev) {
    volatile uint8_t *csr = dev->csr_base;

    /* Pass 2: Draw only where stencil is NOT 1 (creates border) */
    pixelforge_stencil_op_config_t stencil = {0};
    stencil.compare_op = PIXELFORGE_CMP_NOT_EQUAL;
    stencil.reference = 1;
    stencil.mask = 0xFF;
    stencil.write_mask = 0x00; /* Don't write */
    stencil.fail_op = PIXELFORGE_STENCIL_KEEP;
    stencil.depth_fail_op = PIXELFORGE_STENCIL_KEEP;
    stencil.pass_op = PIXELFORGE_STENCIL_KEEP;
    pf_csr_set_stencil_front(csr, &stencil);
    pf_csr_set_stencil_back(csr, &stencil);
}

int main(int argc, char **argv) {
    int frames = 90;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--verbose")) g_verbose = true;
        else if (!strcmp(argv[i], "--frames") && i + 1 < argc) frames = atoi(argv[++i]);
    }

    signal(SIGINT, handle_sigint);

    pixelforge_dev *dev = pixelforge_open_dev();
    if (!dev) {
        fprintf(stderr, "Failed to open device\n");
        return 1;
    }

    printf("PixelForge Stencil Demo: Object Outline/Glow Effect\n");
    printf("Rendering %d frames...\n", frames);

    /* Allocate buffers from VRAM */
    struct vram_block vb_block, ds_block;

    if (vram_alloc(&dev->vram, VB_REGION_SIZE, PAGE_SIZE, &vb_block) ||
        vram_alloc(&dev->vram, dev->x_resolution * dev->y_resolution * 4, PAGE_SIZE, &ds_block)) {
        fprintf(stderr, "VRAM allocation failed\n");
        pixelforge_close_dev(dev);
        return 1;
    }

    /* Create geometry (cube) */
    struct vertex *vertices = (struct vertex *)vb_block.virt;
    uint16_t *indices;
    uint32_t idx_count;
    uint32_t idx_addr;
    uint32_t pos_addr;
    uint32_t norm_addr;
    uint32_t col_addr;
    uint32_t vb_phys = vb_block.phys;
    uint32_t ds_phys = ds_block.phys;
    uint8_t *ds_virt = ds_block.virt;

    indices = (uint16_t*)(vertices + 24);  /* Index data follows vertex data */
    create_cube(vertices, indices, &idx_count);

    idx_addr = vb_phys + sizeof(struct vertex) * 24;
    pos_addr = vb_phys + offsetof(struct vertex, pos);
    norm_addr = vb_phys + offsetof(struct vertex, norm);
    col_addr = vb_phys + offsetof(struct vertex, col);
    uint16_t stride = sizeof(struct vertex);

    /* Projection matrix (match demo_cube) */
    float p[16];
    mat4_perspective(p, 45.0f * M_PI / 180.0f, (float)dev->x_resolution / (float)dev->y_resolution, 0.5f, 5.0f);

    /* Animation loop */
    for (int frame = 0; frame < frames && keep_running; frame++) {
        float t = (float)frame / 30.0f;

        /* Get back buffer for rendering */
        uint8_t *buffer = pixelforge_get_back_buffer(dev);
        uint32_t buffer_phys = dev->buffer_phys[dev->render_buffer];

        /* Clear buffers */
        memset(buffer, 0x0A, dev->buffer_size);  /* Dark background */
        memset(ds_virt, 0x00, dev->x_resolution * dev->y_resolution * 4); /* Clear depth AND stencil */

        /* Build transforms (match demo_cube) */
        float rot[16], trans[16], mv[16];
        mat4_rotate_xyz(rot, t * 0.7f, t, t * 0.5f);
        mat4_translate(trans, 0.0f, 0.0f, -4.0f);
        mat4_multiply(mv, rot, trans);

        /* === PASS 1: Draw object and mark stencil === */
        configure_gpu_base(dev, idx_addr, idx_count, pos_addr, norm_addr, col_addr,
                  stride, buffer_phys, ds_phys, mv, p);
        set_stencil_write_mode(dev);

        pf_csr_start(dev->csr_base);
        if (wait_for_gpu_ready(dev) != 0) break;

        /* === PASS 2: Draw enlarged object where stencil != 1 (outline) === */
        float scale[16], mv_outline[16];
        mat4_scale(scale, 1.15f, 1.15f, 1.15f); /* Slightly larger */
        mat4_multiply(mv_outline, mv, scale);

        /* Override color to glow color */
        for (int i = 0; i < 6; i++) {
            vertices[i].col[0] = fp16_16(1.0f); /* Bright yellow/orange glow */
            vertices[i].col[1] = fp16_16(0.8f);
            vertices[i].col[2] = fp16_16(0.0f);
            vertices[i].col[3] = fp16_16(1.0f);
        }

        configure_gpu_base(dev, idx_addr, idx_count, pos_addr, norm_addr, col_addr,
                  stride, buffer_phys, ds_phys, mv_outline, p);
        set_stencil_outline_mode(dev);

        /* Disable depth write for outline so it doesn't interfere */
        pixelforge_depth_test_config_t depth = {
            .test_enabled = false,
            .write_enabled = false,
            .compare_op = PIXELFORGE_CMP_ALWAYS,
        };
        pf_csr_set_depth(dev->csr_base, &depth);

        pf_csr_start(dev->csr_base);
        if (wait_for_gpu_ready(dev) != 0) break;

        /* Restore original color for next frame */
        for (int i = 0; i < 6; i++) {
            vertices[i].col[0] = fp16_16(0.8f);
            vertices[i].col[1] = fp16_16(0.5f);
            vertices[i].col[2] = fp16_16(0.2f);
            vertices[i].col[3] = fp16_16(1.0f);
        }

        pixelforge_swap_buffers(dev);
        printf("Frame %d/%d rendered (with outline effect)\n", frame + 1, frames);
    }

    pixelforge_close_dev(dev);
    printf("Done!\n");
    return 0;
}
