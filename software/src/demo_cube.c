/*
 * PixelForge Demo: Plain Rotating Cube with Face Culling
 *
 * This demo showcases:
 * - Basic vertex transformation and rotation
 * - Back-face culling (without depth testing)
 * - Simple cube geometry
 * - Continuous rotation animation
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
#include "demo_utils.h"
#include "pixelforge_utils.h"

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

typedef struct demo_vertex vertex;

static void configure_gpu(pixelforge_dev *dev, uint32_t idx_addr, uint32_t idx_count,
                         uint32_t pos_addr, uint32_t norm_addr, uint32_t col_addr,
                         uint16_t stride, uint32_t color_addr,
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

    /* Set transforms */
    float nm[9];
    mat3_from_mat4(nm, mv);

    pixelforge_vtx_xf_config_t xf = {0};
    xf.enabled.normal_enable = true;
    mat4_to_fp16_16(xf.position_mv, mv);
    mat4_to_fp16_16(xf.position_p, p);
    mat3_to_fp16_16(xf.normal_mv_inv_t, nm);
    pf_csr_set_vtx_xf(csr, &xf);

    /* Material: ambient only (no lighting) */
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

    /* No face culling; using depth testing instead */
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
    fb.depthstencil_address = 0;
    fb.depthstencil_pitch = 0;
    pf_csr_set_fb(csr, &fb);

    /* Depth test enabled */
    pixelforge_depth_test_config_t depth = {
        .test_enabled = false,
        .write_enabled = false,
    };
    pf_csr_set_depth(csr, &depth);

    pixelforge_stencil_op_config_t stencil = {0};
    stencil.compare_op = PIXELFORGE_CMP_ALWAYS;
    stencil.reference = 0;
    stencil.mask = 0x00;
    stencil.write_mask = 0x00;
    stencil.fail_op = PIXELFORGE_STENCIL_KEEP;
    stencil.depth_fail_op = PIXELFORGE_STENCIL_KEEP;
    stencil.pass_op = PIXELFORGE_STENCIL_KEEP;
    pf_csr_set_stencil_front(csr, &stencil);
    pf_csr_set_stencil_back(csr, &stencil);

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

volatile uint32_t data[1024 * 1024];

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

    printf("PixelForge Simple Cube Demo: Rotating Cube with Face Culling\n");
    printf("Rendering %d frames...\n", frames);

    /* Allocate vertex buffer */
    struct vram_block vb_block;
    if (vram_alloc(&dev->vram, VB_REGION_SIZE, PAGE_SIZE, &vb_block)) {
        fprintf(stderr, "VRAM allocation failed\n");
        pixelforge_close_dev(dev);
        return 1;
    }

    /* Create geometry */
    vertex *vertices = (vertex*)vb_block.virt;
    uint16_t *indices = (uint16_t*)(vb_block.virt + sizeof(vertex) * 24);
    uint32_t idx_count;
    demo_create_cube(vertices, indices, &idx_count);

    uint32_t idx_addr = vb_block.phys + sizeof(vertex) * 24;
    uint32_t pos_addr = vb_block.phys + offsetof(vertex, pos);
    uint32_t norm_addr = vb_block.phys + offsetof(vertex, norm);
    uint32_t col_addr = vb_block.phys + offsetof(vertex, col);
    uint16_t stride = sizeof(vertex);

    /* Projection matrix */
    float p[16];
    mat4_perspective(p, 45.0f * M_PI / 180.0f, (float)dev->x_resolution / (float)dev->y_resolution, 0.5f, 5.0f);

    /* Animation loop */
    for (int frame = 0; frame < frames && keep_running; frame++) {
        float t = (float)frame / 30.0f;

        /* Get back buffer for rendering */
        uint8_t *buffer = pixelforge_get_back_buffer(dev);
        uint32_t buffer_phys = dev->buffer_phys[dev->render_buffer];

        memset(buffer, 0x10, dev->buffer_size);  /* Dark gray background */

        /* Build model-view matrix (rotate on all axes + translate back) */
        float rot[16], trans[16], mv[16];
        mat4_rotate_xyz(rot, t * 0.7f, t, t * 0.5f);
        mat4_translate(trans, 0.0f, 0.0f, -4.0f);
        mat4_multiply(mv, rot, trans);

        /* Configure and render */
        configure_gpu(dev, idx_addr, idx_count, pos_addr, norm_addr, col_addr,
                     stride, buffer_phys, mv, p);

        pf_csr_start(dev->csr_base);

        if (!pixelforge_wait_for_gpu_ready(dev, GPU_STAGE_PER_PIXEL, &keep_running)) {
            fprintf(stderr, "Frame %d: GPU timeout\n", frame);
            break;
        }

        pixelforge_swap_buffers(dev);
    }

    pixelforge_close_dev(dev);
    printf("Done!\n");
    return 0;
}
