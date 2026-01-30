/*
 * PixelForge Demo: Alpha Blending Kaleidoscope
 *
 * This demo showcases:
 * - Source-alpha blending (src * a + dst * (1-a))
 * - Depth-tested translucent geometry with depth writes disabled
 * - Painter-style back-to-front ordering for clean composites
 * - An additive glow pass stacked on top for a neon highlight
 */

#define _GNU_SOURCE
#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stddef.h>

#include "graphics_pipeline_csr_access.h"
#include "vram_alloc.h"
#include "pixelforge_utils.h"
#include "demo_utils.h"
#include "frame_capture.h"

#define PAGE_SIZE 4096u
#define QUAD_VERTS 6u

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

static void fill_quad(vertex *out, float size, float alpha) {
    const float half = size * 0.5f;

    const float colors[4][4] = {
        {0.95f, 0.38f, 0.25f, alpha},
        {0.22f, 0.82f, 0.92f, alpha},
        {0.93f, 0.24f, 0.72f, alpha},
        {0.34f, 0.76f, 0.38f, alpha},
    };

    const float positions[4][3] = {
        {-half, -half, 0.0f},
        { half, -half, 0.0f},
        { half,  half, 0.0f},
        {-half,  half, 0.0f},
    };

    /* Two triangles: (0,1,2) and (0,2,3) */
    int order[QUAD_VERTS] = {0, 1, 2, 0, 2, 3};
    for (unsigned int i = 0; i < QUAD_VERTS; i++) {
        int idx = order[i];
        out[i].pos[0] = fp16_16(positions[idx][0]);
        out[i].pos[1] = fp16_16(positions[idx][1]);
        out[i].pos[2] = fp16_16(positions[idx][2]);
        out[i].pos[3] = fp16_16(1.0f);

        out[i].col[0] = fp16_16(colors[idx][0]);
        out[i].col[1] = fp16_16(colors[idx][1]);
        out[i].col[2] = fp16_16(colors[idx][2]);
        out[i].col[3] = fp16_16(colors[idx][3]);
    }
}

static void configure_gpu(
    pixelforge_dev *dev,
    uint32_t vb_addr,
    uint16_t stride,
    uint32_t color_addr,
    const float mv[16],
    const float p[16],
    const float *override_color,
    const pixelforge_blend_config_t *blend
) {
    volatile uint8_t *csr = dev->csr_base;

    pixelforge_idx_config_t idx_cfg = {
        .address = 0,
        .count = QUAD_VERTS,
        .kind = PIXELFORGE_INDEX_NOT_INDEXED,
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
        .info.per_vertex = {.address = vb_addr + offsetof(vertex, pos), .stride = stride},
    };
    pf_csr_set_attr_position(csr, &attr);

    /* Normal is unused; feed a constant so the slot is well-defined. */
    attr.mode = PIXELFORGE_ATTR_CONSTANT;
    attr.info.constant_value.value[0] = fp16_16(0.0f);
    attr.info.constant_value.value[1] = fp16_16(0.0f);
    attr.info.constant_value.value[2] = fp16_16(1.0f);
    attr.info.constant_value.value[3] = fp16_16(0.0f);
    pf_csr_set_attr_normal(csr, &attr);

    if (override_color) {
        attr.mode = PIXELFORGE_ATTR_CONSTANT;
        attr.info.constant_value.value[0] = fp16_16(override_color[0]);
        attr.info.constant_value.value[1] = fp16_16(override_color[1]);
        attr.info.constant_value.value[2] = fp16_16(override_color[2]);
        attr.info.constant_value.value[3] = fp16_16(override_color[3]);
    } else {
        attr.mode = PIXELFORGE_ATTR_PER_VERTEX;
        attr.info.per_vertex.address = vb_addr + offsetof(vertex, col);
        attr.info.per_vertex.stride = stride;
    }
    pf_csr_set_attr_color(csr, &attr);

    /* Transforms */
    pixelforge_vtx_xf_config_t xf = {0};
    xf.enabled.normal_enable = false;
    mat4_to_fp16_16(xf.position_mv, mv);
    mat4_to_fp16_16(xf.position_p, p);
    pf_csr_set_vtx_xf(csr, &xf);

    /* Flat shading driven by vertex colors. */
    pixelforge_material_t mat = {0};
    mat.ambient[0] = fp16_16(1.0f);
    mat.ambient[1] = fp16_16(1.0f);
    mat.ambient[2] = fp16_16(1.0f);
    pf_csr_set_material(csr, &mat);

    pixelforge_light_t light = {0};
    light.ambient[0] = fp16_16(1.0f);
    light.ambient[1] = fp16_16(1.0f);
    light.ambient[2] = fp16_16(1.0f);
    pf_csr_set_light0(csr, &light);

    pixelforge_prim_config_t prim = {
        .type = PIXELFORGE_PRIM_TRIANGLES,
        .cull = PIXELFORGE_CULL_NONE,
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

    pixelforge_depth_test_config_t depth = {
        .test_enabled = false,
        .write_enabled = false,
        .compare_op = PIXELFORGE_CMP_GREATER_OR_EQUAL,
    };
    pf_csr_set_depth(csr, &depth);

    pixelforge_stencil_op_config_t stencil = {
        .compare_op = PIXELFORGE_CMP_ALWAYS,
        .mask = 0xFF,
        .write_mask = 0x00,
    };
    pf_csr_set_stencil_front(csr, &stencil);
    pf_csr_set_stencil_back(csr, &stencil);

    if (blend) {
        pf_csr_set_blend(csr, blend);
    }
}

static void fill_gradient(uint8_t *buffer, pixelforge_dev *dev) {
    for (uint32_t y = 0; y < dev->y_resolution; y++) {
        float t = (float)y / (float)(dev->y_resolution - 1);
        uint8_t r = (uint8_t)(15 + 60 * (1.0f - t));
        uint8_t g = (uint8_t)(20 + 90 * t);
        uint8_t b = (uint8_t)(30 + 80 * (0.5f + 0.5f * sinf(t * 4.0f)));
        uint32_t *row = (uint32_t *)(buffer + y * dev->buffer_stride);
        uint32_t packed = (0xFFu << 24) | (r << 16) | (g << 8) | b;
        for (uint32_t x = 0; x < dev->x_resolution; x++) {
            row[x] = packed;
        }
    }
}

struct layer {
    float mv[16];
};

int main(int argc, char **argv) {
    int frames = 240;
    bool capture_frames = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--verbose")) g_verbose = true;
        else if (!strcmp(argv[i], "--capture-frames")) capture_frames = true;
        else if (!strcmp(argv[i], "--frames") && i + 1 < argc) frames = atoi(argv[++i]);
    }

    signal(SIGINT, handle_sigint);

    pixelforge_dev *dev = pixelforge_open_dev();
    if (!dev) {
        fprintf(stderr, "Failed to open device\n");
        return 1;
    }

    /* Vertex data: one quad (6 vertices), drawn repeatedly with different transforms. */
    vertex *vertices = NULL;
    if (posix_memalign((void **)&vertices, PAGE_SIZE, QUAD_VERTS * sizeof(vertex)) != 0) {
        vertices = NULL;
    }
    if (!vertices) {
        fprintf(stderr, "Vertex allocation failed\n");
        pixelforge_close_dev(dev);
        return 1;
    }
    fill_quad(vertices, 1.8f, 0.42f);

    /* Upload quad into VRAM. */
    size_t vb_size = QUAD_VERTS * sizeof(vertex);
    vb_size = (vb_size + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);

    struct vram_block vb_block;
    if (vram_alloc(&dev->vram, vb_size, PAGE_SIZE, &vb_block)) {
        fprintf(stderr, "VRAM allocation failed\n");
        free(vertices);
        pixelforge_close_dev(dev);
        return 1;
    }

    memcpy(vb_block.virt, vertices, QUAD_VERTS * sizeof(vertex));
    free(vertices);

    float p[16];
    mat4_perspective(p, 70.0f * M_PI / 180.0f, (float)dev->x_resolution / (float)dev->y_resolution, 0.1f, 60.0f);

    pixelforge_blend_config_t blend_alpha = {
        .src_factor = PIXELFORGE_BLEND_SRC_ALPHA,
        .dst_factor = PIXELFORGE_BLEND_ONE_MINUS_SRC_ALPHA,
        .src_a_factor = PIXELFORGE_BLEND_ONE,
        .dst_a_factor = PIXELFORGE_BLEND_ONE_MINUS_SRC_ALPHA,
        .enabled = true,
        .blend_op = PIXELFORGE_BLEND_ADD,
        .blend_a_op = PIXELFORGE_BLEND_ADD,
        .color_write_mask = 0xF,
    };

    pixelforge_blend_config_t blend_add = {
        .src_factor = PIXELFORGE_BLEND_SRC_ALPHA,
        .dst_factor = PIXELFORGE_BLEND_ONE,
        .src_a_factor = PIXELFORGE_BLEND_ONE,
        .dst_a_factor = PIXELFORGE_BLEND_ONE,
        .enabled = true,
        .blend_op = PIXELFORGE_BLEND_ADD,
        .blend_a_op = PIXELFORGE_BLEND_ADD,
        .color_write_mask = 0xF,
    };

    const float glow_color[4] = {0.2f, 0.3f, 0.2f, 1.0f};

    printf("PixelForge Alpha Blending Demo\n");
    printf("Rendering %d frames...\n", frames);

    for (int frame = 0; frame < frames && keep_running; frame++) {
        float t = (float)frame / 60.0f;

        uint8_t *buffer = pixelforge_get_back_buffer(dev);

        fill_gradient(buffer, dev);

        const int layer_count = 6;
        struct layer layers[layer_count];
        for (int i = 0; i < layer_count; i++) {
            float yaw = t * 0.7f + i * 0.5f;
            float radius = 1.4f + 0.2f * i + 0.15f * cosf(t * 0.9f + i * 0.3f);

            float rot[16], trans[16];
            mat4_rotate_xyz(rot, 0.0f, 0.0f, yaw);
            mat4_translate(trans, cosf(yaw) * radius, sinf(yaw) * radius, -4.0f);
            mat4_multiply(layers[i].mv, rot, trans);
        }

        /* Translucent layers. */
        for (int i = 0; i < layer_count && keep_running; i++) {
            configure_gpu(dev, vb_block.phys, sizeof(vertex),
                          dev->buffer_phys[dev->render_buffer],
                          layers[i].mv, p, NULL, &blend_alpha);
            pf_csr_start(dev->csr_base);

            if (!pixelforge_wait_for_gpu_ready(dev, GPU_STAGE_VTX_TRANSFORM, &keep_running)) {
                fprintf(stderr, "Frame %d: GPU timeout\n", frame);
                break;
            }
        }

        if (!pixelforge_wait_for_gpu_ready(dev, GPU_STAGE_PER_PIXEL, &keep_running)) {
            fprintf(stderr, "Frame %d: GPU timeout\n", frame);
            break;
        }


        float glow_mv[16], spin[16], trans[16];
        mat4_rotate_xyz(spin, 0.0f, 0.0f, t * 1.6f);
        mat4_translate(trans, 0.0f, 0.0f, -3.8f);
        mat4_multiply(glow_mv, spin, trans);

        configure_gpu(dev, vb_block.phys, sizeof(vertex),
                  dev->buffer_phys[dev->render_buffer],
                  glow_mv, p, glow_color, &blend_add);
        pf_csr_start(dev->csr_base);

        if (!pixelforge_wait_for_gpu_ready(dev, GPU_STAGE_PER_PIXEL, &keep_running)) {
            fprintf(stderr, "Frame %d: GPU timeout\n", frame);
            break;
        }

        pixelforge_swap_buffers(dev);

        if (capture_frames) {
            char filename[256];
            if (frame_capture_gen_filename(filename, sizeof(filename), "alpha", frame, ".png") == 0) {
                uint8_t *display_buffer = pixelforge_get_front_buffer(dev);
                frame_capture_rgba(filename, display_buffer, dev->x_resolution,
                                 dev->y_resolution, dev->buffer_stride);
            }
        }

        printf("Frame %d/%d rendered (alpha blend)\n", frame + 1, frames);
    }

    pixelforge_close_dev(dev);
    printf("Done!\n");
    return 0;
}
