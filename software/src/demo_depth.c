/*
 * PixelForge Demo: Depth Test with Occluding Cubes
 *
 * This demo showcases:
 * - Depth buffer usage and depth testing
 * - Multiple objects at different depths
 * - Proper occlusion (front objects hide back objects)
 * - Depth write and compare operations
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
#define VB_REGION_SIZE  0x00020000u

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

enum gpu_stage {
    GPU_STAGE_IA = 0,
    GPU_STAGE_VTX_TRANSFORM = 1,
    GPU_STAGE_PREP_RASTER = 2,
    GPU_STAGE_PER_PIXEL = 3,
};

static int wait_for_gpu_ready(pixelforge_dev *dev, enum gpu_stage stage) {
    // component is really ready if all prior stages are also ready
    uint32_t mask = (1u << (stage + 1)) - 1;

    for (int i = 0; i < 10000000 && keep_running; ++i) {
        uint32_t ready_components = pf_csr_get_ready_components(dev->csr_base);
        if ((ready_components & mask) == mask) return 0;
        usleep(50);
    }
    return -1;
}

/* Create cube geometry */
static void create_cube(struct vertex *vertices, uint16_t *indices, uint32_t *idx_count,
                       float r, float g, float b) {
    /* 8 vertices of a unit cube centered at origin */
    float verts[8][3] = {
        {-0.5f, -0.5f, -0.5f}, { 0.5f, -0.5f, -0.5f},
        { 0.5f,  0.5f, -0.5f}, {-0.5f,  0.5f, -0.5f},
        {-0.5f, -0.5f,  0.5f}, { 0.5f, -0.5f,  0.5f},
        { 0.5f,  0.5f,  0.5f}, {-0.5f,  0.5f,  0.5f}
    };

    for (int i = 0; i < 8; i++) {
        vertices[i].pos[0] = fp16_16(verts[i][0]);
        vertices[i].pos[1] = fp16_16(verts[i][1]);
        vertices[i].pos[2] = fp16_16(verts[i][2]);
        vertices[i].pos[3] = fp16_16(1.0f);

        /* Simple normals (will be per-face ideally, but per-vertex for simplicity) */
        float len = sqrtf(verts[i][0]*verts[i][0] + verts[i][1]*verts[i][1] + verts[i][2]*verts[i][2]);
        vertices[i].norm[0] = fp16_16(verts[i][0] / len);
        vertices[i].norm[1] = fp16_16(verts[i][1] / len);
        vertices[i].norm[2] = fp16_16(verts[i][2] / len);

        vertices[i].col[0] = fp16_16(r);
        vertices[i].col[1] = fp16_16(g);
        vertices[i].col[2] = fp16_16(b);
        vertices[i].col[3] = fp16_16(1.0f);
    }

    /* 12 triangles (2 per face, 6 faces) */
    uint16_t faces[12][3] = {
        /* Front face */
        {0, 1, 2}, {0, 2, 3},
        /* Back face */
        {5, 4, 7}, {5, 7, 6},
        /* Top face */
        {3, 2, 6}, {3, 6, 7},
        /* Bottom face */
        {4, 5, 1}, {4, 1, 0},
        /* Right face */
        {1, 5, 6}, {1, 6, 2},
        /* Left face */
        {4, 0, 3}, {4, 3, 7}
    };

    for (int i = 0; i < 12; i++) {
        indices[i*3 + 0] = faces[i][0];
        indices[i*3 + 1] = faces[i][1];
        indices[i*3 + 2] = faces[i][2];
    }

    *idx_count = 36; /* 12 triangles * 3 indices */
}

static void configure_gpu(pixelforge_dev *dev, uint32_t idx_addr, uint32_t idx_count,
                         uint32_t pos_addr, uint32_t norm_addr, uint32_t col_addr,
                         uint16_t stride, uint32_t color_addr, uint32_t ds_addr,
                         const float mv[16], const float p[16], bool clear_depth) {
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

    /* Material: ambient only (simple shading) */
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

    /* Depth test enabled */
    pixelforge_depth_test_config_t depth = {
        .test_enabled = true,
        .write_enabled = true,
        .compare_op = PIXELFORGE_CMP_GREATER_OR_EQUAL,
    };
    pf_csr_set_depth(csr, &depth);

    pixelforge_stencil_op_config_t stencil = {0};
    stencil.compare_op = PIXELFORGE_CMP_ALWAYS;
    stencil.reference = 0;
    stencil.mask = 0xFF;
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

int main(int argc, char **argv) {
    int frames = 120;

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

    printf("PixelForge Depth Test Demo: Occluding Cubes\n");
    printf("Rendering %d frames...\n", frames);

    /* Allocate buffers */
    struct vram_block vb_block, ds_block;
    if (vram_alloc(&dev->vram, VB_REGION_SIZE, PAGE_SIZE, &vb_block) ||
        vram_alloc(&dev->vram, dev->x_resolution * dev->y_resolution * 4, PAGE_SIZE, &ds_block)) {
        fprintf(stderr, "VRAM allocation failed\n");
        pixelforge_close_dev(dev);
        return 1;
    }

    /* Create 4 cubes with different colors - for orbital scene */
    struct vertex *cubes[4];
    uint16_t *indices[4];
    float colors[4][3] = {
        {1.0f, 0.2f, 0.2f},  /* Red */
        {0.2f, 1.0f, 0.2f},  /* Green */
        {0.2f, 0.2f, 1.0f},  /* Blue */
        {1.0f, 1.0f, 0.2f}   /* Yellow */
    };

    uint32_t idx_count;
    for (int i = 0; i < 4; i++) {
        cubes[i] = (struct vertex*)(vb_block.virt + i * 1024);
        indices[i] = (uint16_t*)(vb_block.virt + (4 + i) * 1024);
        create_cube(cubes[i], indices[i], &idx_count, colors[i][0], colors[i][1], colors[i][2]);
    }

    /* Projection matrix */
    float p[16];
    mat4_perspective(p, 60.0f * M_PI / 180.0f, (float)dev->x_resolution / (float)dev->y_resolution, 0.1f, 10.0f);

    /* Animation loop */
    for (int frame = 0; frame < frames && keep_running; frame++) {
        float t = (float)frame / 30.0f;

        /* Get back buffer for rendering */
        uint8_t *buffer = pixelforge_get_back_buffer(dev);
        uint32_t buffer_phys = dev->buffer_phys[dev->render_buffer];

        /* Clear buffers */
        memset(buffer, 0x00, dev->buffer_size);
        memset(ds_block.virt, 0x00, dev->x_resolution * dev->y_resolution * 4);

        /* Render 4 cubes orbiting in a circle at varying depths
         * They circle around the camera at z=-2.5, creating occlusion */
        float radius = 1.5f;

        for (int i = 0; i < 4; i++) {
            /* Each cube is offset by 90 degrees around the circle */
            float angle = t + (i * M_PI / 2.0f);
            float x = radius * cosf(angle);
            float z = -2.5f + radius * sinf(angle);  /* Z oscillates to create depth variation */

            float mv[16], trans[16], rot[16], scale[16];

            /* Rotate cube slightly to see all faces */
            mat4_rotate_xyz(rot, angle * 0.5f, angle, 0.0f);

            /* Translate to orbit position */
            mat4_translate(trans, x, 0.0f, z);

            /* Scale slightly smaller than in previous demo */
            mat4_scale(scale, 0.5f, 0.5f, 0.5f);

            /* Apply transformations: scale -> rotate -> translate */
            mat4_multiply(mv, scale, rot);
            mat4_multiply(mv, mv, trans);

            configure_gpu(dev, vb_block.phys + (4 + i) * 1024, idx_count,
                         vb_block.phys + i * 1024 + offsetof(struct vertex, pos),
                         vb_block.phys + i * 1024 + offsetof(struct vertex, norm),
                         vb_block.phys + i * 1024 + offsetof(struct vertex, col),
                         sizeof(struct vertex), buffer_phys, ds_block.phys, mv, p, (i == 0));
            pf_csr_start(dev->csr_base);

            // We can send next cube right after the previous one finishes vertex transform
            // -> we overlap rasterization of previous cube with transform of next cube
            if (wait_for_gpu_ready(dev, GPU_STAGE_VTX_TRANSFORM) != 0) break;
        }

        if (wait_for_gpu_ready(dev, GPU_STAGE_PER_PIXEL) != 0) {
            fprintf(stderr, "Frame %d: GPU timeout\n", frame);
            break;
        }
        pixelforge_swap_buffers(dev);
        printf("Frame %d/%d rendered\n", frame + 1, frames);
    }

    pixelforge_close_dev(dev);
    printf("Done!\n");
    return 0;
}
