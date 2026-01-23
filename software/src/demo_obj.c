/*
 * PixelForge Demo: OBJ Model Viewer
 *
 * This demo showcases:
 * - Loading 3D models from OBJ files
 * - Automatic model scaling and centering
 * - Rotation animation
 * - Depth testing with loaded geometry
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
#include "obj_loader.h"

#define PAGE_SIZE       4096u

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

/* Convert OBJ model to GPU vertex format, duplicating vertices for per-face normals */
static size_t convert_obj_to_vertices(const obj_model *model, struct vertex **out_vertices,
                                      vec3f *model_center, float *model_scale) {
    /* Get bounding box and compute center + scale */
    vec3f min, max;
    obj_get_bounds(model, &min, &max);

    model_center->x = (min.x + max.x) * 0.5f;
    model_center->y = (min.y + max.y) * 0.5f;
    model_center->z = (min.z + max.z) * 0.5f;

    float size_x = max.x - min.x;
    float size_y = max.y - min.y;
    float size_z = max.z - min.z;
    float max_size = size_x > size_y ? size_x : size_y;
    max_size = max_size > size_z ? max_size : size_z;
    *model_scale = max_size > 0.0f ? 2.0f / max_size : 1.0f;  /* Scale to fit in ~2 unit box */

    printf("Model bounds: (%.2f,%.2f,%.2f) to (%.2f,%.2f,%.2f)\n",
           min.x, min.y, min.z, max.x, max.y, max.z);
    printf("Model center: (%.2f,%.2f,%.2f), scale: %.2f\n",
           model_center->x, model_center->y, model_center->z, *model_scale);

    /* Duplicate vertices for each face to handle per-face normals correctly */
    size_t num_triangles = model->num_faces / 3;
    size_t num_vertices = model->num_faces;  /* One unique vertex per face-vertex */

    *out_vertices = malloc(num_vertices * sizeof(struct vertex));

    if (!*out_vertices) {
        return 0;
    }

    /* Create a unique vertex for each face-vertex combination */
    for (size_t i = 0; i < model->num_faces; i++) {
        const face_vertex *fv = &model->faces[i];
        struct vertex *v = &(*out_vertices)[i];

        /* Get position and transform it */
        if ((size_t)fv->v_idx < model->num_positions) {
            const vec3f *pos = &model->positions[fv->v_idx];

            /* Center and scale model */
            float x = (pos->x - model_center->x) * (*model_scale);
            float y = (pos->y - model_center->y) * (*model_scale);
            float z = (pos->z - model_center->z) * (*model_scale);

            v->pos[0] = fp16_16(x);
            v->pos[1] = fp16_16(y);
            v->pos[2] = fp16_16(z);
            v->pos[3] = fp16_16(1.0f);
        }

        /* Set normal if available */
        if (fv->vn_idx >= 0 && (size_t)fv->vn_idx < model->num_normals) {
            const vec3f *n = &model->normals[fv->vn_idx];
            v->norm[0] = fp16_16(n->x);
            v->norm[1] = fp16_16(n->y);
            v->norm[2] = fp16_16(n->z);
        } else {
            /* Default normal if not specified */
            v->norm[0] = fp16_16(0.0f);
            v->norm[1] = fp16_16(0.0f);
            v->norm[2] = fp16_16(1.0f);
        }

        /* Default white color */
        v->col[0] = fp16_16(0.8f);
        v->col[1] = fp16_16(0.8f);
        v->col[2] = fp16_16(0.8f);
        v->col[3] = fp16_16(1.0f);
    }

    printf("Converted to %zu vertices (duplicated for per-face normals), %zu triangles\n",
           num_vertices, num_triangles);

    return num_vertices;  /* Return number of vertices */
}

static void configure_gpu(pixelforge_dev *dev, uint32_t vertex_count,
                         uint32_t pos_addr, uint32_t norm_addr, uint32_t col_addr,
                         uint16_t stride, uint32_t color_addr, uint32_t ds_addr,
                         const float mv[16], const float p[16]) {
    volatile uint8_t *csr = dev->csr_base;

    pixelforge_idx_config_t idx_cfg = {
        .address = 0,
        .count = vertex_count,
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

    /* Material: simple ambient + diffuse */
    pixelforge_material_t mat = {0};
    mat.ambient[0] = fp16_16(0.3f);
    mat.ambient[1] = fp16_16(0.3f);
    mat.ambient[2] = fp16_16(0.3f);
    mat.diffuse[0] = fp16_16(0.7f);
    mat.diffuse[1] = fp16_16(0.7f);
    mat.diffuse[2] = fp16_16(0.7f);
    mat.specular[0] = fp16_16(0.2f);
    mat.specular[1] = fp16_16(0.2f);
    mat.specular[2] = fp16_16(0.2f);
    mat.shininess = fp16_16(32.0f);
    pf_csr_set_material(csr, &mat);

    /* Light */
    pixelforge_light_t light = {0};
    light.position[0] = fp16_16(1.0f);
    light.position[1] = fp16_16(1.0f);
    light.position[2] = fp16_16(1.0f);
    light.position[3] = fp16_16(0.0f);
    light.ambient[0] = fp16_16(0.3f);
    light.ambient[1] = fp16_16(0.3f);
    light.ambient[2] = fp16_16(0.3f);
    light.diffuse[0] = fp16_16(0.7f);
    light.diffuse[1] = fp16_16(0.7f);
    light.diffuse[2] = fp16_16(0.7f);
    light.specular[0] = fp16_16(0.5f);
    light.specular[1] = fp16_16(0.5f);
    light.specular[2] = fp16_16(0.5f);
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

static void set_stencil_write_mode(pixelforge_dev *dev) {
    volatile uint8_t *csr = dev->csr_base;

    pixelforge_stencil_op_config_t stencil = {
        .compare_op = PIXELFORGE_CMP_ALWAYS,
        .reference = 1,
        .mask = 0xFF,
        .write_mask = 0xFF,
        .fail_op = PIXELFORGE_STENCIL_KEEP,
        .depth_fail_op = PIXELFORGE_STENCIL_KEEP,
        .pass_op = PIXELFORGE_STENCIL_REPLACE,
    };
    pf_csr_set_stencil_front(csr, &stencil);
    pf_csr_set_stencil_back(csr, &stencil);
}

static void set_stencil_outline_mode(pixelforge_dev *dev) {
    volatile uint8_t *csr = dev->csr_base;

    pixelforge_stencil_op_config_t stencil = {
        .compare_op = PIXELFORGE_CMP_NOT_EQUAL,
        .reference = 1,
        .mask = 0xFF,
        .write_mask = 0x00,
        .fail_op = PIXELFORGE_STENCIL_KEEP,
        .depth_fail_op = PIXELFORGE_STENCIL_KEEP,
        .pass_op = PIXELFORGE_STENCIL_KEEP,
    };
    pf_csr_set_stencil_front(csr, &stencil);
    pf_csr_set_stencil_back(csr, &stencil);

    pixelforge_light_t light = {
        .ambient = { fp16_16(1.0f), fp16_16(1.0f), fp16_16(1.0f) },
        .diffuse = { fp16_16(0.0f), fp16_16(0.0f), fp16_16(0.0f) },
        .specular = { fp16_16(0.0f), fp16_16(0.0f), fp16_16(0.0f) },
    };
    pf_csr_set_light0(csr, &light);

    pixelforge_material_t mat = {
        .ambient = { fp16_16(1.0f), fp16_16(1.0f), fp16_16(1.0f) },
        .diffuse = { fp16_16(0.0f), fp16_16(0.0f), fp16_16(0.0f) },
        .specular = { fp16_16(0.0f), fp16_16(0.0f), fp16_16(0.0f) },
        .shininess = fp16_16(1.0f),
    };
    pf_csr_set_material(csr, &mat);
}

static void set_object_color(pixelforge_dev *dev, float r, float g, float b, float a) {
    volatile uint8_t *csr = dev->csr_base;

    pixelforge_input_attr_t attr = {
        .mode = PIXELFORGE_ATTR_CONSTANT,
        .info.constant_value.value = { fp16_16(r), fp16_16(g), fp16_16(b), fp16_16(a) },
    };
    pf_csr_set_attr_color(csr, &attr);
}

int main(int argc, char **argv) {
    int frames = 120;
    const char *obj_file = NULL;
    bool stencil_outline = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--verbose")) g_verbose = true;
        else if (!strcmp(argv[i], "--frames") && i + 1 < argc) frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--stencil-outline")) stencil_outline = true;
        else if (!strcmp(argv[i], "--obj") && i + 1 < argc) obj_file = argv[++i];
        else if (argv[i][0] != '-') obj_file = argv[i];
    }

    if (!obj_file) {
        fprintf(stderr, "Usage: %s [--verbose] [--frames N] [--stencil-outline] <model.obj>\n", argv[0]);
        return 1;
    }

    signal(SIGINT, handle_sigint);

    /* Load OBJ model */
    obj_model model;
    if (obj_load(obj_file, &model) != 0) {
        fprintf(stderr, "Failed to load OBJ file: %s\n", obj_file);
        return 1;
    }

    /* Convert to GPU format */
    struct vertex *vertices = NULL;
    vec3f center;
    float scale;
    size_t vertex_count = convert_obj_to_vertices(&model, &vertices, &center, &scale);

    if (vertex_count == 0) {
        fprintf(stderr, "Failed to convert model\n");
        obj_free(&model);
        return 1;
    }

    pixelforge_dev *dev = pixelforge_open_dev();
    if (!dev) {
        fprintf(stderr, "Failed to open device\n");
        free(vertices);
        obj_free(&model);
        return 1;
    }

    printf("PixelForge OBJ Model Viewer: %s\n", obj_file);
    printf("Rendering %d frames...\n", frames);

    /* Calculate buffer sizes */
    size_t vb_size = vertex_count * sizeof(struct vertex);
    vb_size = (vb_size + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);  /* Align to page */

    /* Allocate buffers */
    struct vram_block vb_block, ds_block;
    if (vram_alloc(&dev->vram, vb_size, PAGE_SIZE, &vb_block) ||
        vram_alloc(&dev->vram, dev->x_resolution * dev->y_resolution * 4, PAGE_SIZE, &ds_block)) {
        fprintf(stderr, "VRAM allocation failed\n");
        pixelforge_close_dev(dev);
        free(vertices);
        obj_free(&model);
        return 1;
    }

    /* Copy geometry to VRAM */
    memcpy(vb_block.virt, vertices, vertex_count * sizeof(struct vertex));

    free(vertices);
    obj_free(&model);

    /* Projection matrix */
    float p[16];
    mat4_perspective(p, 60.0f * M_PI / 180.0f, (float)dev->x_resolution / (float)dev->y_resolution, 0.1f, 100.0f);

    /* Animation loop */
    for (int frame = 0; frame < frames && keep_running; frame++) {
        float t = (float)frame / 30.0f;

        /* Get back buffer for rendering */
        uint8_t *buffer = pixelforge_get_back_buffer(dev);
        uint32_t buffer_phys = dev->buffer_phys[dev->render_buffer];

        /* Clear buffers */
        memset(buffer, 0x00, dev->buffer_size);
        memset(ds_block.virt, 0x00, dev->x_resolution * dev->y_resolution * 4);

        /* Modelview matrix: rotate model */
        float mv[16], rot[16], trans[16];
        mat4_rotate_xyz(rot, t * 0.2f, t * 0.5f, 0.0f);
        mat4_translate(trans, 0.0f, 0.0f, -5.0f);
        mat4_multiply(mv, rot, trans);

        if (!stencil_outline) {
            configure_gpu(dev, vertex_count,
                         vb_block.phys + offsetof(struct vertex, pos),
                         vb_block.phys + offsetof(struct vertex, norm),
                         vb_block.phys + offsetof(struct vertex, col),
                         sizeof(struct vertex), buffer_phys, ds_block.phys, mv, p);
            pf_csr_start(dev->csr_base);
            if (wait_for_gpu_ready(dev) != 0) break;
        } else {
            /* Pass 1: draw model and write stencil */
            configure_gpu(dev, vertex_count,
                         vb_block.phys + offsetof(struct vertex, pos),
                         vb_block.phys + offsetof(struct vertex, norm),
                         vb_block.phys + offsetof(struct vertex, col),
                         sizeof(struct vertex), buffer_phys, ds_block.phys, mv, p);
            set_stencil_write_mode(dev);
            pf_csr_start(dev->csr_base);
            if (wait_for_gpu_ready(dev) != 0) break;

            /* Pass 2: draw enlarged model where stencil != 1 (outline) */
            float scale_m[16], mv_outline[16];
            mat4_scale(scale_m, 1.15f, 1.15f, 1.15f);
            mat4_multiply(mv_outline, scale_m, mv);

            configure_gpu(dev, vertex_count,
                         vb_block.phys + offsetof(struct vertex, pos),
                         vb_block.phys + offsetof(struct vertex, norm),
                         vb_block.phys + offsetof(struct vertex, col),
                         sizeof(struct vertex), buffer_phys, ds_block.phys, mv_outline, p);
            set_stencil_outline_mode(dev);
            set_object_color(dev, 1.0f, 0.8f, 0.0f, 1.0f);

            pixelforge_depth_test_config_t depth = {
                .test_enabled = false,
                .write_enabled = false,
                .compare_op = PIXELFORGE_CMP_ALWAYS,
            };
            pf_csr_set_depth(dev->csr_base, &depth);

            pf_csr_start(dev->csr_base);
            if (wait_for_gpu_ready(dev) != 0) break;
        }

        pixelforge_swap_buffers(dev);
        if (stencil_outline) printf("Frame %d/%d rendered (stencil-outline)\n", frame + 1, frames);
        else printf("Frame %d/%d rendered\n", frame + 1, frames);
    }

    pixelforge_close_dev(dev);
    printf("Done!\n");
    return 0;
}
