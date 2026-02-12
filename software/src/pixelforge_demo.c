

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
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
#include <sys/syscall.h>

#include "graphics_pipeline_csr_access.h"
#include "vram_alloc.h"
#include "pixelforge_utils.h"
#include "demo_utils.h"
#include "frame_capture.h"

/*
 * PixelForge demo inspired by Altera's video DMA controller API.
 * Provides clean buffer management with device structure and helpers for:
 * - Opening/initializing video framebuffer device
 * - Setting back buffer address
 * - Swapping buffers
 * - Screen clear and fill operations
 */

#define PAGE_SIZE       4096u
#define VB_REGION_SIZE  0x00010000u

static volatile bool keep_running = true;
static bool g_verbose = false;
static bool g_throttle = false;

#define DBG(fmt, ...) do { \
    if (g_verbose) fprintf(stderr, "[dbg] " fmt "\n", ##__VA_ARGS__); \
    if (g_throttle) usleep(300000); \
} while (0)

static void handle_sigint(int sig) {
    (void)sig;
    keep_running = false;
}

/*
 * Fixed-point 16.16 conversion
 */
static int32_t fp16_16(float v) {
    return (int32_t)(v * 65536.0f);
}

/*
 * Vertex structure
 */
struct vertex {
    int32_t pos[4];
    int32_t norm[3];
    int32_t col[4];
};

/*
 * Setup a simple triangle in vertex buffer
 */
static void setup_triangle_geometry(uint32_t vb_phys, uint8_t *vb_virt,
                                    uint32_t *idx_addr, uint32_t *idx_count,
                                    uint32_t *pos_addr, uint32_t *norm_addr,
                                    uint32_t *col_addr, uint16_t *stride) {
    struct vertex *v = (struct vertex*)vb_virt;

    /* RGB triangle */
    v[0] = (struct vertex){
        {fp16_16(-0.7f), fp16_16(-0.7f), fp16_16(0.2f), fp16_16(1.0f)},
        {fp16_16(0.0f), fp16_16(0.0f), fp16_16(1.0f)},
        {fp16_16(1.0f), fp16_16(0.0f), fp16_16(0.0f), fp16_16(1.0f)},
    };
    v[1] = (struct vertex){
        {fp16_16(0.7f), fp16_16(-0.7f), fp16_16(0.2f), fp16_16(1.0f)},
        {fp16_16(0.0f), fp16_16(0.0f), fp16_16(1.0f)},
        {fp16_16(0.0f), fp16_16(1.0f), fp16_16(0.0f), fp16_16(1.0f)},
    };
    v[2] = (struct vertex){
        {fp16_16(0.0f), fp16_16(0.7f), fp16_16(0.2f), fp16_16(1.0f)},
        {fp16_16(0.0f), fp16_16(0.0f), fp16_16(1.0f)},
        {fp16_16(0.0f), fp16_16(0.0f), fp16_16(1.0f), fp16_16(1.0f)},
    };
    v[3] = (struct vertex){
        {fp16_16(0.0f), fp16_16(0.0f), fp16_16(0.0f), fp16_16(1.0f)},
        {fp16_16(0.0f), fp16_16(0.0f), fp16_16(1.0f)},
        {fp16_16(1.0f), fp16_16(1.0f), fp16_16(1.0f), fp16_16(1.0f)},
    };

    uint16_t *indices = (uint16_t*)(vb_virt + sizeof(struct vertex) * 4);
    indices[0] = 0;
    indices[1] = 1;
    indices[2] = 2;
    indices[3] = 3;

    *idx_addr = vb_phys + sizeof(struct vertex) * 4;
    *idx_count = 4;
    *stride = sizeof(struct vertex);
    *pos_addr = vb_phys + offsetof(struct vertex, pos);
    *norm_addr = vb_phys + offsetof(struct vertex, norm);
    *col_addr = vb_phys + offsetof(struct vertex, col);
}

/*
 * Configure GPU CSRs for rendering
 */
static void configure_gpu_pipeline(pixelforge_dev *dev,
                                   uint32_t idx_addr, uint32_t idx_count,
                                   uint32_t pos_addr, uint32_t norm_addr,
                                   uint32_t col_addr, uint16_t stride,
                                   uint32_t color_addr) {
    volatile uint8_t *csr = dev->csr_base;

    /* Index buffer */
    pixelforge_idx_config_t idx_cfg = {
        .address = idx_addr,
        .count = idx_count,
        .kind = PIXELFORGE_INDEX_U16,
    };
    pf_csr_set_idx(csr, &idx_cfg);
    DBG("Index buffer: addr=0x%08x count=%u", idx_addr, idx_count);

    /* Topology */
    pixelforge_topo_config_t topo = {
        .input_topology = PIXELFORGE_TOPOLOGY_TRIANGLE_STRIP,
        .primitive_restart_enable = false,
        .primitive_restart_index = 0,
        .base_vertex = 0,
    };
    pf_csr_set_topology(csr, &topo);
    DBG("Topology configured: TRIANGLE_LIST");

    /* Vertex attributes */
    pixelforge_input_attr_t pos_attr = {
        .mode = PIXELFORGE_ATTR_PER_VERTEX,
        .info.per_vertex = { .address = pos_addr, .stride = stride },
    };
    pixelforge_input_attr_t norm_attr = pos_attr;
    norm_attr.info.per_vertex.address = norm_addr;
    pixelforge_input_attr_t col_attr = pos_attr;
    col_attr.info.per_vertex.address = col_addr;

    pf_csr_set_attr_position(csr, &pos_attr);
    pf_csr_set_attr_normal(csr, &norm_attr);
    pf_csr_set_attr_color(csr, &col_attr);
    DBG("Vertex attributes set: pos=0x%08x norm=0x%08x col=0x%08x stride=%u",
        pos_addr, norm_addr, col_addr, stride);

    /* test if vertex atts were set */
    pixelforge_input_attr_t test_attr;
    pf_csr_get_attr_position(csr, &test_attr);
    DBG("Verified position attribute: addr=0x%08x stride=%u",
        test_attr.info.per_vertex.address, test_attr.info.per_vertex.stride);
    pf_csr_get_attr_normal(csr, &test_attr);
    DBG("Verified normal attribute: addr=0x%08x stride=%u",
        test_attr.info.per_vertex.address, test_attr.info.per_vertex.stride);
    pf_csr_get_attr_color(csr, &test_attr);
    DBG("Verified color attribute: addr=0x%08x stride=%u",
        test_attr.info.per_vertex.address, test_attr.info.per_vertex.stride);

    /* Identity transforms */
    float id[16];
    mat4_identity(id);

    pixelforge_vtx_xf_config_t xf = {0};
    xf.enabled.normal_enable = false;
    mat4_to_fp16_16(xf.position_mv, id);
    mat4_to_fp16_16(xf.position_p, id);
    pf_csr_set_vtx_xf(csr, &xf);
    DBG("Vertex transforms set to identity");

    /* Material and lighting */
    pixelforge_material_t mat = {0};
    for (int i = 0; i < 3; ++i) {
        mat.ambient[i] = fp16_16(1.0f);
        mat.diffuse[i] = fp16_16(0.0f);
        mat.specular[i] = fp16_16(0.0f);
    }
    mat.shininess = fp16_16(1.0f);
    pf_csr_set_material(csr, &mat);
    DBG("Material set: ambient=1.0 diffuse=0.0 specular=0.0 shininess=1.0");

    pixelforge_light_t light = {0};
    light.position[0] = fp16_16(0.0f);
    light.position[1] = fp16_16(0.0f);
    light.position[2] = fp16_16(1.0f);
    light.position[3] = fp16_16(1.0f);
    for (int i = 0; i < 3; ++i) {
        light.ambient[i] = fp16_16(1.0f);
        light.diffuse[i] = fp16_16(0.0f);
        light.specular[i] = fp16_16(0.0f);
    }
    pf_csr_set_light(csr, 0, &light);
    DBG("Light 0 set: pos=(0,0,1) ambient=1.0 diffuse=0.0 specular=0.0");

    /* Primitive config */
    pixelforge_prim_config_t prim = {
        .type = PIXELFORGE_PRIM_TRIANGLES,
        .cull = PIXELFORGE_CULL_NONE,
        .winding = PIXELFORGE_WINDING_CCW,
    };
    pf_csr_set_prim(csr, &prim);
    DBG("Primitive config set: TRIANGLES, CULL_NONE, WINDING_CCW");

    /* Framebuffer */
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
    fb.depthstencil_address = 0; /* disable depth/stencil */
    fb.depthstencil_pitch = 0; /* D16_X8_S8 combined */
    pf_csr_set_fb(csr, &fb);
    DBG("Framebuffer configured: %ux%u color_addr=0x%08x depthstencil_addr=0x%08x",
        fb.width, fb.height, fb.color_address, fb.depthstencil_address);

    /* Depth test disabled, keep compare op defined */
    pixelforge_depth_test_config_t depth_cfg = {
        .test_enabled = false,
        .write_enabled = false,
        .compare_op = PIXELFORGE_CMP_ALWAYS,
    };
    pf_csr_set_depth(csr, &depth_cfg);
    DBG("Depth test disabled; compare=ALWAYS");

    /* Stencil: explicitly set masks and compare op to ALWAYS */
    pixelforge_stencil_op_config_t stencil_cfg = {0};
    stencil_cfg.compare_op = PIXELFORGE_CMP_ALWAYS;
    stencil_cfg.reference = 0x00;
    stencil_cfg.mask = 0xFF;        /* read mask */
    stencil_cfg.write_mask = 0xFF;  /* write mask */
    stencil_cfg.fail_op = PIXELFORGE_STENCIL_KEEP;
    stencil_cfg.depth_fail_op = PIXELFORGE_STENCIL_KEEP;
    stencil_cfg.pass_op = PIXELFORGE_STENCIL_KEEP;
    pf_csr_set_stencil_front(csr, &stencil_cfg);
    pf_csr_set_stencil_back(csr, &stencil_cfg);
    DBG("Stencil set: compare=ALWAYS, masks=FF/FF, ops=KEEP");

    /* Blending disabled */
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
    DBG("Blending disabled");
}

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "  --clear-test          Fill screen with test pattern and exit\n");
    fprintf(stderr, "  --xor-test            Fill screen with XOR pattern and exit\n");
    fprintf(stderr, "  --render-triangle     Render triangle using GPU pipeline\n");
    fprintf(stderr, "  --frames N            Render N frames (default: 1)\n");
    fprintf(stderr, "  --verbose             Enable debug output\n");
    fprintf(stderr, "  --throttle            Throttle debug output with delays\n");
    fprintf(stderr, "  --front               Operate on front buffer instead of back buffer\n");
}

int main(int argc, char **argv) {
    bool clear_test = false;
    bool xor_test = false;
    bool render_triangle = false;
    bool front = false;
    bool capture_frames = false;
    int frames = 1;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--clear-test")) {
            clear_test = 1;
        } else if (!strcmp(argv[i], "--xor-test")) {
            xor_test = 1;
        } else if (!strcmp(argv[i], "--capture-frames")) {
            capture_frames = true;
        } else if (!strcmp(argv[i], "--render-triangle")) {
            render_triangle = 1;
        } else if (!strcmp(argv[i], "--frames") && i + 1 < argc) {
            frames = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--verbose")) {
            g_verbose = 1;
        } else if (!strcmp(argv[i], "--front")) {
            front = true;
        } else if (!strcmp(argv[i], "--throttle")) {
            g_throttle = true;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (!clear_test && !render_triangle && !xor_test) {
        fprintf(stderr, "Error: specify --clear-test or --render-triangle or --xor-test\n\n");
        usage(argv[0]);
        return 1;
    }

    signal(SIGINT, handle_sigint);

    pixelforge_dev *dev = pixelforge_open_dev();
    if (!dev) {
        fprintf(stderr, "Failed to open PixelForge device\n");
        return 1;
    }

    printf("PixelForge device: %ux%u, %u bytes/pixel\n",
           dev->x_resolution, dev->y_resolution, dev->data_width);

    /* Simple clear/fill test */
    if (clear_test || xor_test) {
        printf("Clear/XOR test: filling screen with XOR pattern...\n");
        /* Get back buffer to work with */
        uint32_t *pixels;
        if (front) {
            pixels = (uint32_t*)pixelforge_get_front_buffer(dev);
        } else {
            pixels = (uint32_t*)pixelforge_get_back_buffer(dev);
        }

        if (xor_test) {
            for (uint32_t y = 0; y < dev->y_resolution; ++y) {
                for (uint32_t x = 0; x < dev->x_resolution; ++x) {
                    uint8_t r = x ^ y;
                    uint8_t g = (x * 3) ^ (y * 7);
                    uint8_t b = (x * 5) ^ (y * 11);
                    pixels[y * dev->x_resolution + x] =
                        (0xFF << 24) | (r << 16) | (g << 8) | b;
                }
            }
        } else {
            memset(pixels, 0, dev->buffer_size);
        }

        /* Submit buffer for display */
        if (!front) pixelforge_swap_buffers(dev);
        printf("Pattern written and buffer submitted\n");
        pixelforge_close_dev(dev);
        return 0;
    }

    /* GPU triangle rendering */
    if (render_triangle) {
        /* Allocate vertex buffer from VRAM */
        struct vram_block vb_block;
        if (vram_alloc(&dev->vram, VB_REGION_SIZE, PAGE_SIZE, &vb_block)) {
            fprintf(stderr, "Failed to allocate vertex buffer from VRAM\n");
            pixelforge_close_dev(dev);
            return 1;
        }
        uint8_t *vb_virt = vb_block.virt;

        /* Setup geometry */
        uint32_t idx_addr, idx_count, pos_addr, norm_addr, col_addr;
        uint16_t stride;
        setup_triangle_geometry(vb_block.phys, vb_virt,
                               &idx_addr, &idx_count,
                               &pos_addr, &norm_addr, &col_addr, &stride);

        printf("Rendering %d frame(s)...\n", frames);

        for (int frame = 0; frame < frames && keep_running; ++frame) {
            /* Request next buffer */
            uint8_t *buffer = pixelforge_get_back_buffer(dev);
            uint32_t buffer_phys = dev->buffer_phys[dev->render_buffer];

            /* Clear buffer */
            memset(buffer, 0, dev->buffer_size);
            DBG("Frame %d: buffer cleared", frame);

            /* Configure pipeline */
            configure_gpu_pipeline(dev, idx_addr, idx_count,
                                 pos_addr, norm_addr, col_addr, stride,
                                 buffer_phys);
            DBG("Frame %d: GPU pipeline configured", frame);
            DBG("Drawing to buffer at 0x%08x", buffer_phys);

            /* Start rendering */
            pf_csr_start(dev->csr_base);

            DBG("Frame %d: GPU started", frame);

            /* Wait for completion */
            if (!pixelforge_wait_for_gpu_ready(dev, GPU_STAGE_PER_PIXEL, &keep_running)) {
                fprintf(stderr, "Frame %d: GPU timeout\n", frame);
                break;
            }

            /* Submit buffer for display */
            pixelforge_swap_buffers(dev);

            if (capture_frames) {
                char filename[256];
                if (frame_capture_gen_filename(filename, sizeof(filename), "pixelforge_demo", frame, ".png") == 0) {
                    uint8_t *display_buffer = pixelforge_get_front_buffer(dev);
                    frame_capture_rgba(filename, display_buffer, dev->x_resolution,
                                     dev->y_resolution, dev->buffer_stride);
                }
            }

            printf("Frame %d rendered\n", frame);
        }

        pixelforge_close_dev(dev);
        return 0;
    }

    pixelforge_close_dev(dev);
    return 0;
}
