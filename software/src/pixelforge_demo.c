

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
#include "vga_dma.h"

/*
 * PixelForge demo inspired by Altera's video DMA controller API.
 * Provides clean buffer management with device structure and helpers for:
 * - Opening/initializing video framebuffer device
 * - Setting back buffer address
 * - Swapping buffers
 * - Screen clear and fill operations
 */

#define PAGE_SIZE       4096u
#define PAGE_MASK       (~(PAGE_SIZE - 1u))
#define MAP_ALIGN(v)    (((v) + PAGE_SIZE - 1u) & PAGE_MASK)
#define PAGE_ALIGN_DOWN(a) ((a) & PAGE_MASK)
#define PAGE_OFFSET(a)  ((a) & ~PAGE_MASK)

/* Physical base addresses */
#define PF_CSR_BASE_PHYS     0xFF200000u
#define PF_CSR_MAP_SIZE      0x4000u
#define VRAM_BASE_PHYS       0x3C000000u  /* reserved VRAM carveout */
#define VRAM_SIZE            0x04000000u  /* 64MB */
#define VB_REGION_SIZE       0x00010000u

/* Framebuffer configuration */
#define FB_WIDTH             640u
#define FB_HEIGHT            480u
#define FB_DATA_WIDTH        4u    /* 4 bytes per pixel (RGBA8888) */

/*
 * PixelForge device structure (inspired by alt_up_video_dma_dev)
 */
typedef struct {
    int memfd;                         /* /dev/mem file descriptor */
    volatile uint8_t *csr_base;        /* GPU CSR mapped base */
    volatile struct vga_dma_regs *dma_regs; /* VGA pixel DMA registers */

    struct vram_allocator vram;        /* simple bump allocator over VRAM carveout */
    uint8_t *vram_base_virt;           /* mapped VRAM base */
    uint32_t vram_base_phys;           /* VRAM base phys */
    uint32_t vram_size;                /* VRAM size */

    uint32_t front_buffer_address;     /* Current display buffer phys addr */
    uint32_t back_buffer_address;      /* Current draw buffer phys addr */

    uint8_t *front_buffer_virt;        /* Front buffer CPU pointer */
    uint8_t *back_buffer_virt;         /* Back buffer CPU pointer */

    uint32_t x_resolution;             /* Width in pixels */
    uint32_t y_resolution;             /* Height in pixels */
    uint32_t data_width;               /* Bytes per pixel */

    size_t buffer_stride;              /* Bytes per scanline */
    size_t buffer_size;                /* Bytes per buffer */
} pixelforge_dev;

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

/*
 * Open and initialize the PixelForge device
 */
pixelforge_dev* pixelforge_open_dev(void) {
    pixelforge_dev *dev = calloc(1, sizeof(pixelforge_dev));
    if (!dev) {
        perror("calloc");
        return NULL;
    }

    dev->memfd = open("/dev/mem", O_RDWR | O_SYNC);
    if (dev->memfd < 0) {
        perror("open /dev/mem");
        free(dev);
        return NULL;
    }

    /* Map CSR window */
    dev->csr_base = map_physical(dev->memfd, PF_CSR_BASE_PHYS, PF_CSR_MAP_SIZE);
    if (!dev->csr_base) {
        close(dev->memfd);
        free(dev);
        return NULL;
    }

    /* Map VGA DMA controller */
    dev->dma_regs = map_physical(dev->memfd, VGA_DMA_BASE_PHYS, VGA_DMA_SPAN);
    if (!dev->dma_regs) {
        close(dev->memfd);
        free(dev);
        return NULL;
    }

    /* Configure device parameters */
    dev->x_resolution = FB_WIDTH;
    dev->y_resolution = FB_HEIGHT;
    dev->data_width = FB_DATA_WIDTH;
    dev->buffer_stride = FB_WIDTH * FB_DATA_WIDTH;
    dev->buffer_size = FB_WIDTH * FB_HEIGHT * FB_DATA_WIDTH;

    /* Map VRAM carveout and allocate front/back color buffers */
    dev->vram_base_virt = map_physical(dev->memfd, VRAM_BASE_PHYS, VRAM_SIZE);
    if (!dev->vram_base_virt) {
        close(dev->memfd);
        free(dev);
        return NULL;
    }
    dev->vram_base_phys = VRAM_BASE_PHYS;
    dev->vram_size = VRAM_SIZE;
    vram_allocator_init(&dev->vram, dev->vram_base_virt, dev->vram_base_phys, dev->vram_size);

    struct vram_block fb0, fb1;
    if (vram_alloc(&dev->vram, dev->buffer_size, PAGE_SIZE, &fb0) ||
        vram_alloc(&dev->vram, dev->buffer_size, PAGE_SIZE, &fb1)) {
        fprintf(stderr, "VRAM allocation failed for framebuffers\n");
        close(dev->memfd);
        free(dev);
        return NULL;
    }

    dev->front_buffer_virt = fb0.virt;
    dev->back_buffer_virt = fb1.virt;
    dev->front_buffer_address = fb0.phys;
    dev->back_buffer_address = fb1.phys;

    DBG("PixelForge device opened: %ux%u, front=0x%08x back=0x%08x",
        dev->x_resolution, dev->y_resolution,
        dev->front_buffer_address, dev->back_buffer_address);

    /* setup front and back buffers in the DMA controller */
    dev->dma_regs->back_buffer = dev->front_buffer_address;
    dev->dma_regs->front_buffer = 1; // trigger swap to load front buffer address
    while (dev->dma_regs->status.bits.swap_busy && keep_running) {
        usleep(50);  /* wait for previous swap to complete */
    }
    dev->dma_regs->back_buffer = dev->back_buffer_address;

    return dev;
}

/* Report GPU readiness status (components + vector) */
static void report_ready_status(volatile uint8_t *csr) {
    uint32_t ready = pf_csr_get_ready(csr);
    uint32_t comps = pf_csr_get_ready_components(csr);
    uint32_t vec = pf_csr_get_ready_vec(csr);

    printf("[GPU READY] ready=%u  ia=%s vt=%s rast=%s pix=%s  vec=0x%08x\n",
           (ready & 1u),
           (comps & 0x1) ? "ready" : "busy",
           (comps & 0x2) ? "ready" : "busy",
           (comps & 0x4) ? "ready" : "busy",
           (comps & 0x8) ? "ready" : "busy",
           vec);
}

/*
 * Close device and release resources
 */
void pixelforge_close_dev(pixelforge_dev *dev) {
    if (dev) {
        if (dev->memfd >= 0) {
            close(dev->memfd);
        }
        free(dev);
    }
}

/*
 * Set the back buffer address (similar to alt_up_video_dma_ctrl_set_bb_addr)
 */
int pixelforge_set_back_buffer(pixelforge_dev *dev, uint32_t new_address, void *new_virt) {
    if (!dev || !dev->dma_regs) return -1;

    dev->dma_regs->back_buffer = new_address;
    dev->back_buffer_address = new_address;
    dev->back_buffer_virt = new_virt;

    DBG("Set back buffer: 0x%08x", dev->back_buffer_address);
    return 0;
}

/*
 * Swap front and back buffers (similar to alt_up_video_dma_ctrl_swap_buffers)
 * The DMA will perform the swap at the next vertical refresh
 */
int pixelforge_swap_buffers(pixelforge_dev *dev) {
    if (!dev || !dev->dma_regs) return -1;

    /* Trigger swap by writing to front register */
    dev->dma_regs->front_buffer = 1;
    while (dev->dma_regs->status.bits.swap_busy && keep_running) {
        usleep(10);
    }

    /* Update tracked addresses */
    uint32_t temp = dev->back_buffer_address;
    dev->back_buffer_address = dev->front_buffer_address;
    dev->front_buffer_address = temp;

    /* Update virtual pointers */
    uint8_t *temp_virt = dev->back_buffer_virt;
    dev->back_buffer_virt = dev->front_buffer_virt;
    dev->front_buffer_virt = temp_virt;

    DBG("Swapped buffers: front=0x%08x back=0x%08x",
        dev->front_buffer_address, dev->back_buffer_address);

    return 0;
}

int pixelforge_swap_buffers_to(pixelforge_dev *dev, uint32_t address, void *virt) {
    int ret;
    ret = pixelforge_set_back_buffer(dev, address, virt);
    if (ret < 0) return ret;
    return pixelforge_swap_buffers(dev);
}

/*
 * Check if buffer swap has completed (similar to alt_up_video_dma_ctrl_check_swap_status)
 * Returns 0 if complete, 1 if still in progress
 */
int pixelforge_check_swap_status(pixelforge_dev *dev) {
    if (!dev || !dev->dma_regs) return -1;
    return dev->dma_regs->status.bits.swap_busy;
}

/*
 * Fill entire screen with a given color (similar to alt_up_video_dma_screen_fill)
 * backbuffer: 1 = draw to back buffer, 0 = draw to front buffer
 */
void pixelforge_screen_fill(pixelforge_dev *dev, uint32_t color, int backbuffer) {
    if (!dev) return;

    uint8_t *buf = backbuffer ? dev->back_buffer_virt : dev->front_buffer_virt;
    uint32_t *pixels = (uint32_t*)buf;
    uint32_t pixel_count = dev->x_resolution * dev->y_resolution;

    DBG("Fill screen with 0x%08x (%s buffer)", color, backbuffer ? "back" : "front");

    /* Fast 32-bit fill */
    for (uint32_t i = 0; i < pixel_count; ++i) {
        pixels[i] = color;
    }
}

/*
 * Clear entire screen (fill with 0) (similar to alt_up_video_dma_screen_clear)
 */
void pixelforge_screen_clear(pixelforge_dev *dev, int backbuffer) {
    pixelforge_screen_fill(dev, 0, backbuffer);
}

/*
 * Helper to wait for GPU ready (polling or UIO)
 */
static int wait_for_gpu_ready(pixelforge_dev *dev, const char *uio_path) {
    if (uio_path) {
        int uio = open(uio_path, O_RDWR);
        if (uio < 0) {
            perror("open uio");
            return -1;
        }
        uint32_t count = 1;
        if (read(uio, &count, sizeof(count)) != sizeof(count)) {
            perror("uio read");
            close(uio);
            return -1;
        }
        write(uio, &count, sizeof(count));
        close(uio);
        return 0;
    }

    /* Poll ready bit */
    for (int i = 0; i < 10000000 && keep_running; ++i) {
        uint32_t ready = pf_csr_get_ready(dev->csr_base);
        if (ready & 0x1) {
            DBG("GPU ready after %d polls", i);
            return 0;
        }
        usleep(50);
    }
    fprintf(stderr, "timeout waiting for GPU ready\n");
    return -1;
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
static void setup_triangle_geometry(pixelforge_dev *dev, uint32_t vb_phys, uint8_t *vb_virt,
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
                                   uint32_t color_addr, uint32_t depthstencil_addr) {
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
    pixelforge_vtx_xf_config_t xf = {0};
    xf.enabled.normal_enable = true;
    for (int i = 0; i < 16; ++i) {
        xf.position_mv[i] = (i % 5 == 0) ? fp16_16(1.0f) : fp16_16(0.0f);
        xf.position_p[i] = xf.position_mv[i];
    }
    for (int i = 0; i < 9; ++i) {
        xf.normal_mv_inv_t[i] = (i % 4 == 0) ? fp16_16(1.0f) : fp16_16(0.0f);
    }
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
    pf_csr_set_light0(csr, &light);
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
    fb.depthstencil_address = depthstencil_addr;
    fb.depthstencil_pitch = dev->x_resolution * 4; /* D16_X8_S8 combined */
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
    fprintf(stderr, "  --uio /dev/uioX       Use UIO for IRQ (otherwise poll)\n");
    fprintf(stderr, "  --verbose             Enable debug output\n");
    fprintf(stderr, "  --throttle            Throttle debug output with delays\n");
    fprintf(stderr, "  --front               Operate on front buffer instead of back buffer\n");
}

int main(int argc, char **argv) {
    const char *uio_path = NULL;
    bool clear_test = false;
    bool xor_test = false;
    bool render_triangle = false;
    bool front = false;
    int frames = 1;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--clear-test")) {
            clear_test = 1;
        } else if (!strcmp(argv[i], "--xor-test")) {
            xor_test = 1;
        } else if (!strcmp(argv[i], "--render-triangle")) {
            render_triangle = 1;
        } else if (!strcmp(argv[i], "--frames") && i + 1 < argc) {
            frames = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--uio") && i + 1 < argc) {
            uio_path = argv[++i];
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

        uint32_t *pixels = (uint32_t*)(front ? dev->front_buffer_virt : dev->back_buffer_virt);

        if (front) {
            printf("Operating on FRONT buffer\n");
            pixelforge_swap_buffers_to(dev, dev->front_buffer_address, dev->front_buffer_virt);
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

        if (!front) {
            pixelforge_swap_buffers_to(dev, dev->back_buffer_address, dev->back_buffer_virt);
        }
        printf("Pattern written and buffer swapped\n");
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
        setup_triangle_geometry(dev, vb_block.phys, vb_virt,
                               &idx_addr, &idx_count,
                               &pos_addr, &norm_addr, &col_addr, &stride);

        /* Allocate combined depth+stencil buffer (D16_X8_S8) */
        size_t ds_size = dev->x_resolution * dev->y_resolution * 4;
        struct vram_block ds_block;
        if (vram_alloc(&dev->vram, ds_size, PAGE_SIZE, &ds_block)) {
            fprintf(stderr, "Failed to allocate depth/stencil buffer from VRAM\n");
            pixelforge_close_dev(dev);
            return 1;
        }

        printf("Rendering %d frame(s)...\n", frames);

        for (int frame = 0; frame < frames && keep_running; ++frame) {
            /* Clear back buffer */
            pixelforge_screen_fill(dev, 0x00000000, 1);
            DBG("Frame %d: buffer cleared", frame);

            /* Configure pipeline for back buffer */
            configure_gpu_pipeline(dev, idx_addr, idx_count,
                                 pos_addr, norm_addr, col_addr, stride,
                                 dev->back_buffer_address,
                                 ds_block.phys);
            DBG("Frame %d: GPU pipeline configured", frame);
            DBG("Drawing to %x08x", dev->back_buffer_address);

            /* Start rendering */
            pf_csr_start(dev->csr_base);

            DBG("Frame %d: GPU started", frame);

            /* Wait for completion */
            if (wait_for_gpu_ready(dev, uio_path) != 0) {
                fprintf(stderr, "Frame %d: GPU timeout\n", frame);
                break;
            }

            /* Swap buffers */
            pixelforge_swap_buffers(dev);
            printf("Frame %d rendered\n", frame);
        }

        pixelforge_close_dev(dev);
        return 0;
    }

    pixelforge_close_dev(dev);
    return 0;
}
