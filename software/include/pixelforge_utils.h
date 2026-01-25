#ifndef PIXELFORGE_UTILS_H
#define PIXELFORGE_UTILS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "soc_system.h"
#include "vram_alloc.h"
#include "vga_dma.h"
#include "udma_alloc.h"

/* GPU CSR mapping parameters from soc_system.h */
#define PF_CSR_BASE_PHYS GPU_BASE
#define PF_CSR_MAP_SIZE GPU_SPAN

/* VRAM allocation parameters */
#define PF_VRAM_SIZE (64 << 20) /* 64 MB VRAM size */

#define FB_DATA_WIDTH 4u  /* Bytes per pixel (32-bit BGRA) */

typedef struct {
    int memfd;
    struct udma_buffer vram_dma;  /* DMA-backed VRAM allocation */
    volatile uint8_t *csr_base;
    volatile struct vga_dma_regs *vga_dma_regs;
    struct vram_allocator vram;
    uint8_t *vram_base_virt;
    uint32_t vram_base_phys;
    uint32_t vram_size;
    uint8_t *buffers[3];
    uint32_t buffer_phys[3];
    int old_display_buffer;         /* Buffer previously displayed (possibly still being scanned out) */
    int current_display_buffer;     /* Buffer currently being displayed */
    int render_buffer;              /* Buffer being rendered to */
    uint32_t x_resolution;
    uint32_t y_resolution;
    uint32_t data_width;
    size_t buffer_stride;                /* Line pitch in bytes */
    size_t buffer_size;                  /* Single buffer size in bytes */
} pixelforge_dev;

pixelforge_dev* pixelforge_open_dev(void);
void pixelforge_close_dev(pixelforge_dev *dev);

void pixelforge_swap_buffers(pixelforge_dev *dev);
void pixelforge_swap_buffers_novsync(pixelforge_dev *dev);

uint8_t* pixelforge_get_back_buffer(pixelforge_dev *dev);
uint8_t* pixelforge_get_front_buffer(pixelforge_dev *dev);

enum gpu_stage {
    GPU_STAGE_IA = 0,
    GPU_STAGE_VTX_TRANSFORM = 1,
    GPU_STAGE_PREP_RASTER = 2,
    GPU_STAGE_PER_PIXEL = 3,
};

bool pixelforge_wait_for_gpu_ready(pixelforge_dev *dev, enum gpu_stage stage, volatile bool *keep_running);

#endif /* PIXELFORGE_UTILS_H */
