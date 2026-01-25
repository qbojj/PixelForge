#define _GNU_SOURCE
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <dirent.h>
#include <inttypes.h>
#include <linux/udmabuf.h>

#include "pixelforge_utils.h"
#include "graphics_pipeline_csr_access.h"
#include "udma_alloc.h"

#ifndef PAGE_SIZE
#define PAGE_SIZE       4096u
#endif

#define PAGE_MASK       (~(PAGE_SIZE - 1u))
#define MAP_ALIGN(v)    (((v) + PAGE_SIZE - 1u) & PAGE_MASK)
#define PAGE_ALIGN_DOWN(a) ((a) & PAGE_MASK)
#define PAGE_OFFSET(a)  ((a) & ~PAGE_MASK)

static void* map_physical(int memfd, uint32_t phys, size_t length) {
    uint32_t aligned_phys = PAGE_ALIGN_DOWN(phys);
    uint32_t offset = PAGE_OFFSET(phys);
    size_t aligned_length = MAP_ALIGN(length + offset);
    uint8_t *mapped = mmap(NULL, aligned_length, PROT_READ | PROT_WRITE,
                          MAP_SHARED, memfd, aligned_phys);
    if (mapped == MAP_FAILED) {
        return NULL;
    }
    return mapped + offset;
}

static int init_udmabuf(pixelforge_dev *dev, size_t size) {
    if (udma_alloc(size, &dev->vram_dma) != 0) {
        fprintf(stderr, "Failed to allocate VRAM from udmabuf\n");
        return -1;
    }

    dev->vram_base_virt = dev->vram_dma.virt;
    dev->vram_base_phys = dev->vram_dma.phys;
    dev->vram_size = dev->vram_dma.size;

    printf("VRAM allocated: phys=0x%08x, virt=%p, size=%zu\n",
           dev->vram_base_phys, dev->vram_base_virt, dev->vram_size);

    return 0;
}

pixelforge_dev* pixelforge_open_dev(void) {
    pixelforge_dev *dev = calloc(1, sizeof(pixelforge_dev));
    if (!dev) return NULL;

    dev->memfd = -1;
    memset(&dev->vram_dma, 0, sizeof(dev->vram_dma));
    dev->vram_dma.dmafd = dev->vram_dma.memfd = dev->vram_dma.ctrl_fd = -1;

    dev->memfd = open("/dev/mem", O_RDWR | O_SYNC);
    if (dev->memfd < 0) {
        free(dev);
        return NULL;
    }

    dev->csr_base = map_physical(dev->memfd, PF_CSR_BASE_PHYS, PF_CSR_MAP_SIZE);
    dev->vga_dma_regs = (volatile struct vga_dma_regs *)map_physical(dev->memfd, VGA_DMA_BASE_PHYS, VGA_DMA_SPAN);
    if (!dev->csr_base || !dev->vga_dma_regs) {
        close(dev->memfd);
        free(dev);
        return NULL;
    }

    /* Read resolution from VGA DMA hardware */
    dev->x_resolution = dev->vga_dma_regs->resolution.bits.x_resolution;
    dev->y_resolution = dev->vga_dma_regs->resolution.bits.y_resolution;
    dev->data_width = FB_DATA_WIDTH;
    dev->buffer_stride = dev->x_resolution * dev->data_width;
    dev->buffer_size = (size_t)dev->buffer_stride * dev->y_resolution;

    printf("x resolution: %u, y resolution: %u, buffer size: %zu bytes\n",
          dev->x_resolution, dev->y_resolution, dev->buffer_size);

    if (init_udmabuf(dev, PF_VRAM_SIZE)) {
        goto error;
    }
    vram_allocator_init(&dev->vram, dev->vram_base_virt, dev->vram_base_phys, dev->vram_size);

    /* Allocate space for three buffers (triple buffering) */
    struct vram_block bufs[3];
    for (int i = 0; i < 3; i++) {
        if (vram_alloc(&dev->vram, dev->buffer_size, PAGE_SIZE, &bufs[i])) {
            goto error;
        }
        dev->buffers[i] = bufs[i].virt;
        dev->buffer_phys[i] = bufs[i].phys;
        memset(dev->buffers[i], 0, dev->buffer_size);
        printf("Buffer %d:          0x%08x\n", i, dev->buffer_phys[i]);
    }

    /* Initialize buffer indices:
     * - Buffer 0: old display (possibly still being scanned out)
     * - Buffer 1: current display
     * - Buffer 2: available for rendering */
    dev->old_display_buffer = 0;
    dev->current_display_buffer = 1;
    dev->render_buffer = 2;

    /* Initialize VGA DMA with buffer 0 */
    dev->vga_dma_regs->back_buffer = dev->buffer_phys[1];
    dev->vga_dma_regs->front_buffer = 1; /* Trigger initial swap */

    return dev;

error:
    pixelforge_close_dev(dev);
    return NULL;
}

void pixelforge_close_dev(pixelforge_dev *dev) {
    if (dev) {
        udma_free(&dev->vram_dma);
        if (dev->memfd >= 0) close(dev->memfd);
        free(dev);
    }
}

static void pixelforge_swap_buffers_impl(pixelforge_dev *dev, bool vsync) {
    if (!dev || !dev->vga_dma_regs) return;

    /* Wait for previous swap to complete BEFORE triggering new swap */
    while (vsync && dev->vga_dma_regs->status.bits.swap_busy) {
        usleep(10);
    }

    dev->vga_dma_regs->back_buffer = dev->buffer_phys[dev->render_buffer];
    dev->vga_dma_regs->front_buffer = 1;

    int old_old_display = dev->old_display_buffer;
    dev->old_display_buffer = dev->current_display_buffer;
    dev->current_display_buffer = dev->render_buffer;
    dev->render_buffer = old_old_display;
}

void pixelforge_swap_buffers(pixelforge_dev *dev) {
    pixelforge_swap_buffers_impl(dev, true);
}

void pixelforge_swap_buffers_novsync(pixelforge_dev *dev) {
    pixelforge_swap_buffers_impl(dev, false);
}

uint8_t* pixelforge_get_back_buffer(pixelforge_dev *dev) {
    if (!dev) return NULL;
    return dev->buffers[dev->render_buffer];
}

uint8_t* pixelforge_get_front_buffer(pixelforge_dev *dev) {
    if (!dev) return NULL;
    return dev->buffers[dev->current_display_buffer];
}

bool pixelforge_wait_for_gpu_ready(pixelforge_dev *dev, enum gpu_stage stage, volatile bool *keep_running) {
    // component is really ready if all prior stages are also ready
    // TODO: actually make use of interrupts (uio or a real kernel driver)

    uint32_t mask = (1u << (stage + 1)) - 1;

    while (true) {
        if (keep_running && !*keep_running)
            return false;

        uint32_t ready_components = pf_csr_get_ready_components(dev->csr_base);

        if ((ready_components & mask) == mask)
            return true;

        usleep(50);
    }
}
