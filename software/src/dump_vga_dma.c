/*
 * VGA Pixel Buffer DMA Register Dump Utility
 *
 * Reads and displays all registers from the Altera VGA Pixel Buffer DMA controller.
 * Based on altera_up_avalon_video_dma_controller register layout.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>

#include "vga_dma.h"

#define PAGE_SIZE       4096u
#define PAGE_MASK       (~(PAGE_SIZE - 1u))
#define PAGE_ALIGN_DOWN(a) ((a) & PAGE_MASK)
#define PAGE_OFFSET(a)  ((a) & ~PAGE_MASK)

static void* map_physical(int memfd, uint32_t phys, size_t length) {
    uint32_t aligned_phys = PAGE_ALIGN_DOWN(phys);
    uint32_t offset = PAGE_OFFSET(phys);
    size_t aligned_length = ((length + offset + PAGE_SIZE - 1) / PAGE_SIZE) * PAGE_SIZE;

    uint8_t *mapped = mmap(NULL, aligned_length, PROT_READ | PROT_WRITE,
                          MAP_SHARED, memfd, aligned_phys);
    if (mapped == MAP_FAILED) {
        perror("mmap");
        return NULL;
    }
    return mapped + offset;
}

static void dump_vga_dma_registers(volatile struct vga_dma_regs *regs, uint32_t base_phys) {
    printf("VGA Pixel Buffer DMA Controller\n");
    printf("================================\n");
    printf("Base address:      0x%08X\n\n", base_phys);

    printf("Buffer Addresses:\n");
    printf("  Front buffer:    0x%08X\n", regs->front_buffer);
    printf("  Back buffer:     0x%08X\n\n", regs->back_buffer);

    uint16_t x_res = regs->resolution.bits.x_resolution;
    uint16_t y_res = regs->resolution.bits.y_resolution;
    printf("Resolution:\n");
    printf("  Width:           %u pixels\n", x_res);
    printf("  Height:          %u pixels\n\n", y_res);

    printf("Status Register:   0x%08X\n", regs->status.raw);
    printf("  Swap busy:       %s\n", regs->status.bits.swap_busy ? "YES" : "NO");
    printf("  Addressing mode: %s\n", regs->status.bits.addr_mode ? "Consecutive" : "X-Y");
    printf("  Bytes per pixel: %u\n", regs->status.bits.color_type);
    printf("  Width coord:     %u bits\n", regs->status.bits.width_bits);
    printf("  Height coord:    %u bits\n\n", regs->status.bits.height_bits);
}

static void dump_raw_registers(volatile struct vga_dma_regs *regs) {
    volatile uint32_t *base = (volatile uint32_t *)regs;
    printf("\nRaw Register Dump:\n");
    printf("==================\n");
    for (size_t i = 0; i < 4; i++) {
        printf("  [0x%02zX] = 0x%08X", i * 4, base[i]);

        /* Annotate known registers */
        switch (i) {
            case 0: printf("  (Front buffer / Swap trigger)"); break;
            case 1: printf("  (Back buffer)"); break;
            case 2: printf("  (Resolution)"); break;
            case 3: printf("  (Status)"); break;
        }

        printf("\n");
    }
}

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "  --base ADDR     DMA base address (default: 0x%08X)\n", VGA_DMA_BASE_PHYS);
    fprintf(stderr, "  --raw           Also dump raw register values\n");
    fprintf(stderr, "  --help          Show this help\n");
}

int main(int argc, char **argv) {
    uint32_t dma_base = VGA_DMA_BASE_PHYS;
    int show_raw = 0;

    /* Parse arguments */
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--base") && i + 1 < argc) {
            dma_base = strtoul(argv[++i], NULL, 0);
        } else if (!strcmp(argv[i], "--raw")) {
            show_raw = 1;
        } else if (!strcmp(argv[i], "--help")) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    /* Open /dev/mem */
    int memfd = open("/dev/mem", O_RDWR | O_SYNC);
    if (memfd < 0) {
        perror("open /dev/mem");
        fprintf(stderr, "Note: This tool requires root privileges\n");
        return 1;
    }

    /* Map DMA controller registers */
    volatile struct vga_dma_regs *dma_regs = map_physical(memfd, dma_base, VGA_DMA_SPAN);
    if (!dma_regs) {
        close(memfd);
        return 1;
    }

    /* Dump register state */
    dump_vga_dma_registers(dma_regs, dma_base);

    if (show_raw) {
        dump_raw_registers(dma_regs);
    }

    close(memfd);
    return 0;
}
