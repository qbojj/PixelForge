/*
 * Frame Buffer II Register Dump Utility
 *
 * Reads and displays all registers from the Intel/Altera Frame Buffer II controller.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>

#include "vga_dma.h"
#include "fb2_csr.h"

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

static void dump_fb2_registers(volatile struct fb2_regs *regs, uint32_t base_phys) {
    printf("Frame Buffer II Controller\n");
    printf("==========================\n");
}

/* Dump entire VGA DMA CSR address range word-by-word */
static void dump_full_range(volatile void *regs_void, uint32_t span_bytes) {
    volatile uint32_t *base = (volatile uint32_t *)regs_void;
    size_t words = span_bytes / sizeof(uint32_t);

    printf("\nFull VGA DMA CSR Range Dump (%u bytes):\n", span_bytes);
    printf("======================================\n");

    for (size_t i = 0; i < words; ++i) {
        uint32_t offset = (uint32_t)(i * 4);
        printf("  [0x%03X] = 0x%08X\n", offset, base[i]);
    }
}

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "  --base ADDR     Frame Buffer II base address (default: 0x%08X)\n", FB2_BASE_PHYS);
    fprintf(stderr, "  --all           Dump entire Frame Buffer II CSR address range\n");
    fprintf(stderr, "  --help          Show this help\n");
}

int main(int argc, char **argv) {
    uint32_t fb_base = FB2_BASE_PHYS;
    int show_all = 0;

    /* Parse arguments */
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--base") && i + 1 < argc) {
            fb_base = strtoul(argv[++i], NULL, 0);
        } else if (!strcmp(argv[i], "--all")) {
            show_all = 1;
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
        fprintf(stderr, "Note: This utility requires root privileges\n");
        return 1;
    }

    /* Map VGA DMA registers */
    volatile struct fb2_regs *regs = map_physical(memfd, fb_base, FB2_SPAN);
    if (!regs) {
        fprintf(stderr, "Failed to map VGA DMA registers at 0x%08X\n", fb_base);
        close(memfd);
        return 1;
    }

    if (show_all) {
        /* Dump entire mapped span */
        dump_full_range((volatile void *)regs, FB2_SPAN);
    } else {
        /* Dump decoded registers */
        dump_fb2_registers(regs, fb_base);
    }

    close(memfd);
    return 0;
}
