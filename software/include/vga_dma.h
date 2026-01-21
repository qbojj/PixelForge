#pragma once

#include <stdint.h>
#include "soc_system.h"

/* Default VGA DMA base address on H2F lightweight bridge */
#define VGA_DMA_BASE_PHYS   VIDEO_PIXEL_BUFFER_DMA_0_BASE
#define VGA_DMA_SPAN        VIDEO_PIXEL_BUFFER_DMA_0_SPAN

/* VGA DMA register offsets (for reference) */
#define VGA_DMA_FRONT_REG   0x00
#define VGA_DMA_BACK_REG    0x04
#define VGA_DMA_RES_REG     0x08
#define VGA_DMA_STATUS_REG  0x0C

/* VGA DMA register structure */
struct __attribute__((packed)) vga_dma_regs {
    uint32_t front_buffer;      /* 0x00: Front buffer address (read) / swap trigger (write) */
    uint32_t back_buffer;       /* 0x04: Back buffer address */
    union {
        uint32_t raw;        /* 0x08: Resolution */
        struct __attribute__((packed)) {
            uint32_t x_resolution : 16;
            uint32_t y_resolution : 16;
        } bits;
    } resolution;
    union {
        uint32_t raw;           /* 0x0C: Status and configuration */
        struct __attribute__((packed)) {
            uint32_t swap_busy       : 1;
            uint32_t addr_mode       : 1;
            uint32_t _res0           : 2;
            uint32_t color_type      : 4;
            uint32_t _res1           : 8;
            uint32_t width_bits      : 8;
            uint32_t height_bits     : 8;
        } bits;
    } status;
};
