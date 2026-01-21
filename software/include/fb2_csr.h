#ifndef FB2_CSR_H
#define FB2_CSR_H

#include <stdint.h>
#include "soc_system.h"

/*
 * Altera VIP Frame Reader / Frame Buffer II Register Definitions
 * Based on altvipfb.c Linux driver for Altera VIP Frame Reader
 */

/* Frame Buffer II base address from soc_system.h */
#define FB2_BASE_PHYS    0xff210000 // FB_BASE
#define FB2_SPAN         64 //FB_SPAN

/* Register offsets - from Altera VIP Frame Reader driver */
#define FB2_CONTROL_REG               0x00   /* Control register (0=stop, 1=start) */
#define FB2_FRAME_SELECT_REG          0x0C   /* Frame select */
#define FB2_FRAME0_BASE_ADDRESS_REG   0x10   /* Frame 0 base address */
#define FB2_FRAME0_NUM_WORDS_REG      0x14   /* Number of memory words */
#define FB2_FRAME0_SAMPLES_REG        0x18   /* Total samples (width * height) */
#define FB2_FRAME0_WIDTH_REG          0x20   /* Frame width in pixels */
#define FB2_FRAME0_HEIGHT_REG         0x24   /* Frame height in pixels */
#define FB2_FRAME0_INTERLACED_REG     0x28   /* Interlaced mode (3=progressive) */

/* Control register values */
#define FB2_CONTROL_STOP          0
#define FB2_CONTROL_START         1

/* Frame Buffer II register structure - based on Altera VIP Frame Reader */
struct __attribute__((packed)) fb2_regs {
    uint32_t control;              /* 0x00: Control (0=stop, 1=start) */
    uint32_t status;               /* 0x04: Status */
    uint32_t _pad0[2];             /* 0x08-0x0B: Reserved */
    uint32_t frame_select;         /* 0x0C: Frame select */
    uint32_t frame0_base_address;  /* 0x10: Frame buffer base address */
    uint32_t frame0_num_words;     /* 0x14: Number of memory words */
    uint32_t frame0_samples;       /* 0x18: Total samples */
    uint32_t _pad1;                /* 0x1C: Reserved */
    uint32_t frame0_width;         /* 0x20: Frame width */
    uint32_t frame0_height;        /* 0x24: Frame height */
    uint32_t frame0_interlaced;    /* 0x28: Interlaced mode */
};

/* Helper functions */
static inline void fb2_write_reg(volatile struct fb2_regs *regs, uint32_t offset, uint32_t value) {
    volatile uint32_t *reg = (volatile uint32_t *)((volatile uint8_t *)regs + offset);
    *reg = value;
}

static inline uint32_t fb2_read_reg(volatile struct fb2_regs *regs, uint32_t offset) {
    volatile uint32_t *reg = (volatile uint32_t *)((volatile uint8_t *)regs + offset);
    return *reg;
}

/* Frame Buffer II control functions */
static inline void fb2_start(volatile struct fb2_regs *regs) {
    regs->control = FB2_CONTROL_START;
}

static inline void fb2_stop(volatile struct fb2_regs *regs) {
    regs->control = FB2_CONTROL_STOP;
}

static inline int fb2_is_running(volatile struct fb2_regs *regs) {
    return regs->control == FB2_CONTROL_START;
}

/* Configure frame buffer - based on Altera VIP Frame Reader driver altvipfb_start_hw() */
static inline void fb2_configure_frame(volatile struct fb2_regs *regs,
                                       uint32_t base_addr,
                                       uint32_t width,
                                       uint32_t height,
                                       uint32_t mem_word_width) {
    /* Calculate number of memory words and samples */
    uint32_t num_words = (width * height) / (mem_word_width / 32);
    uint32_t samples = width * height;

    /* Write configuration registers in same order as altvipfb_start_hw() */
    regs->frame0_base_address = base_addr;
    regs->frame0_num_words = num_words;
    regs->frame0_samples = samples;
    regs->frame0_width = width;
    regs->frame0_height = height;
    regs->frame0_interlaced = 3;  /* Progressive mode */
    regs->frame_select = 0;       /* Select frame 0 */
}

#endif /* FB2_CSR_H */
