#pragma once

#include <stddef.h>
#include <stdint.h>

struct udma_buffer {
    uint8_t *virt;
    uint32_t phys;
    size_t size;
    int dmafd;      /* fd returned by udmabuf ioctl */
    int memfd;      /* backing memfd */
    int ctrl_fd;    /* /dev/udmabuf control fd */
};

int udma_alloc(size_t size, struct udma_buffer *buf);
int udma_free(struct udma_buffer *buf);
uint32_t udma_get_phys(const struct udma_buffer *buf);
uint8_t* udma_get_virt(const struct udma_buffer *buf);
size_t udma_get_size(const struct udma_buffer *buf);
