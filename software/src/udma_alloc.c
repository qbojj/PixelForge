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
#include <stdbool.h>

#include "udma_alloc.h"

static void close_fd(int *fd) {
    if (*fd >= 0) {
        close(*fd);
        *fd = -1;
    }
}

static size_t page_align_up(size_t size) {
    size_t page = (size_t)getpagesize();
    return (size + page - 1u) & ~(page - 1u);
}

static bool used = false;

int udma_alloc(size_t size, struct udma_buffer *buf) {
    if (!buf || size == 0) return -1;
    size = page_align_up(size);
    memset(buf, 0, sizeof(*buf));
    buf->dmafd = buf->memfd = buf->ctrl_fd = -1;

    buf->phys = 0x3C000000;

    int devmem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (devmem_fd < 0) {
        perror("open /dev/mem");
        goto error;
    }

    buf->virt = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, devmem_fd, buf->phys);
    close(devmem_fd);

    if (buf->virt == MAP_FAILED) {
        buf->virt = NULL;
        perror("mmap /dev/mem");
        goto error;
    }

    if (used) {
        fprintf(stderr, "udma_alloc: warning: for now only a single allocation is supported "
                        "-> returing aliasing allocation\n");
    }
    used = true;

    buf->size = size;
    return 0;

error:
    udma_free(buf);
    return -1;
}

int udma_free(struct udma_buffer *buf) {
    if (!buf) return -1;
    if (buf->virt && buf->size) {
        munmap(buf->virt, buf->size);
        buf->virt = NULL;
    }
    close_fd(&buf->dmafd);
    close_fd(&buf->memfd);
    close_fd(&buf->ctrl_fd);
    buf->phys = 0;
    buf->size = 0;
    return 0;
}

uint32_t udma_get_phys(const struct udma_buffer *buf) {
    return buf ? buf->phys : 0;
}

uint8_t* udma_get_virt(const struct udma_buffer *buf) {
    return buf ? buf->virt : NULL;
}

size_t udma_get_size(const struct udma_buffer *buf) {
    return buf ? buf->size : 0;
}
