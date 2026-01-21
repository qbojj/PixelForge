#pragma once

#include <stddef.h>
#include <stdint.h>

#include "udma_alloc.h"

/**
 * Vertex/Index Buffer Management using DMA
 *
 * Provides a simplified interface for allocating vertex and index buffers
 * from the userspace DMA kernel module.
 */

struct vertex_buffer {
    struct udma_buffer buffer;
    size_t allocated_size;
    size_t used_size;
};

/**
 * Allocate a vertex buffer from DMA
 *
 * Args:
 *   size: Requested buffer size in bytes
 *   vbuf: Output vertex buffer structure
 *
 * Returns:
 *   0 on success, -1 on failure
 */
static inline int vertex_buffer_alloc_dma(size_t size, struct vertex_buffer *vbuf) {
    if (!vbuf || size == 0) return -1;

    if (udma_alloc(size, &vbuf->buffer) != 0) {
        return -1;
    }

    vbuf->allocated_size = size;
    vbuf->used_size = 0;
    return 0;
}

/**
 * Free a previously allocated vertex buffer
 *
 * Args:
 *   vbuf: Vertex buffer structure
 *
 * Returns:
 *   0 on success, -1 on failure
 */
static inline int vertex_buffer_free_dma(struct vertex_buffer *vbuf) {
    if (!vbuf) return -1;

    int ret = udma_free(&vbuf->buffer);
    vbuf->allocated_size = 0;
    vbuf->used_size = 0;
    return ret;
}

/**
 * Get virtual address of vertex buffer for CPU access
 */
static inline uint8_t* vertex_buffer_get_virt(const struct vertex_buffer *vbuf) {
    return vbuf ? vbuf->buffer.virt : NULL;
}

/**
 * Get physical address of vertex buffer for GPU access
 */
static inline uint32_t vertex_buffer_get_phys(const struct vertex_buffer *vbuf) {
    return vbuf ? vbuf->buffer.phys : 0;
}

/**
 * Get the size of the vertex buffer
 */
static inline size_t vertex_buffer_get_size(const struct vertex_buffer *vbuf) {
    return vbuf ? vbuf->allocated_size : 0;
}
