#pragma once

#include <stddef.h>
#include <stdint.h>

struct vram_block {
    uint8_t *virt;
    uint32_t phys;
    size_t size;
};

struct vram_allocator {
    uint8_t *virt_base;
    uint32_t phys_base;
    uint32_t size;
    uint32_t offset;
};

static inline void vram_allocator_init(struct vram_allocator *a, uint8_t *virt_base, uint32_t phys_base, uint32_t size) {
    a->virt_base = virt_base;
    a->phys_base = phys_base;
    a->size = size;
    a->offset = 0;
}

/* Align up to align (must be power of two). */
static inline uint32_t vram_align(uint32_t val, uint32_t align) {
    return (val + align - 1u) & ~(align - 1u);
}

/* Simple bump allocator: returns 0 on success, -1 on OOM. */
static inline int vram_alloc(struct vram_allocator *a, size_t size, size_t align, struct vram_block *out) {
    if (align == 0) align = 4;
    uint32_t aligned = vram_align(a->offset, (uint32_t)align);
    uint64_t end = (uint64_t)aligned + (uint64_t)size;
    if (end > a->size) return -1;
    out->phys = a->phys_base + aligned;
    out->virt = a->virt_base + aligned;
    out->size = size;
    a->offset = (uint32_t)end;
    return 0;
}
