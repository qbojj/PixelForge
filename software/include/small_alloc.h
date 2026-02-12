#ifndef SMALL_ALLOC_H
#define SMALL_ALLOC_H

// not so optimized but simple malloc implementation for VRAM allocations
// only supports static memory region for backing storage

// all allocations are aligned to 16 bytes

#include <stddef.h>

typedef struct pool_t pool_t;

pool_t* small_init(void *memory, size_t size);
void small_destroy(pool_t *pool);
void *small_malloc(pool_t *pool, size_t size);
void *small_calloc(pool_t *pool, size_t n, size_t size);
void *small_realloc(pool_t *pool, void *ptr, size_t size);
void small_free(pool_t *pool, void* ptr);

#endif /* SMALL_ALLOC_H */
