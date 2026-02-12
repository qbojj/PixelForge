#include "small_alloc.h"
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

typedef struct block_header {
    uint64_t free: 1;
    uint64_t size: 63;
    struct block_header* next;
} block_header;

typedef struct pool_t {
    block_header* free_list;
} pool_t;

static size_t align(size_t size) {
    return (size + 15) & ~15; // align to 16 bytes
}

static void split_block(block_header *block, size_t size) {
    if (block->size < size + sizeof(*block) + sizeof(void*)) return;

    block_header *new_block = (block_header*)((uint8_t*)(block + 1) + size);
    new_block->size = block->size - size - sizeof(*block);
    new_block->next = block->next;
    new_block->free = true;

    block->size = size;
    block->next = new_block;
}

static void coalesce_block(pool_t *pool, struct block_header *block) {
    if (block->next && block->next->free) {
        block->size += block->next->size + sizeof(*block);
        block->next = block->next->next;
    }

    struct block_header *current = pool->free_list;

    while (current && current->next != block) {
        current = current->next;
    }

    if (current && current->free) {
        current->size += block->size + sizeof(*block);
        current->next = block->next;
    }
}

pool_t* small_init(void *memory, size_t size) {
    if (!memory || size < sizeof(pool_t) + sizeof(block_header)) return NULL;

    pool_t *pool = (pool_t*)memory;
    pool->free_list = (block_header*)(pool + 1);
    pool->free_list->size = size - sizeof(pool_t) - sizeof(block_header);
    pool->free_list->next = NULL;
    pool->free_list->free = true;

    return pool;
}

void small_destroy(pool_t *pool) {
    (void)pool; // No dynamic resources to free
}

void *small_malloc(pool_t *pool, size_t size) {
    if (!pool) return NULL;
    if (size == 0) return NULL;
    if (!pool->free_list) return NULL; // no memory available

    size = align(size);

    block_header *block = pool->free_list;
    for (; block; block = block->next)
        if (block->free && block->size >= size)
            break;

    if (!block) return NULL; // no suitable block found

    split_block(block, size);
    block->free = false;

    return block + 1;
}

void *small_calloc(pool_t *pool, size_t n, size_t size) {
    if (!pool) return NULL;
    if (n == 0 || size == 0) return NULL;
    if (size > SIZE_MAX / n) return NULL;

    void *ptr = small_malloc(pool, n * size);
    if (!ptr) return NULL;

    memset(ptr, 0, n * size);
    return ptr;
}

void *small_realloc(pool_t *pool, void *ptr, size_t size) {
    if (!pool) return NULL;
    if (!ptr) return small_malloc(pool, size);
    if (size == 0) {
        small_free(pool, ptr);
        return NULL;
    }

    block_header *block = (block_header*)ptr - 1;
    if (block->size >= size) {
        split_block(block, size);
        coalesce_block(pool, block->next);
        return ptr;
    }

    if (block->next && block->next->free && block->size + sizeof(*block) + block->next->size >= size) {
        coalesce_block(pool, block);
        split_block(block, size);
        return ptr;
    }

    void *new_ptr = small_malloc(pool, size);
    if (!new_ptr) return NULL;

    memcpy(new_ptr, ptr, block->size);
    small_free(pool, ptr);
    return new_ptr;
}

void small_free(pool_t *pool, void* ptr) {
    if (!pool) return;
    if (!ptr) return;

    block_header *block = (block_header*)ptr - 1;
    block->free = true;
    coalesce_block(pool, block);
}
