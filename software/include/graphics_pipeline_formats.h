/*
 * PixelForge GPU Pipeline - Data Formats & Enums Header
 *
 * This header defines all enums, structs, and types used throughout the PixelForge GPU pipeline.
 * It mirrors the Amaranth Python definitions for hardware compatibility.
 */

#ifndef PIXELFORGE_FORMATS_H
#define PIXELFORGE_FORMATS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* ============================================================================
 * Basic Types & Constants (aligned with gpu/utils/types.py and gpu/utils/layouts.py)
 * ============================================================================ */

/* Fixed-point formats
 * - FixedPoint_mem:   SQ(16,16) -> 32-bit signed storage (used for matrices/attributes in memory)
 */
typedef int32_t pixelforge_fixed16_16_t;

typedef uint32_t pixelforge_addr_t;      /* address_shape: unsigned(32) */
typedef uint16_t pixelforge_stride_t;    /* stride_shape:  unsigned(16) */
typedef uint16_t pixelforge_index_t;     /* index_shape:   unsigned(16) */

#define PIXELFORGE_MAX_TEXTURE_DIM     4096  /* 12-bit texture coords */
#define PIXELFORGE_TEXTURE_COORD_WIDTH 12
#define PIXELFORGE_NUM_TEXTURES        0
#define PIXELFORGE_NUM_LIGHTS          1

/* ============================================================================
 * Enumerations
 * ============================================================================ */

/* Index buffer format */
typedef enum {
    PIXELFORGE_INDEX_NOT_INDEXED = 0,
    PIXELFORGE_INDEX_U8 = 1,
    PIXELFORGE_INDEX_U16 = 2,
    PIXELFORGE_INDEX_U32 = 3,
} pixelforge_index_kind_t;

/* Input primitive topology */
typedef enum {
    PIXELFORGE_TOPOLOGY_POINT_LIST = 0,
    PIXELFORGE_TOPOLOGY_LINE_LIST = 1,
    PIXELFORGE_TOPOLOGY_LINE_STRIP = 2,
    PIXELFORGE_TOPOLOGY_TRIANGLE_LIST = 3,
    PIXELFORGE_TOPOLOGY_TRIANGLE_STRIP = 4,
    PIXELFORGE_TOPOLOGY_TRIANGLE_FAN = 5,
    PIXELFORGE_TOPOLOGY_LINE_LIST_ADJACENCY = 6,
    PIXELFORGE_TOPOLOGY_LINE_STRIP_ADJACENCY = 7,
    PIXELFORGE_TOPOLOGY_TRIANGLE_LIST_ADJACENCY = 8,
    PIXELFORGE_TOPOLOGY_TRIANGLE_STRIP_ADJACENCY = 9,
    PIXELFORGE_TOPOLOGY_PATCH_LIST = 10,
} pixelforge_input_topology_t;

/* Output primitive type (after assembly) */
typedef enum {
    PIXELFORGE_PRIM_POINTS = 0,
    PIXELFORGE_PRIM_LINES = 1,
    PIXELFORGE_PRIM_TRIANGLES = 2,
} pixelforge_primitive_type_t;

/* Component format/scaling */
typedef enum {
    PIXELFORGE_SCALING_UNORM = 0,    /* Normalized unsigned: value / (2^n - 1) */
    PIXELFORGE_SCALING_SNORM = 1,    /* Normalized signed: value / (2^(n-1) - 1) */
    PIXELFORGE_SCALING_UINT = 2,     /* Unsigned integer */
    PIXELFORGE_SCALING_SINT = 3,     /* Signed integer */
    PIXELFORGE_SCALING_FIXED = 4,    /* Fixed-point (16.16) */
    PIXELFORGE_SCALING_FLOAT = 5,    /* IEEE 754 floating point */
} pixelforge_scaling_type_t;

/* Cull face mode */
typedef enum {
    PIXELFORGE_CULL_NONE = 0,
    PIXELFORGE_CULL_FRONT = 1,
    PIXELFORGE_CULL_BACK = 2,
    PIXELFORGE_CULL_FRONT_AND_BACK = 3,
} pixelforge_cull_face_t;

/* Front face winding */
typedef enum {
    PIXELFORGE_WINDING_CCW = 0,
    PIXELFORGE_WINDING_CW = 1,
} pixelforge_front_face_t;

/* Depth/stencil comparison operator */
typedef enum {
    PIXELFORGE_CMP_NEVER = 0,
    PIXELFORGE_CMP_LESS = 1,
    PIXELFORGE_CMP_EQUAL = 2,
    PIXELFORGE_CMP_LESS_OR_EQUAL = 3,
    PIXELFORGE_CMP_GREATER = 4,
    PIXELFORGE_CMP_NOT_EQUAL = 5,
    PIXELFORGE_CMP_GREATER_OR_EQUAL = 6,
    PIXELFORGE_CMP_ALWAYS = 7,
} pixelforge_compare_op_t;

/* Stencil operation */
typedef enum {
    PIXELFORGE_STENCIL_KEEP = 0,
    PIXELFORGE_STENCIL_ZERO = 1,
    PIXELFORGE_STENCIL_REPLACE = 2,
    PIXELFORGE_STENCIL_INCR = 3,
    PIXELFORGE_STENCIL_DECR = 4,
    PIXELFORGE_STENCIL_INVERT = 5,
    PIXELFORGE_STENCIL_INCR_WRAP = 6,
    PIXELFORGE_STENCIL_DECR_WRAP = 7,
} pixelforge_stencil_op_t;

/* Blending operation */
typedef enum {
    PIXELFORGE_BLEND_ADD = 0,
    PIXELFORGE_BLEND_SUBTRACT = 1,
    PIXELFORGE_BLEND_REVERSE_SUBTRACT = 2,
    PIXELFORGE_BLEND_MIN = 3,
    PIXELFORGE_BLEND_MAX = 4,
} pixelforge_blend_op_t;

/* Blending factor */
typedef enum {
    PIXELFORGE_BLEND_ZERO = 0,
    PIXELFORGE_BLEND_ONE = 1,
    PIXELFORGE_BLEND_SRC_COLOR = 2,
    PIXELFORGE_BLEND_ONE_MINUS_SRC_COLOR = 3,
    PIXELFORGE_BLEND_DST_COLOR = 4,
    PIXELFORGE_BLEND_ONE_MINUS_DST_COLOR = 5,
    PIXELFORGE_BLEND_SRC_ALPHA = 6,
    PIXELFORGE_BLEND_ONE_MINUS_SRC_ALPHA = 7,
    PIXELFORGE_BLEND_DST_ALPHA = 8,
    PIXELFORGE_BLEND_ONE_MINUS_DST_ALPHA = 9,
} pixelforge_blend_factor_t;

/* Input vertex attribute mode */
typedef enum {
    PIXELFORGE_ATTR_CONSTANT = 0,
    PIXELFORGE_ATTR_PER_VERTEX = 1,
} pixelforge_input_mode_t;

/* ============================================================================
 * Structures
 * ============================================================================ */

/* Index generator configuration */
typedef struct {
    uint32_t address;   /* Start address in index buffer (addr_shape: u32) */
    uint32_t count;     /* Number of indices (u32) */
    pixelforge_index_kind_t kind;
} pixelforge_idx_config_t;

/* Topology configuration */
typedef struct {
    pixelforge_input_topology_t input_topology;
    bool primitive_restart_enable;
    uint32_t primitive_restart_index;   /* u32 */
    uint32_t base_vertex;               /* u32 */
} pixelforge_topo_config_t;

/* Input attribute configuration (matches InputAssemblyAttrConfigLayout)
 * info is a union in RTL; use the larger constant_value payload (4x FixedPoint_mem)
 * to cover both modes. When mode == CONSTANT, fill constant_value[4].
 * When mode == PER_VERTEX, fill per_vertex.address/stride and ignore constant_value.
 */
typedef struct {
    pixelforge_input_mode_t mode;
    union {
        struct {
            pixelforge_fixed16_16_t value[4]; /* Vector4_mem: 4 x SQ(16,16) */
        } constant_value;
        struct {
            uint32_t address;    /* addr_shape: u32 */
            uint16_t stride;     /* stride_shape: u16 */
        } per_vertex;
    } info;
} pixelforge_input_attr_t;

/* Vertex transform enablement (num_textures currently 0) */
typedef struct {
    bool normal_enable;
} pixelforge_vtx_enable_t;

/* Vertex transform configuration */
typedef struct {
    pixelforge_vtx_enable_t enabled;
    pixelforge_fixed16_16_t position_mv[16];     /* Model-view matrix (4x4) */
    pixelforge_fixed16_16_t position_p[16];      /* Projection matrix (4x4) */
    pixelforge_fixed16_16_t normal_mv_inv_t[9];  /* Normal model-view inverse transpose (3x3) */
#if PIXELFORGE_NUM_TEXTURES > 0
    pixelforge_fixed16_16_t texture_transform[PIXELFORGE_NUM_TEXTURES][16];
#endif
} pixelforge_vtx_xf_config_t;

/* Material properties for lighting (Vector3 FixedPoint_mem in CSRs) */
typedef struct {
    pixelforge_fixed16_16_t ambient[3];
    pixelforge_fixed16_16_t diffuse[3];
    pixelforge_fixed16_16_t specular[3];
    pixelforge_fixed16_16_t shininess;
} pixelforge_material_t;

/* Light properties (one light supported: num_lights=1) */
typedef struct {
    pixelforge_fixed16_16_t position[4];
    pixelforge_fixed16_16_t ambient[3];
    pixelforge_fixed16_16_t diffuse[3];
    pixelforge_fixed16_16_t specular[3];
} pixelforge_light_t;

/* Stencil operation configuration (total 40 bits) */
typedef struct {
    pixelforge_compare_op_t compare_op;
    pixelforge_stencil_op_t pass_op;
    pixelforge_stencil_op_t fail_op;
    pixelforge_stencil_op_t depth_fail_op;
    uint8_t reference;
    uint8_t mask;
    uint8_t write_mask;
} pixelforge_stencil_op_config_t;

/* Depth test configuration (8 bits) */
typedef struct {
    bool test_enabled;
    bool write_enabled;
    pixelforge_compare_op_t compare_op;
} pixelforge_depth_test_config_t;

/* Blend configuration (32 bits) */
typedef struct {
    pixelforge_blend_factor_t src_factor;
    pixelforge_blend_factor_t dst_factor;
    pixelforge_blend_factor_t src_a_factor;
    pixelforge_blend_factor_t dst_a_factor;
    bool enabled;
    pixelforge_blend_op_t blend_op;
    pixelforge_blend_op_t blend_a_op;
    uint8_t color_write_mask;  /* RGBA mask */
} pixelforge_blend_config_t;

/* Primitive assembly configuration (5 bits used) */
typedef struct {
    pixelforge_primitive_type_t type;
    pixelforge_cull_face_t cull;
    pixelforge_front_face_t winding;
} pixelforge_prim_config_t;

/* Framebuffer configuration (FramebufferInfoLayout) */
typedef struct {
    uint16_t width;   /* texture_coord_shape: 12 bits used */
    uint16_t height;  /* texture_coord_shape: 12 bits used */

    /* positions in SQ(13, 4) */
    pixelforge_fixed16_16_t viewport_x;
    pixelforge_fixed16_16_t viewport_y;
    pixelforge_fixed16_16_t viewport_width;
    pixelforge_fixed16_16_t viewport_height;
    /* depth in UQ(1,15) normalized range [0,1] */
    pixelforge_fixed16_16_t viewport_min_depth;
    pixelforge_fixed16_16_t viewport_max_depth;

    int32_t  scissor_offset_x;
    int32_t  scissor_offset_y;
    uint32_t scissor_width;
    uint32_t scissor_height;

    uint32_t color_address;
    uint16_t color_pitch;
    uint32_t depth_address;
    uint16_t depth_pitch;
    uint32_t stencil_address;
    uint16_t stencil_pitch;
} pixelforge_framebuffer_config_t;

#endif /* PIXELFORGE_FORMATS_H */
