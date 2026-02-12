/*
 * OpenGL ES 1.1 Common-Lite Wrapper for PixelForge GPU
 * Implementation with state tracking and dirty flags
 */

#define _GNU_SOURCE
#include "gles11_wrapper.h"
#include "pixelforge_utils.h"
#include "graphics_pipeline_csr_access.h"
#include "demo_utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

/* ============================================================================
 * Matrix Stack Configuration
 * ============================================================================ */

#define MAX_MODELVIEW_STACK_DEPTH  32
#define MAX_PROJECTION_STACK_DEPTH 2
#define MAX_TEXTURE_STACK_DEPTH    2

#define NUM_TEXTURES 0
#define MAX_LIGHTS 1

/* ============================================================================
 * Dirty Flags - Track what needs to be uploaded to GPU
 * ============================================================================ */

typedef enum {
    DIRTY_MATRICES         = (1 << 0),
    DIRTY_VIEWPORT         = (1 << 1),
    DIRTY_MATERIAL         = (1 << 2),
    DIRTY_LIGHTS           = (1 << 3),
    DIRTY_DEPTH            = (1 << 4),
    DIRTY_BLEND            = (1 << 5),
    DIRTY_STENCIL          = (1 << 6),
    DIRTY_CULL             = (1 << 7),
    DIRTY_VERTEX_ARRAYS    = (1 << 8),
    DIRTY_FRAMEBUFFER      = (1 << 9),
} dirty_flags_t;

/* ============================================================================
 * Matrix Stack
 * ============================================================================ */

typedef struct {
    float matrices[MAX_MODELVIEW_STACK_DEPTH][32];
    int depth;
} matrix_stack_t;

typedef struct {
    float ambient[4];
    float diffuse[4];
    float specular[4];
    float shininess;
} material_t;

typedef struct {
    bool enabled;
    float position[4];
    float ambient[4];
    float diffuse[4];
    float specular[4];
} light_t;

typedef struct {
    GLenum func;
    GLint ref;
    GLuint mask;
    GLuint writemask;
    GLenum fail_op;
    GLenum zfail_op;
    GLenum zpass_op;
} stencil_config_t;

typedef struct {
    bool enabled;
    GLuint buffer;
    size_t offset;
    GLint size;
    GLenum type;
    GLsizei stride;
} attribute_config_t;

typedef struct {
    GLuint id;
    void *virt;
    uint32_t phys;
    size_t size;
    bool alive;
} gl_buffer_t;

/* ============================================================================
 * Global State Structure
 * ============================================================================ */

typedef struct {
    /* PixelForge device */
    pixelforge_dev *dev;

    /* Dirty flags */
    uint32_t dirty;

    /* Matrix state */
    GLenum matrix_mode;
    matrix_stack_t modelview_stack;
    matrix_stack_t projection_stack;
    matrix_stack_t texture_stack;
    float *current_matrix;

    /* Lighting state */
    bool lighting_enabled;
    material_t material;
    light_t lights[MAX_LIGHTS];

    /* Depth state */
    bool depth_test_enabled;
    bool depth_write_enabled;
    GLenum depth_func;

    /* Blend state */
    bool blend_enabled;
    GLenum blend_src_factor;
    GLenum blend_dst_factor;

    /* Stencil state */
    bool stencil_test_enabled;
    stencil_config_t stencil_front;
    stencil_config_t stencil_back;

    /* Cull state */
    bool cull_face_enabled;
    GLenum cull_face_mode;
    GLenum front_face;

    /* Viewport state */
    float viewport_x;
    float viewport_y;
    float viewport_width;
    float viewport_height;

    GLint scissor_x;
    GLint scissor_y;
    GLsizei scissor_width;
    GLsizei scissor_height;

    /* Clear color */
    float clear_color[4];
    float clear_depth;
    GLint clear_stencil;

    /* Vertex array state */
    attribute_config_t vertex_array;
    attribute_config_t normal_array;
    attribute_config_t color_array;
    attribute_config_t texcoord_arrays[NUM_TEXTURES];

    /* Buffer objects */
    gl_buffer_t *buffers;
    size_t buffer_count;
    size_t buffer_capacity;
    GLuint next_buffer_id;
    GLuint array_buffer_binding;
    GLuint element_array_buffer_binding;
} gles_context_t;

static gles_context_t *g_ctx = NULL;

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

static int32_t fp16_16(float v) {
    return (int32_t)(v * 65536.0f);
}

static void set_vec4(float *dest, float r, float g, float b, float a) {
    dest[0] = r;
    dest[1] = g;
    dest[2] = b;
    dest[3] = a;
}

static void set_fp_vec_v(int32_t *dest, float *src, int count) {
    for (int i = 0; i < count; i++) {
        dest[i] = fp16_16(src[i]);
    }
}

static void mat4_copy(float *dst, const float *src) {
    memcpy(dst, src, 16 * sizeof(float));
}

static gl_buffer_t* get_buffer_by_id(gles_context_t *ctx, GLuint id) {
    if (!ctx || id == 0) return NULL;
    for (size_t i = 0; i < ctx->buffer_count; i++) {
        gl_buffer_t *buf = &ctx->buffers[i];
        if (buf->alive && buf->id == id) {
            return buf;
        }
    }
    return NULL;
}

static gl_buffer_t* create_buffer(gles_context_t *ctx, GLuint id) {
    if (!ctx) return NULL;
    if (ctx->buffer_count == ctx->buffer_capacity) {
        size_t new_capacity = ctx->buffer_capacity == 0 ? 16 : ctx->buffer_capacity * 2;
        gl_buffer_t *new_buffers = realloc(ctx->buffers, new_capacity * sizeof(gl_buffer_t));
        if (!new_buffers) return NULL;
        ctx->buffers = new_buffers;
        ctx->buffer_capacity = new_capacity;
    }

    gl_buffer_t *buf = &ctx->buffers[ctx->buffer_count++];
    memset(buf, 0, sizeof(*buf));
    buf->id = id;
    buf->alive = true;
    return buf;
}

static void init_matrix_stack(matrix_stack_t *stack) {
    stack->depth = 0;
    mat4_identity(stack->matrices[0]);
}

static float* get_current_stack_matrix(gles_context_t *ctx) {
    switch (ctx->matrix_mode) {
        case GL_MODELVIEW:
            return ctx->modelview_stack.matrices[ctx->modelview_stack.depth];
        case GL_PROJECTION:
            return ctx->projection_stack.matrices[ctx->projection_stack.depth];
        case GL_TEXTURE:
            return ctx->texture_stack.matrices[ctx->texture_stack.depth];
        default:
            return ctx->modelview_stack.matrices[ctx->modelview_stack.depth];
    }
}

static GLenum gl_compare_to_pf_compare(GLenum func) {
    switch (func) {
        case GL_NEVER:    return PIXELFORGE_CMP_NEVER;
        case GL_LESS:     return PIXELFORGE_CMP_LESS;
        case GL_EQUAL:    return PIXELFORGE_CMP_EQUAL;
        case GL_LEQUAL:   return PIXELFORGE_CMP_LESS_OR_EQUAL;
        case GL_GREATER:  return PIXELFORGE_CMP_GREATER;
        case GL_NOTEQUAL: return PIXELFORGE_CMP_NOT_EQUAL;
        case GL_GEQUAL:   return PIXELFORGE_CMP_GREATER_OR_EQUAL;
        case GL_ALWAYS:   return PIXELFORGE_CMP_ALWAYS;
        default:          return PIXELFORGE_CMP_ALWAYS;
    }
}

static GLenum gl_blend_to_pf_blend(GLenum factor) {
    switch (factor) {
        case GL_ZERO:                   return PIXELFORGE_BLEND_ZERO;
        case GL_ONE:                    return PIXELFORGE_BLEND_ONE;
        case GL_SRC_COLOR:              return PIXELFORGE_BLEND_SRC_COLOR;
        case GL_ONE_MINUS_SRC_COLOR:    return PIXELFORGE_BLEND_ONE_MINUS_SRC_COLOR;
        case GL_DST_COLOR:              return PIXELFORGE_BLEND_DST_COLOR;
        case GL_ONE_MINUS_DST_COLOR:    return PIXELFORGE_BLEND_ONE_MINUS_DST_COLOR;
        case GL_SRC_ALPHA:              return PIXELFORGE_BLEND_SRC_ALPHA;
        case GL_ONE_MINUS_SRC_ALPHA:    return PIXELFORGE_BLEND_ONE_MINUS_SRC_ALPHA;
        case GL_DST_ALPHA:              return PIXELFORGE_BLEND_DST_ALPHA;
        case GL_ONE_MINUS_DST_ALPHA:    return PIXELFORGE_BLEND_ONE_MINUS_DST_ALPHA;
        default:                        return PIXELFORGE_BLEND_ONE;
    }
}

static GLenum gl_stencil_op_to_pf(GLenum op) {
    switch (op) {
        case GL_KEEP:      return PIXELFORGE_STENCIL_KEEP;
        case GL_REPLACE:   return PIXELFORGE_STENCIL_REPLACE;
        case GL_INCR:      return PIXELFORGE_STENCIL_INCR;
        case GL_DECR:      return PIXELFORGE_STENCIL_DECR;
        case GL_INVERT:    return PIXELFORGE_STENCIL_INVERT;
        case GL_INCR_WRAP: return PIXELFORGE_STENCIL_INCR_WRAP;
        case GL_DECR_WRAP: return PIXELFORGE_STENCIL_DECR_WRAP;
        default:           return PIXELFORGE_STENCIL_KEEP;
    }
}

static pixelforge_input_topology_t gl_mode_to_topology(GLenum mode) {
    switch (mode) {
        case GL_POINTS:         return PIXELFORGE_TOPOLOGY_POINT_LIST;
        case GL_LINES:          return PIXELFORGE_TOPOLOGY_LINE_LIST;
        case GL_LINE_STRIP:     return PIXELFORGE_TOPOLOGY_LINE_STRIP;
        case GL_TRIANGLES:      return PIXELFORGE_TOPOLOGY_TRIANGLE_LIST;
        case GL_TRIANGLE_STRIP: return PIXELFORGE_TOPOLOGY_TRIANGLE_STRIP;
        case GL_TRIANGLE_FAN:   return PIXELFORGE_TOPOLOGY_TRIANGLE_FAN;
        default:                return PIXELFORGE_TOPOLOGY_TRIANGLE_LIST;
    }
}

/* ============================================================================
 * State Upload Functions - Only upload what's dirty
 * ============================================================================ */

static void upload_matrices(gles_context_t *ctx) {
    if (!(ctx->dirty & DIRTY_MATRICES)) return;

    // wait for vertex transform stage to be idle before uploading matrices
    pixelforge_wait_for_gpu_ready(ctx->dev, GPU_STAGE_VTX_TRANSFORM, NULL);

    volatile uint8_t *csr = ctx->dev->csr_base;

    float *mv = ctx->modelview_stack.matrices[ctx->modelview_stack.depth];
    float *p = ctx->projection_stack.matrices[ctx->projection_stack.depth];

    pixelforge_vtx_xf_config_t xf = {0};
    xf.enabled.normal_enable = 1;

    mat4_to_fp16_16(xf.position_mv, mv);
    mat4_to_fp16_16(xf.position_p, p);

    float nm[9];
    mat3_from_mat4(nm, mv);
    mat3_to_fp16_16(xf.normal_mv_inv_t, nm);

    // TODO: Texture matrices

    pf_csr_set_vtx_xf(csr, &xf);
    ctx->dirty &= ~DIRTY_MATRICES;
}

static void upload_material(gles_context_t *ctx) {
    if (!(ctx->dirty & DIRTY_MATERIAL)) return;

    // wait for vertex transform stage to be idle before uploading material
    pixelforge_wait_for_gpu_ready(ctx->dev, GPU_STAGE_VTX_TRANSFORM, NULL);

    volatile uint8_t *csr = ctx->dev->csr_base;

    pixelforge_material_t mat = {0};
    set_fp_vec_v(mat.ambient, ctx->material.ambient, 3);
    set_fp_vec_v(mat.diffuse, ctx->material.diffuse, 3);
    set_fp_vec_v(mat.specular, ctx->material.specular, 3);
    set_fp_vec_v(&mat.shininess, &ctx->material.shininess, 1);

    pf_csr_set_material(csr, &mat);
    ctx->dirty &= ~DIRTY_MATERIAL;
}

static void upload_lights(gles_context_t *ctx) {
    if (!(ctx->dirty & DIRTY_LIGHTS)) return;

    // wait for vertex transform stage to be idle before uploading matrices
    pixelforge_wait_for_gpu_ready(ctx->dev, GPU_STAGE_VTX_TRANSFORM, NULL);

    volatile uint8_t *csr = ctx->dev->csr_base;

    if (ctx->lighting_enabled) {
        for (int i = 0; i < MAX_LIGHTS; ++i) {
            pixelforge_light_t light = {0};

            if (ctx->lights[i].enabled) {
                set_fp_vec_v(light.position, ctx->lights[i].position, 4);
                set_fp_vec_v(light.ambient, ctx->lights[i].ambient, 3);
                set_fp_vec_v(light.diffuse, ctx->lights[i].diffuse, 3);
                set_fp_vec_v(light.specular, ctx->lights[i].specular, 3);
            }

            pf_csr_set_light(csr, i, &light);
        }
    } else {
        // add a single ambient light if lighting is disabled to ensure we get some color output
        pixelforge_light_t light = {0};
        set_fp_vec_v(light.ambient, (float[]){1.0f, 1.0f, 1.0f}, 3);
        pf_csr_set_light(csr, 0, &light);
        for (int i = 1; i < MAX_LIGHTS; ++i) {
            pixelforge_light_t off_light = {0};
            pf_csr_set_light(csr, i, &off_light);
        }
    }

    ctx->dirty &= ~DIRTY_LIGHTS;
}

static void upload_depth(gles_context_t *ctx) {
    if (!(ctx->dirty & DIRTY_DEPTH)) return;

    // wait for depth test stage to be idle before uploading depth state
    pixelforge_wait_for_gpu_ready(ctx->dev, GPU_STAGE_PER_PIXEL, NULL);

    volatile uint8_t *csr = ctx->dev->csr_base;

    pixelforge_depth_test_config_t depth = {
        .test_enabled = ctx->depth_test_enabled,
        .write_enabled = ctx->depth_write_enabled,
        .compare_op = gl_compare_to_pf_compare(ctx->depth_func),
    };

    pf_csr_set_depth(csr, &depth);
    ctx->dirty &= ~DIRTY_DEPTH;
}

static void upload_blend(gles_context_t *ctx) {
    if (!(ctx->dirty & DIRTY_BLEND)) return;

    // wait for blend stage to be idle before uploading blend state
    pixelforge_wait_for_gpu_ready(ctx->dev, GPU_STAGE_PER_PIXEL, NULL);

    volatile uint8_t *csr = ctx->dev->csr_base;

    pixelforge_blend_config_t blend = {
        .enabled = ctx->blend_enabled,
        .src_factor = gl_blend_to_pf_blend(ctx->blend_src_factor),
        .dst_factor = gl_blend_to_pf_blend(ctx->blend_dst_factor),
        .src_a_factor = gl_blend_to_pf_blend(ctx->blend_src_factor),
        .dst_a_factor = gl_blend_to_pf_blend(ctx->blend_dst_factor),
        .blend_op = PIXELFORGE_BLEND_ADD,
        .blend_a_op = PIXELFORGE_BLEND_ADD,
        .color_write_mask = 0xF,
    };

    pf_csr_set_blend(csr, &blend);
    ctx->dirty &= ~DIRTY_BLEND;
}

static void upload_stencil(gles_context_t *ctx) {
    if (!(ctx->dirty & DIRTY_STENCIL)) return;

    // wait for stencil stage to be idle before uploading stencil state
    pixelforge_wait_for_gpu_ready(ctx->dev, GPU_STAGE_PER_PIXEL, NULL);

    volatile uint8_t *csr = ctx->dev->csr_base;

    pixelforge_stencil_op_config_t stencil_front = {
        .compare_op = gl_compare_to_pf_compare(ctx->stencil_front.func),
        .reference = ctx->stencil_front.ref,
        .mask = ctx->stencil_front.mask,
        .write_mask = ctx->stencil_front.writemask,
        .fail_op = gl_stencil_op_to_pf(ctx->stencil_front.fail_op),
        .depth_fail_op = gl_stencil_op_to_pf(ctx->stencil_front.zfail_op),
        .pass_op = gl_stencil_op_to_pf(ctx->stencil_front.zpass_op),
    };
    pixelforge_stencil_op_config_t stencil_back = {
        .compare_op = gl_compare_to_pf_compare(ctx->stencil_back.func),
        .reference = ctx->stencil_back.ref,
        .mask = ctx->stencil_back.mask,
        .write_mask = ctx->stencil_back.writemask,
        .fail_op = gl_stencil_op_to_pf(ctx->stencil_back.fail_op),
        .depth_fail_op = gl_stencil_op_to_pf(ctx->stencil_back.zfail_op),
        .pass_op = gl_stencil_op_to_pf(ctx->stencil_back.zpass_op),
    };

    pf_csr_set_stencil_front(csr, &stencil_front);
    pf_csr_set_stencil_back(csr, &stencil_back);

    ctx->dirty &= ~DIRTY_STENCIL;
}

static void upload_cull(gles_context_t *ctx) {
    if (!(ctx->dirty & DIRTY_CULL)) return;

    // wait for rasterizer stage to be idle before uploading cull state
    pixelforge_wait_for_gpu_ready(ctx->dev, GPU_STAGE_PREP_RASTER, NULL);

    volatile uint8_t *csr = ctx->dev->csr_base;

    pixelforge_cull_face_t cull_mode = PIXELFORGE_CULL_NONE;

    if (ctx->cull_face_enabled) {
        if (ctx->cull_face_mode == GL_FRONT) {
            cull_mode = PIXELFORGE_CULL_FRONT;
        } else if (ctx->cull_face_mode == GL_BACK) {
            cull_mode = PIXELFORGE_CULL_BACK;
        } else if (ctx->cull_face_mode == GL_FRONT_AND_BACK) {
            cull_mode = PIXELFORGE_CULL_FRONT_AND_BACK;
        }
    }

    pixelforge_prim_config_t prim = {
        .type = PIXELFORGE_PRIM_TRIANGLES,
        .cull = cull_mode,
        .winding = ctx->front_face == GL_CCW ? PIXELFORGE_WINDING_CCW : PIXELFORGE_WINDING_CW,
    };

    pf_csr_set_prim(csr, &prim);
    ctx->dirty &= ~DIRTY_CULL;
}

static void upload_framebuffer(gles_context_t *ctx) {
    if (!(ctx->dirty & DIRTY_FRAMEBUFFER)) return;

    // wait for per pixel ops to end before changing framebuffer configuration
    pixelforge_wait_for_gpu_ready(ctx->dev, GPU_STAGE_PER_PIXEL, NULL);

    volatile uint8_t *csr = ctx->dev->csr_base;

    pixelforge_framebuffer_config_t fb = {0};
    fb.width = ctx->dev->x_resolution;
    fb.height = ctx->dev->y_resolution;
    fb.viewport_x = fp16_16(ctx->viewport_x);
    fb.viewport_y = fp16_16(ctx->viewport_y);
    fb.viewport_width = fp16_16(ctx->viewport_width);
    fb.viewport_height = fp16_16(ctx->viewport_height);
    fb.viewport_min_depth = fp16_16(0.0f);
    fb.viewport_max_depth = fp16_16(1.0f);
    fb.scissor_offset_x = ctx->scissor_x;
    fb.scissor_offset_y = ctx->scissor_y;
    fb.scissor_width = ctx->scissor_width;
    fb.scissor_height = ctx->scissor_height;
    fb.color_address = ctx->dev->buffer_phys[ctx->dev->render_buffer];
    fb.color_pitch = ctx->dev->buffer_stride;
    fb.depthstencil_address = 0;
    fb.depthstencil_pitch = 0;

    pf_csr_set_fb(csr, &fb);
    ctx->dirty &= ~DIRTY_FRAMEBUFFER;
}

/* ============================================================================
 * Context Management
 * ============================================================================ */

static void wait_for_draw(gles_context_t *ctx) {
    pixelforge_wait_for_gpu_ready(ctx->dev, GPU_STAGE_PER_PIXEL, NULL);
}

bool glInit(void) {
    if (g_ctx) {
        return false;  /* Already initialized */
    }

    g_ctx = calloc(1, sizeof(gles_context_t));
    if (!g_ctx) return false;

    g_ctx->dev = pixelforge_open_dev();
    if (!g_ctx->dev) {
        free(g_ctx);
        g_ctx = NULL;
        return false;
    }

    /* Initialize matrix stacks */
    init_matrix_stack(&g_ctx->modelview_stack);
    init_matrix_stack(&g_ctx->projection_stack);
    init_matrix_stack(&g_ctx->texture_stack);
    g_ctx->matrix_mode = GL_MODELVIEW;
    g_ctx->current_matrix = get_current_stack_matrix(g_ctx);

    /* Initialize default material */
    set_vec4(g_ctx->material.ambient, 0.2f, 0.2f, 0.2f, 1.0f);
    set_vec4(g_ctx->material.diffuse, 0.8f, 0.8f, 0.8f, 1.0f);
    set_vec4(g_ctx->material.specular, 0.0f, 0.0f, 0.0f, 1.0f);
    g_ctx->material.shininess = 0.0f;

    /* Initialize default light */
    for (int i = 0; i < MAX_LIGHTS; i++) {
        g_ctx->lights[i].enabled = false;
        set_vec4(g_ctx->lights[i].position, 0.0f, 0.0f, 1.0f, 0.0f);
        set_vec4(g_ctx->lights[i].ambient, 0.0f, 0.0f, 0.0f, 1.0f);
        set_vec4(g_ctx->lights[i].diffuse, 1.0f, 1.0f, 1.0f, 1.0f);
        set_vec4(g_ctx->lights[i].specular, 1.0f, 1.0f, 1.0f, 1.0f);
    }

    /* Initialize default state */
    g_ctx->depth_test_enabled = false;
    g_ctx->depth_write_enabled = true;
    g_ctx->depth_func = GL_LESS;

    g_ctx->blend_enabled = false;
    g_ctx->blend_src_factor = GL_ONE;
    g_ctx->blend_dst_factor = GL_ZERO;

    g_ctx->stencil_test_enabled = false;

    for (int i = 0; i < 2; i++) {
        stencil_config_t *conf = (i == 0) ? &g_ctx->stencil_front : &g_ctx->stencil_back;
        conf->func = GL_ALWAYS;
        conf->ref = 0;
        conf->mask = 0xFF;
        conf->writemask = 0xFF;
        conf->fail_op = GL_KEEP;
        conf->zfail_op = GL_KEEP;
        conf->zpass_op = GL_KEEP;
    }

    g_ctx->cull_face_enabled = false;
    g_ctx->cull_face_mode = GL_BACK;
    g_ctx->front_face = GL_CCW;

    g_ctx->viewport_x = 0;
    g_ctx->viewport_y = 0;
    g_ctx->viewport_width = g_ctx->dev->x_resolution;
    g_ctx->viewport_height = g_ctx->dev->y_resolution;

    g_ctx->scissor_x = 0;
    g_ctx->scissor_y = 0;
    g_ctx->scissor_width = g_ctx->dev->x_resolution;
    g_ctx->scissor_height = g_ctx->dev->y_resolution;

    set_vec4(g_ctx->clear_color, 0.0f, 0.0f, 0.0f, 1.0f);
    g_ctx->clear_depth = 1.0f;
    g_ctx->clear_stencil = 0;

    g_ctx->next_buffer_id = 1;
    g_ctx->array_buffer_binding = 0;
    g_ctx->element_array_buffer_binding = 0;

    /* Mark everything as dirty */
    g_ctx->dirty = 0xFFFFFFFF;

    return true;
}

void glDestroy(void) {
    if (!g_ctx) return;

    /* Wait for any in-flight draws */
    wait_for_draw(g_ctx);

    pixelforge_close_dev(g_ctx->dev);
    free(g_ctx->buffers);
    free(g_ctx);
    g_ctx = NULL;
}

/* ============================================================================
 * State Management
 * ============================================================================ */

void glEnable(GLenum cap) {
    if (!g_ctx) return;

    switch (cap) {
        case GL_DEPTH_TEST:
            g_ctx->depth_test_enabled = true;
            g_ctx->dirty |= DIRTY_DEPTH;
            break;
        case GL_BLEND:
            g_ctx->blend_enabled = true;
            g_ctx->dirty |= DIRTY_BLEND;
            break;
        case GL_STENCIL_TEST:
            g_ctx->stencil_test_enabled = true;
            g_ctx->dirty |= DIRTY_STENCIL;
            break;
        case GL_CULL_FACE:
            g_ctx->cull_face_enabled = true;
            g_ctx->dirty |= DIRTY_CULL;
            break;
        case GL_LIGHTING:
            g_ctx->lighting_enabled = true;
            g_ctx->dirty |= DIRTY_MATERIAL | DIRTY_LIGHTS;
            break;
    }

    if (cap >= GL_LIGHT0 && cap < GL_LIGHT0 + MAX_LIGHTS) {
        int light_index = cap - GL_LIGHT0;
        g_ctx->lights[light_index].enabled = true;
        g_ctx->dirty |= DIRTY_LIGHTS;
    }
}

void glDisable(GLenum cap) {
    if (!g_ctx) return;

    switch (cap) {
        case GL_DEPTH_TEST:
            g_ctx->depth_test_enabled = false;
            g_ctx->dirty |= DIRTY_DEPTH;
            break;
        case GL_BLEND:
            g_ctx->blend_enabled = false;
            g_ctx->dirty |= DIRTY_BLEND;
            break;
        case GL_STENCIL_TEST:
            g_ctx->stencil_test_enabled = false;
            g_ctx->dirty |= DIRTY_STENCIL;
            break;
        case GL_CULL_FACE:
            g_ctx->cull_face_enabled = false;
            g_ctx->dirty |= DIRTY_CULL;
            break;
        case GL_LIGHTING:
            g_ctx->lighting_enabled = false;
            g_ctx->dirty |= DIRTY_MATERIAL | DIRTY_LIGHTS;
            break;
    }

    if (cap >= GL_LIGHT0 && cap < GL_LIGHT0 + MAX_LIGHTS) {
        int light_index = cap - GL_LIGHT0;
        g_ctx->lights[light_index].enabled = false;
        g_ctx->dirty |= DIRTY_LIGHTS;
    }
}

void glClearColor(GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha) {
    if (!g_ctx) return;
    g_ctx->clear_color[0] = red;
    g_ctx->clear_color[1] = green;
    g_ctx->clear_color[2] = blue;
    g_ctx->clear_color[3] = alpha;
}

void glClearDepthf(GLclampf depth) {
    if (!g_ctx) return;
    g_ctx->clear_depth = depth;
}

void glClearStencil(GLint s) {
    if (!g_ctx) return;
    g_ctx->clear_stencil = s;
}

void glClear(GLbitfield mask) {
    if (!g_ctx) return;

    /* Wait for GPU before clearing */
    wait_for_draw(g_ctx);

    uint8_t *buffer = pixelforge_get_back_buffer(g_ctx->dev);

    bool clear_color = (mask & GL_COLOR_BUFFER_BIT) != 0;
    bool clear_depth = (mask & GL_DEPTH_BUFFER_BIT) != 0;
    bool clear_stencil = (mask & GL_STENCIL_BUFFER_BIT) != 0;

    if (clear_color) {
        uint8_t r = (uint8_t)(g_ctx->clear_color[0] * 255.0f);
        uint8_t g = (uint8_t)(g_ctx->clear_color[1] * 255.0f);
        uint8_t b = (uint8_t)(g_ctx->clear_color[2] * 255.0f);
        uint8_t a = (uint8_t)(g_ctx->clear_color[3] * 255.0f);
        uint32_t color = (a << 24) | (r << 16) | (g << 8) | b;

        for (size_t i = 0; i < g_ctx->dev->buffer_size / 4; i++) {
            ((uint32_t*)buffer)[i] = color;
        }
    }

    if (clear_depth && clear_stencil) {
        // Depth-Stencil is in D16_X8_S8 format
        uint32_t depth_stencil_value = (uint32_t)(g_ctx->clear_depth * 65535.0f) | ((g_ctx->clear_stencil & 0xFF) << 24);
        uint32_t *depth_stencil_buffer = (uint32_t*)(g_ctx->dev->depthstencil_buffer);

        for (size_t i = 0; i < g_ctx->dev->buffer_size / 4; i++) {
            depth_stencil_buffer[i] = depth_stencil_value;
        }
    } else if (clear_depth) {
        uint32_t depth_value = (uint32_t)(g_ctx->clear_depth * 65535.0f); // Stencil bits are preserved
        uint32_t *depth_stencil_buffer = (uint32_t*)(g_ctx->dev->depthstencil_buffer);

        for (size_t i = 0; i < g_ctx->dev->buffer_size / 4; i++) {
            depth_stencil_buffer[i] = depth_value | (depth_stencil_buffer[i] & 0xFF000000);
        }
    } else if (clear_stencil) {
        uint32_t stencil_value = (g_ctx->clear_stencil & 0xFF) << 24; // Depth bits are preserved
        uint32_t *depth_stencil_buffer = (uint32_t*)(g_ctx->dev->depthstencil_buffer);

        for (size_t i = 0; i < g_ctx->dev->buffer_size / 4; i++) {
            depth_stencil_buffer[i] = (depth_stencil_buffer[i] & 0x00FFFFFF) | stencil_value;
        }
    }
}

void glViewport(GLint x, GLint y, GLsizei width, GLsizei height) {
    if (!g_ctx) return;
    g_ctx->viewport_x = (float)x;
    g_ctx->viewport_y = (float)y;
    g_ctx->viewport_width = (float)width;
    g_ctx->viewport_height = (float)height;
    g_ctx->dirty |= DIRTY_FRAMEBUFFER;
}

void glScissor(GLint x, GLint y, GLsizei width, GLsizei height) {
    if (!g_ctx) return;
    g_ctx->scissor_x = x;
    g_ctx->scissor_y = y;
    g_ctx->scissor_width = width;
    g_ctx->scissor_height = height;
    g_ctx->dirty |= DIRTY_FRAMEBUFFER;
}

/* ============================================================================
 * Depth Testing
 * ============================================================================ */

void glDepthFunc(GLenum func) {
    if (!g_ctx) return;
    g_ctx->depth_func = func;
    g_ctx->dirty |= DIRTY_DEPTH;
}

void glDepthMask(bool flag) {
    if (!g_ctx) return;
    g_ctx->depth_write_enabled = flag;
    g_ctx->dirty |= DIRTY_DEPTH;
}

/* ============================================================================
 * Blending
 * ============================================================================ */

void glBlendFunc(GLenum sfactor, GLenum dfactor) {
    if (!g_ctx) return;
    g_ctx->blend_src_factor = sfactor;
    g_ctx->blend_dst_factor = dfactor;
    g_ctx->dirty |= DIRTY_BLEND;
}

/* ============================================================================
 * Culling
 * ============================================================================ */

void glCullFace(GLenum mode) {
    if (!g_ctx) return;
    g_ctx->cull_face_mode = mode;
    g_ctx->dirty |= DIRTY_CULL;
}

void glFrontFace(GLenum mode) {
    if (!g_ctx) return;
    g_ctx->front_face = mode;
    g_ctx->dirty |= DIRTY_CULL;
}

/* ============================================================================
 * Stencil
 * ============================================================================ */

void glStencilFunc(GLenum func, GLint ref, GLuint mask) {
    if (!g_ctx) return;
    g_ctx->stencil_front.func = func;
    g_ctx->stencil_front.ref = ref;
    g_ctx->stencil_front.mask = mask;
    g_ctx->stencil_back.func = func;
    g_ctx->stencil_back.ref = ref;
    g_ctx->stencil_back.mask = mask;
    g_ctx->dirty |= DIRTY_STENCIL;
}

void glStencilOp(GLenum fail, GLenum zfail, GLenum zpass) {
    if (!g_ctx) return;
    g_ctx->stencil_front.fail_op = fail;
    g_ctx->stencil_front.zfail_op = zfail;
    g_ctx->stencil_front.zpass_op = zpass;
    g_ctx->stencil_back.fail_op = fail;
    g_ctx->stencil_back.zfail_op = zfail;
    g_ctx->stencil_back.zpass_op = zpass;
    g_ctx->dirty |= DIRTY_STENCIL;
}

void glStencilMask(GLuint mask) {
    if (!g_ctx) return;
    g_ctx->stencil_front.writemask = mask;
    g_ctx->stencil_back.writemask = mask;
    g_ctx->dirty |= DIRTY_STENCIL;
}

/* ============================================================================
 * Matrix Operations
 * ============================================================================ */

void glMatrixMode(GLenum mode) {
    if (!g_ctx) return;
    g_ctx->matrix_mode = mode;
    g_ctx->current_matrix = get_current_stack_matrix(g_ctx);
}

void glLoadIdentity(void) {
    if (!g_ctx) return;
    mat4_identity(g_ctx->current_matrix);
    g_ctx->dirty |= DIRTY_MATRICES;
}

void glLoadMatrixf(const GLfloat *m) {
    if (!g_ctx || !m) return;
    mat4_copy(g_ctx->current_matrix, m);
    g_ctx->dirty |= DIRTY_MATRICES;
}

void glMultMatrixf(const GLfloat *m) {
    if (!g_ctx || !m) return;
    mat4_multiply(g_ctx->current_matrix, g_ctx->current_matrix, m);
    g_ctx->dirty |= DIRTY_MATRICES;
}

void glPushMatrix(void) {
    if (!g_ctx) return;

    matrix_stack_t *stack = NULL;
    int max_depth = 0;

    switch (g_ctx->matrix_mode) {
        case GL_MODELVIEW:
            stack = &g_ctx->modelview_stack;
            max_depth = MAX_MODELVIEW_STACK_DEPTH;
            break;
        case GL_PROJECTION:
            stack = &g_ctx->projection_stack;
            max_depth = MAX_PROJECTION_STACK_DEPTH;
            break;
        case GL_TEXTURE:
            stack = &g_ctx->texture_stack;
            max_depth = MAX_TEXTURE_STACK_DEPTH;
            break;
        default:
            return;
    }

    if (stack->depth >= max_depth - 1) {
        assert(false && "Matrix stack overflow");
        return;  /* Stack overflow */
    }

    mat4_copy(stack->matrices[stack->depth + 1], stack->matrices[stack->depth]);
    stack->depth++;
    g_ctx->current_matrix = stack->matrices[stack->depth];
}

void glPopMatrix(void) {
    if (!g_ctx) return;

    matrix_stack_t *stack = NULL;

    switch (g_ctx->matrix_mode) {
        case GL_MODELVIEW:
            stack = &g_ctx->modelview_stack;
            break;
        case GL_PROJECTION:
            stack = &g_ctx->projection_stack;
            break;
        case GL_TEXTURE:
            stack = &g_ctx->texture_stack;
            break;
        default:
            return;
    }

    if (stack->depth <= 0) {
        assert(false && "Matrix stack underflow");
        return;  /* Stack underflow */
    }

    stack->depth--;
    g_ctx->current_matrix = stack->matrices[stack->depth];
    g_ctx->dirty |= DIRTY_MATRICES;
}

void glTranslatef(GLfloat x, GLfloat y, GLfloat z) {
    if (!g_ctx) return;

    float trans[16];
    mat4_identity(trans);
    trans[12] = x;
    trans[13] = y;
    trans[14] = z;

    mat4_multiply(g_ctx->current_matrix, g_ctx->current_matrix, trans);
    g_ctx->dirty |= DIRTY_MATRICES;
}

void glRotatef(GLfloat angle, GLfloat x, GLfloat y, GLfloat z) {
    if (!g_ctx) return;

    float rad = angle * M_PI / 180.0f;
    float c = cosf(rad);
    float s = sinf(rad);
    float len = sqrtf(x*x + y*y + z*z);
    if (len < 0.0001f) return;

    x /= len;
    y /= len;
    z /= len;

    float rot[16];
    rot[0] = x*x*(1-c)+c;   rot[4] = x*y*(1-c)-z*s; rot[8] = x*z*(1-c)+y*s;  rot[12] = 0;
    rot[1] = y*x*(1-c)+z*s; rot[5] = y*y*(1-c)+c;   rot[9] = y*z*(1-c)-x*s;  rot[13] = 0;
    rot[2] = z*x*(1-c)-y*s; rot[6] = z*y*(1-c)+x*s; rot[10] = z*z*(1-c)+c;   rot[14] = 0;
    rot[3] = 0;             rot[7] = 0;             rot[11] = 0;             rot[15] = 1;

    mat4_multiply(g_ctx->current_matrix, g_ctx->current_matrix, rot);
    g_ctx->dirty |= DIRTY_MATRICES;
}

void glScalef(GLfloat x, GLfloat y, GLfloat z) {
    if (!g_ctx) return;

    float scale[16];
    mat4_identity(scale);
    scale[0] = x;
    scale[5] = y;
    scale[10] = z;

    mat4_multiply(g_ctx->current_matrix, g_ctx->current_matrix, scale);
    g_ctx->dirty |= DIRTY_MATRICES;
}

void glFrustumf(GLfloat left, GLfloat right, GLfloat bottom, GLfloat top, GLfloat near, GLfloat far) {
    if (!g_ctx) return;

    float frustum[16];
    memset(frustum, 0, sizeof(frustum));

    frustum[0] = (2.0f * near) / (right - left);
    frustum[5] = (2.0f * near) / (top - bottom);
    frustum[8] = (right + left) / (right - left);
    frustum[9] = (top + bottom) / (top - bottom);
    frustum[10] = -(far + near) / (far - near);
    frustum[11] = -1.0f;
    frustum[14] = -(2.0f * far * near) / (far - near);

    mat4_multiply(g_ctx->current_matrix, g_ctx->current_matrix, frustum);
    g_ctx->dirty |= DIRTY_MATRICES;
}

void glOrthof(GLfloat left, GLfloat right, GLfloat bottom, GLfloat top, GLfloat near, GLfloat far) {
    if (!g_ctx) return;

    float ortho[16];
    mat4_identity(ortho);

    ortho[0] = 2.0f / (right - left);
    ortho[5] = 2.0f / (top - bottom);
    ortho[10] = -2.0f / (far - near);
    ortho[12] = -(right + left) / (right - left);
    ortho[13] = -(top + bottom) / (top - bottom);
    ortho[14] = -(far + near) / (far - near);

    mat4_multiply(g_ctx->current_matrix, g_ctx->current_matrix, ortho);
    g_ctx->dirty |= DIRTY_MATRICES;
}

/* ============================================================================
 * Lighting
 * ============================================================================ */

void glLightfv(GLenum light, GLenum pname, const GLfloat *params) {
    if (!g_ctx || !params) return;

    unsigned int light_index = light - GL_LIGHT0;
    if (light_index >= MAX_LIGHTS) return;

    light_t *light_struct = &g_ctx->lights[light_index];

    switch (pname) {
        case GL_POSITION:
            memcpy(light_struct->position, params, 4 * sizeof(float));
            g_ctx->dirty |= DIRTY_LIGHTS;
            break;
        case GL_AMBIENT:
            memcpy(light_struct->ambient, params, 3 * sizeof(float));
            g_ctx->dirty |= DIRTY_LIGHTS;
            break;
        case GL_DIFFUSE:
            memcpy(light_struct->diffuse, params, 3 * sizeof(float));
            g_ctx->dirty |= DIRTY_LIGHTS;
            break;
        case GL_SPECULAR:
            memcpy(light_struct->specular, params, 3 * sizeof(float));
            g_ctx->dirty |= DIRTY_LIGHTS;
            break;
    }
}

void glMaterialfv(GLenum face, GLenum pname, const GLfloat *params) {
    if (!g_ctx || !params) return;

    (void)face; // not available in GL ES 1.x

    switch (pname) {
        case GL_AMBIENT:
            memcpy(g_ctx->material.ambient, params, 4 * sizeof(float));
            g_ctx->dirty |= DIRTY_MATERIAL;
            break;
        case GL_DIFFUSE:
            memcpy(g_ctx->material.diffuse, params, 4 * sizeof(float));
            g_ctx->dirty |= DIRTY_MATERIAL;
            break;
        case GL_SPECULAR:
            memcpy(g_ctx->material.specular, params, 4 * sizeof(float));
            g_ctx->dirty |= DIRTY_MATERIAL;
            break;
        case GL_EMISSION:
            assert(false && "Material emission not supported");
            break;
        case GL_SHININESS:
            g_ctx->material.shininess = params[0];
            g_ctx->dirty |= DIRTY_MATERIAL;
            break;
        default:
            assert(false && "Unknown material parameter");
            break;
    }
}

/* ============================================================================
 * Vertex Arrays
 * ============================================================================ */

void glEnableClientState(GLenum array) {
    if (!g_ctx) return;

    switch (array) {
        case GL_VERTEX_ARRAY:
            g_ctx->vertex_array.enabled = true;
            g_ctx->dirty |= DIRTY_VERTEX_ARRAYS;
            break;
        case GL_NORMAL_ARRAY:
            g_ctx->normal_array.enabled = true;
            g_ctx->dirty |= DIRTY_VERTEX_ARRAYS;
            break;
        case GL_COLOR_ARRAY:
            g_ctx->color_array.enabled = true;
            g_ctx->dirty |= DIRTY_VERTEX_ARRAYS;
            break;
    }
}

void glDisableClientState(GLenum array) {
    if (!g_ctx) return;

    switch (array) {
        case GL_VERTEX_ARRAY:
            g_ctx->vertex_array.enabled = false;
            g_ctx->dirty |= DIRTY_VERTEX_ARRAYS;
            break;
        case GL_NORMAL_ARRAY:
            g_ctx->normal_array.enabled = false;
            g_ctx->dirty |= DIRTY_VERTEX_ARRAYS;
            break;
        case GL_COLOR_ARRAY:
            g_ctx->color_array.enabled = false;
            g_ctx->dirty |= DIRTY_VERTEX_ARRAYS;
            break;
    }
}

void glVertexPointer(GLint size, GLenum type, GLsizei stride, const GLvoid *pointer) {
    if (!g_ctx) return;

    assert(size == 4);
    assert(type == GL_FIXED);

    if (stride == 0) stride = size * sizeof(uint32_t); // fp16_16

    if (g_ctx->array_buffer_binding == 0) return;

    g_ctx->vertex_array.buffer = g_ctx->array_buffer_binding;
    g_ctx->vertex_array.offset = (size_t)(uintptr_t)pointer;
    g_ctx->vertex_array.size = size;
    g_ctx->vertex_array.type = type;
    g_ctx->vertex_array.stride = stride;
    g_ctx->dirty |= DIRTY_VERTEX_ARRAYS;
}

void glNormalPointer(GLenum type, GLsizei stride, const GLvoid *pointer) {
    if (!g_ctx) return;

    assert(type == GL_FIXED);

    GLint size = 3;  /* Normals are always 3 components */
    if (stride == 0) stride = size * sizeof(uint32_t); // fp16_16

    if (g_ctx->array_buffer_binding == 0) return;

    g_ctx->normal_array.buffer = g_ctx->array_buffer_binding;
    g_ctx->normal_array.offset = (size_t)(uintptr_t)pointer;
    g_ctx->normal_array.size = size;
    g_ctx->normal_array.type = type;
    g_ctx->normal_array.stride = stride;
    g_ctx->dirty |= DIRTY_VERTEX_ARRAYS;
}

void glColorPointer(GLint size, GLenum type, GLsizei stride, const GLvoid *pointer) {
    if (!g_ctx) return;

    assert(size == 4);
    assert(type == GL_FIXED);

    if (stride == 0) stride = size * sizeof(uint32_t); // fp16_16

    if (g_ctx->array_buffer_binding == 0) return;

    g_ctx->color_array.buffer = g_ctx->array_buffer_binding;
    g_ctx->color_array.offset = (size_t)(uintptr_t)pointer;
    g_ctx->color_array.size = size;
    g_ctx->color_array.type = type;
    g_ctx->color_array.stride = stride;
    g_ctx->dirty |= DIRTY_VERTEX_ARRAYS;
}

/* ============================================================================
 * Drawing Commands
 * ============================================================================ */

static void draw_generic(
    gles_context_t *ctx,
    bool indexed,
    GLenum mode,
    GLsizei base_vertex, GLsizei count,
    GLenum idx_type,
    const void *indices) {
    if (!ctx || count <= 0) return;

    /* Upload all dirty state */
    upload_matrices(ctx);
    upload_material(ctx);
    upload_lights(ctx);
    upload_depth(ctx);
    upload_blend(ctx);
    upload_stencil(ctx);
    upload_cull(ctx);
    upload_framebuffer(ctx);

    // wait for input assembly for the per-draw state to be applied before configuring vertex attributes
    pixelforge_wait_for_gpu_ready(ctx->dev, GPU_STAGE_IA, NULL);

    volatile uint8_t *csr = ctx->dev->csr_base;

    /* Set topology */
    pixelforge_topo_config_t topo = {
        .input_topology = gl_mode_to_topology(mode),
        .primitive_restart_enable = false,
        .primitive_restart_index = 0,
        .base_vertex = base_vertex,
    };
    pf_csr_set_topology(csr, &topo);

    /* Determine index type */
    pixelforge_index_kind_t idx_kind = PIXELFORGE_INDEX_NOT_INDEXED;
    if (indexed) {
        switch (idx_type) {
            case GL_UNSIGNED_BYTE:  idx_kind = PIXELFORGE_INDEX_U8; break;
            case GL_UNSIGNED_SHORT: idx_kind = PIXELFORGE_INDEX_U16; break;
            default:                idx_kind = PIXELFORGE_INDEX_U16; break;
        }
    }

    /* Set index config - indices pointer is expected to be offset into bound element buffer */
    pixelforge_idx_config_t idx_cfg = {
        .address = 0,
        .count = count,
        .kind = idx_kind,
    };

    if (indexed) {
        if (g_ctx->element_array_buffer_binding == 0) return;
        gl_buffer_t *idx_buffer = get_buffer_by_id(ctx, g_ctx->element_array_buffer_binding);
        if (!idx_buffer) return;

        size_t idx_offset = (size_t)(uintptr_t)indices;
        if (idx_offset >= idx_buffer->size) return;

        idx_cfg.address = idx_buffer->phys + (uint32_t)idx_offset;
    }

    pf_csr_set_idx(csr, &idx_cfg);

    /* Configure vertex attributes */
    pixelforge_input_attr_t attr = {0};

    if (g_ctx->vertex_array.enabled) {
        if (g_ctx->vertex_array.buffer != 0) {
            gl_buffer_t *pos_buffer = get_buffer_by_id(ctx, g_ctx->vertex_array.buffer);
            if (!pos_buffer) return;
            if (g_ctx->vertex_array.offset >= pos_buffer->size) return;

            attr.mode = PIXELFORGE_ATTR_PER_VERTEX;
            attr.info.per_vertex.address = pos_buffer->phys + (uint32_t)g_ctx->vertex_array.offset;
            attr.info.per_vertex.stride = g_ctx->vertex_array.stride;
            pf_csr_set_attr_position(csr, &attr);
        } else {
            // set constant positions 0,0,0,1
            attr.mode = PIXELFORGE_ATTR_CONSTANT;
            attr.info.constant_value.value[0] = fp16_16(0.0f);
            attr.info.constant_value.value[1] = fp16_16(0.0f);
            attr.info.constant_value.value[2] = fp16_16(0.0f);
            attr.info.constant_value.value[3] = fp16_16(1.0f);
            pf_csr_set_attr_position(csr, &attr);
        }
    }

    if (g_ctx->normal_array.enabled) {
        if (g_ctx->normal_array.buffer != 0) {
            gl_buffer_t *norm_buffer = get_buffer_by_id(ctx, g_ctx->normal_array.buffer);
            if (!norm_buffer) return;
            if (g_ctx->normal_array.offset >= norm_buffer->size) return;

            attr.mode = PIXELFORGE_ATTR_PER_VERTEX;
            attr.info.per_vertex.address = norm_buffer->phys + (uint32_t)g_ctx->normal_array.offset;
            attr.info.per_vertex.stride = g_ctx->normal_array.stride;
            pf_csr_set_attr_normal(csr, &attr);
        } else {
            // set constant normal 0,0,1,0
            attr.mode = PIXELFORGE_ATTR_CONSTANT;
            attr.info.constant_value.value[0] = fp16_16(0.0f);
            attr.info.constant_value.value[1] = fp16_16(0.0f);
            attr.info.constant_value.value[2] = fp16_16(1.0f);
            attr.info.constant_value.value[3] = fp16_16(0.0f);
            pf_csr_set_attr_normal(csr, &attr);
        }
    }

    if (g_ctx->color_array.enabled) {
        if (g_ctx->color_array.buffer != 0) {
            gl_buffer_t *color_buffer = get_buffer_by_id(ctx, g_ctx->color_array.buffer);
            if (!color_buffer) return;
            if (g_ctx->color_array.offset >= color_buffer->size) return;

            attr.mode = PIXELFORGE_ATTR_PER_VERTEX;
            attr.info.per_vertex.address = color_buffer->phys + (uint32_t)g_ctx->color_array.offset;
            attr.info.per_vertex.stride = g_ctx->color_array.stride;
            pf_csr_set_attr_color(csr, &attr);
        } else {
            // set constant color 1,1,1,1
            attr.mode = PIXELFORGE_ATTR_CONSTANT;
            attr.info.constant_value.value[0] = fp16_16(1.0f);
            attr.info.constant_value.value[1] = fp16_16(1.0f);
            attr.info.constant_value.value[2] = fp16_16(1.0f);
            attr.info.constant_value.value[3] = fp16_16(1.0f);
            pf_csr_set_attr_color(csr, &attr);
        }
    }

    for (int i = 0; i < NUM_TEXTURES; i++) {
        if (g_ctx->texcoord_arrays[i].enabled) {
            if (g_ctx->texcoord_arrays[i].buffer == 0) return;

            gl_buffer_t *tex_buffer = get_buffer_by_id(ctx, g_ctx->texcoord_arrays[i].buffer);
            if (!tex_buffer) return;
            if (g_ctx->texcoord_arrays[i].offset >= tex_buffer->size) return;

            attr.mode = PIXELFORGE_ATTR_PER_VERTEX;
            attr.info.per_vertex.address = tex_buffer->phys + (uint32_t)g_ctx->texcoord_arrays[i].offset;
            attr.info.per_vertex.stride = g_ctx->texcoord_arrays[i].stride;
            pf_csr_set_attr_texcoord(csr, i, &attr);
        } else {
            // set constant texture coordinates 0,0,0,1
            attr.mode = PIXELFORGE_ATTR_CONSTANT;
            attr.info.constant_value.value[0] = fp16_16(0.0f);
            attr.info.constant_value.value[1] = fp16_16(0.0f);
            attr.info.constant_value.value[2] = fp16_16(0.0f);
            attr.info.constant_value.value[3] = fp16_16(1.0f);
            pf_csr_set_attr_texcoord(csr, i, &attr);
        }
    }

    // start the draw
    pf_csr_start(csr);
}

void glDrawArrays(GLenum mode, GLint first, GLsizei count) {
    draw_generic(g_ctx, false, mode, first, count, 0, NULL);
}

void glDrawElements(GLenum mode, GLsizei count, GLenum type, const GLvoid *indices) {
    draw_generic(g_ctx, true, mode, 0, count, type, indices);
}

/* ============================================================================
 * Buffer Swap
 * ============================================================================ */

void glSwapBuffers(void) {
    if (!g_ctx) return;

    /* Wait for any in-flight draw operations before swapping */
    wait_for_draw(g_ctx);

    /* Swap buffers using existing implementation */
    pixelforge_swap_buffers(g_ctx->dev);

    /* Framebuffer address changed, mark as dirty for next draw */
    g_ctx->dirty |= DIRTY_FRAMEBUFFER;
}

/* =========================================================================
 * Buffer Objects (Handle-Based)
 * ============================================================================ */

void glGenBuffers(GLsizei n, GLuint *buffers) {
    if (!g_ctx || !buffers || n <= 0) return;

    for (GLsizei i = 0; i < n; i++) {
        GLuint id = g_ctx->next_buffer_id++;
        gl_buffer_t *buf = create_buffer(g_ctx, id);
        if (!buf) {
            buffers[i] = 0;
            continue;
        }
        buffers[i] = id;
    }
}

void glDeleteBuffers(GLsizei n, const GLuint *buffers) {
    if (!g_ctx || !buffers || n <= 0) return;

    for (GLsizei i = 0; i < n; i++) {
        gl_buffer_t *buf = get_buffer_by_id(g_ctx, buffers[i]);
        if (!buf) continue;

        if (g_ctx->array_buffer_binding == buf->id) {
            g_ctx->array_buffer_binding = 0;
        }
        if (g_ctx->element_array_buffer_binding == buf->id) {
            g_ctx->element_array_buffer_binding = 0;
        }

        if (g_ctx->vertex_array.buffer == buf->id) {
            g_ctx->vertex_array.enabled = false;
            g_ctx->vertex_array.buffer = 0;
        }
        if (g_ctx->normal_array.buffer == buf->id) {
            g_ctx->normal_array.enabled = false;
            g_ctx->normal_array.buffer = 0;
        }
        if (g_ctx->color_array.buffer == buf->id) {
            g_ctx->color_array.enabled = false;
            g_ctx->color_array.buffer = 0;
        }

        for (int t = 0; t < NUM_TEXTURES; t++) {
            if (g_ctx->texcoord_arrays[t].buffer == buf->id) {
                g_ctx->texcoord_arrays[t].enabled = false;
                g_ctx->texcoord_arrays[t].buffer = 0;
            }
        }

        buf->alive = false;
        buf->virt = NULL;
        buf->phys = 0;
        buf->size = 0;
    }
}

void glBindBuffer(GLenum target, GLuint buffer) {
    if (!g_ctx) return;

    switch (target) {
        case GL_ARRAY_BUFFER:
            if (buffer == 0 || get_buffer_by_id(g_ctx, buffer)) {
                g_ctx->array_buffer_binding = buffer;
            }
            break;
        case GL_ELEMENT_ARRAY_BUFFER:
            if (buffer == 0 || get_buffer_by_id(g_ctx, buffer)) {
                g_ctx->element_array_buffer_binding = buffer;
            }
            break;
        default:
            break;
    }
}

void glBufferData(GLenum target, size_t size, const void *data, GLenum usage) {
    (void)usage;
    if (!g_ctx) return;

    GLuint bound = 0;
    if (target == GL_ARRAY_BUFFER) {
        bound = g_ctx->array_buffer_binding;
    } else if (target == GL_ELEMENT_ARRAY_BUFFER) {
        bound = g_ctx->element_array_buffer_binding;
    } else {
        return;
    }

    gl_buffer_t *buf = get_buffer_by_id(g_ctx, bound);
    if (!buf) return;

    if (size == 0) {
        buf->size = 0;
        buf->virt = NULL;
        buf->phys = 0;
        return;
    }

    if (!buf->virt || size > buf->size) {
        struct vram_block block;
        if (vram_alloc(&g_ctx->dev->vram, size, 4096, &block) != 0) {
            return;
        }
        buf->virt = block.virt;
        buf->phys = block.phys;
    }

    buf->size = size;

    if (data) {
        pixelforge_wait_for_gpu_ready(g_ctx->dev, GPU_STAGE_IA, NULL);
        memcpy(buf->virt, data, size);
    }
}

void glBufferSubData(GLenum target, size_t offset, size_t size, const void *data) {
    if (!g_ctx || !data) return;

    GLuint bound = 0;
    if (target == GL_ARRAY_BUFFER) {
        bound = g_ctx->array_buffer_binding;
    } else if (target == GL_ELEMENT_ARRAY_BUFFER) {
        bound = g_ctx->element_array_buffer_binding;
    } else {
        return;
    }

    gl_buffer_t *buf = get_buffer_by_id(g_ctx, bound);
    if (!buf || !buf->virt) return;
    if (offset > buf->size || size > buf->size - offset) return;

    pixelforge_wait_for_gpu_ready(g_ctx->dev, GPU_STAGE_IA, NULL);
    memcpy((uint8_t*)buf->virt + offset, data, size);
}
