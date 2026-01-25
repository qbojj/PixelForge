#ifndef PIXELFORGE_DEMO_UTILS_H
#define PIXELFORGE_DEMO_UTILS_H

#include <stdint.h>
#include <math.h>

/* Common vertex structure for demos */
struct demo_vertex {
    int32_t pos[4];
    int32_t norm[3];
    int32_t col[4];
};

/* Geometry helpers */
void demo_create_cube(struct demo_vertex *vertices, uint16_t *indices, uint32_t *idx_count);

/* Column-major 4x4 and 3x3 matrix helpers (OpenGL-style) */
void mat4_identity(float m[16]);
void mat4_perspective(float m[16], float fovy, float aspect, float near, float far);
void mat4_rotate_xyz(float m[16], float rx, float ry, float rz);
void mat4_translate(float m[16], float x, float y, float z);
void mat4_scale(float m[16], float sx, float sy, float sz);
void mat4_multiply(float out[16], const float a[16], const float b[16]);

// Convert 4x4 matrix to 3x3 normal matrix (inverse transpose of upper-left 3x3)
void mat3_from_mat4(float m3[9], const float m4[16]);

void mat4_to_fp16_16(int32_t out[16], const float in[16]);
void mat3_to_fp16_16(int32_t out[9], const float in[9]);

#endif /* PIXELFORGE_DEMO_UTILS_H */
