#ifndef PIXELFORGE_DEMO_UTILS_H
#define PIXELFORGE_DEMO_UTILS_H

#include <math.h>

/* Column-major 4x4 and 3x3 matrix helpers (OpenGL-style) */
void mat4_identity(float m[16]);
void mat4_perspective(float m[16], float fovy, float aspect, float near, float far);
void mat4_rotate_xyz(float m[16], float rx, float ry, float rz);
void mat4_translate(float m[16], float x, float y, float z);
void mat4_scale(float m[16], float sx, float sy, float sz);
void mat4_multiply(float out[16], const float a[16], const float b[16]);
void mat3_from_mat4(float m3[9], const float m4[16]);

#endif /* PIXELFORGE_DEMO_UTILS_H */
