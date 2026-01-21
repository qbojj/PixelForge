#include <string.h>
#include "demo_utils.h"

/* Column-major (OpenGL-style) implementations */

void mat4_identity(float m[16]) {
    memset(m, 0, sizeof(float) * 16);
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}

void mat4_perspective(float m[16], float fovy, float aspect, float near, float far) {
    float f = 1.0f / tanf(fovy / 2.0f);
    memset(m, 0, sizeof(float) * 16);
    m[0] = f / aspect;
    m[5] = f;
    m[10] = -(far + near) / (near - far);
    m[14] = -1.0f;
    m[11] = -(2.0f * far * near) / (near - far);
}

void mat4_rotate_xyz(float m[16], float rx, float ry, float rz) {
    float cx = cosf(rx), sx = sinf(rx);
    float cy = cosf(ry), sy = sinf(ry);
    float cz = cosf(rz), sz = sinf(rz);

    mat4_identity(m);
    m[0] = cy * cz;
    m[4] = cx * sz + sx * sy * cz;
    m[8] = sx * sz - cx * sy * cz;
    m[1] = -cy * sz;
    m[5] = cx * cz - sx * sy * sz;
    m[9] = sx * cz + cx * sy * sz;
    m[2] = sy;
    m[6] = -sx * cy;
    m[10] = cx * cy;
}

void mat4_translate(float m[16], float x, float y, float z) {
    mat4_identity(m);
    m[3] = x;
    m[7] = y;
    m[11] = z;
}

void mat4_scale(float m[16], float sx, float sy, float sz) {
    mat4_identity(m);
    m[0] = sx;
    m[5] = sy;
    m[10] = sz;
}

void mat4_multiply(float out[16], const float a[16], const float b[16]) {
    float temp[16];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            temp[i + j * 4] = 0.0f;
            for (int k = 0; k < 4; k++) {
                temp[i + j * 4] += a[i + k * 4] * b[k + j * 4];
            }
        }
    }
    memcpy(out, temp, sizeof(float) * 16);
}

void mat3_from_mat4(float m3[9], const float m4[16]) {
    m3[0] = m4[0];  m3[3] = m4[1];  m3[6] = m4[2];
    m3[1] = m4[4];  m3[4] = m4[5];  m3[7] = m4[6];
    m3[2] = m4[8];  m3[5] = m4[9];  m3[8] = m4[10];
}
