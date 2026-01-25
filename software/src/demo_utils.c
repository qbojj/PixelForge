#include <string.h>
#include "demo_utils.h"

static int32_t fp16_16(float v) {
    return (int32_t)(v * 65536.0f);
}

/* Create cube with colorful faces */
void demo_create_cube(struct demo_vertex *vertices, uint16_t *indices, uint32_t *idx_count) {
    // Cube vertex positions, normals, and colors
    float vtx_pos[24][3] = {
        // Front face (+Z)
        {-0.5f, -0.5f,  0.5f}, {0.5f, -0.5f,  0.5f}, {0.5f,  0.5f,  0.5f}, {-0.5f,  0.5f,  0.5f},
        // Back face (-Z)
        { 0.5f, -0.5f, -0.5f}, {-0.5f, -0.5f, -0.5f}, {-0.5f,  0.5f, -0.5f}, { 0.5f,  0.5f, -0.5f},
        // Left face (-X)
        {-0.5f, -0.5f, -0.5f}, {-0.5f, -0.5f,  0.5f}, {-0.5f,  0.5f,  0.5f}, {-0.5f,  0.5f, -0.5f},
        // Right face (+X)
        { 0.5f, -0.5f,  0.5f}, { 0.5f, -0.5f, -0.5f}, { 0.5f,  0.5f, -0.5f}, { 0.5f,  0.5f,  0.5f},
        // Top face (+Y)
        {-0.5f,  0.5f,  0.5f}, { 0.5f,  0.5f,  0.5f}, { 0.5f,  0.5f, -0.5f}, {-0.5f,  0.5f, -0.5f},
        // Bottom face (-Y)
        {-0.5f, -0.5f, -0.5f}, { 0.5f, -0.5f, -0.5f}, { 0.5f, -0.5f,  0.5f}, {-0.5f, -0.5f,  0.5f},
    };

    float vtx_norm[24][3] = {
        // Front face (+Z)
        {0,0,1}, {0,0,1}, {0,0,1}, {0,0,1},
        // Back face (-Z)
        {0,0,-1}, {0,0,-1}, {0,0,-1}, {0,0,-1},
        // Left face (-X)
        {-1,0,0}, {-1,0,0}, {-1,0,0}, {-1,0,0},
        // Right face (+X)
        {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0},
        // Top face (+Y)
        {0,1,0}, {0,1,0}, {0,1,0}, {0,1,0},
        // Bottom face (-Y)
        {0,-1,0}, {0,-1,0}, {0,-1,0}, {0,-1,0},
    };

    float vtx_color[24][3] = {
        // Front face (red)
        {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0},
        // Back face (green)
        {0,1,0}, {0,1,0}, {0,1,0}, {0,1,0},
        // Left face (blue)
        {0,0,1}, {0,0,1}, {0,0,1}, {0,0,1},
        // Right face (yellow)
        {1,1,0}, {1,1,0}, {1,1,0}, {1,1,0},
        // Top face (cyan)
        {0,1,1}, {0,1,1}, {0,1,1}, {0,1,1},
        // Bottom face (magenta)
        {1,0,1}, {1,0,1}, {1,0,1}, {1,0,1},
    };

    // Triangle list indices (2 triangles per face)
    int idx[36] = {
        0,1,2, 0,2,3,       // Front
        4,5,6, 4,6,7,       // Back
        8,9,10, 8,10,11,    // Left
        12,13,14, 12,14,15, // Right
        16,17,18, 16,18,19, // Top
        20,21,22, 20,22,23  // Bottom
    };

    // Populate vertex buffer
    for (int i = 0; i < 24; i++) {
        vertices[i].pos[0] = fp16_16(vtx_pos[i][0]);
        vertices[i].pos[1] = fp16_16(vtx_pos[i][1]);
        vertices[i].pos[2] = fp16_16(vtx_pos[i][2]);
        vertices[i].pos[3] = fp16_16(1.0f);
        vertices[i].norm[0] = fp16_16(vtx_norm[i][0]);
        vertices[i].norm[1] = fp16_16(vtx_norm[i][1]);
        vertices[i].norm[2] = fp16_16(vtx_norm[i][2]);
        vertices[i].col[0] = fp16_16(vtx_color[i][0]);
        vertices[i].col[1] = fp16_16(vtx_color[i][1]);
        vertices[i].col[2] = fp16_16(vtx_color[i][2]);
        vertices[i].col[3] = fp16_16(1.0f);
    }

    // Populate index buffer
    for (int i = 0; i < 36; i++) {
        indices[i] = (uint16_t)idx[i];
    }

    *idx_count = 36;
}

/* Column-major (OpenGL-style) implementations */

void mat4_identity(float m[16]) {
    memset(m, 0, sizeof(float) * 16);
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}

static void mat3_identity(float m[9]) {
    memset(m, 0, sizeof(float) * 9);
    m[0] = m[4] = m[8] = 1.0f;
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

static void mat4_cast_to_mat3(float out[9], const float m[16]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            out[i * 3 + j] = m[i * 4 + j];
        }
    }
}

static void mat4_transpose(float out[16], const float m[16]) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            out[i * 4 + j] = m[j * 4 + i];
        }
    }
}

static float mat3_det(const float m[9]) {
    return m[0] * (m[4] * m[8] - m[5] * m[7]) -
           m[1] * (m[3] * m[8] - m[5] * m[6]) +
           m[2] * (m[3] * m[7] - m[4] * m[6]);
}

void mat3_from_mat4(float m3[9], const float m4[16]) {
    float m4_t[16];
    mat4_transpose(m4_t, m4);

    float minv[9];
    mat4_cast_to_mat3(minv, m4_t);

    float det = mat3_det(minv);
    if (fabsf(det) < 1e-6f) {
        // Singular matrix, return identity
        mat3_identity(m3);
        return;
    }

    float invdet = 1.0f / det;
    m3[0] = (minv[4] * minv[8] - minv[5] * minv[7]) * invdet;
    m3[1] = (minv[2] * minv[7] - minv[1] * minv[8]) * invdet;
    m3[2] = (minv[1] * minv[5] - minv[2] * minv[4]) * invdet;
    m3[3] = (minv[5] * minv[6] - minv[3] * minv[8]) * invdet;
    m3[4] = (minv[0] * minv[8] - minv[2] * minv[6]) * invdet;
    m3[5] = (minv[2] * minv[3] - minv[0] * minv[5]) * invdet;
    m3[6] = (minv[3] * minv[7] - minv[4] * minv[6]) * invdet;
    m3[7] = (minv[1] * minv[6] - minv[0] * minv[7]) * invdet;
    m3[8] = (minv[0] * minv[4] - minv[1] * minv[3]) * invdet;
}

void mat4_to_fp16_16(int32_t out[16], const float in[16]) {
    for (int i = 0; i < 16; i++) {
        out[i] = fp16_16(in[i]);
    }
}

void mat3_to_fp16_16(int32_t out[9], const float in[9]) {
    for (int i = 0; i < 9; i++) {
        out[i] = fp16_16(in[i]);
    }
}
