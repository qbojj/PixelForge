/*
 * PixelForge Demo: OpenGL ES 1.1 Wrapper Example
 *
 * This demo showcases the OpenGL ES 1.1 Common-Lite wrapper API
 * for the PixelForge GPU. It demonstrates:
 * - Context initialization
 * - Matrix stack operations
 * - Vertex arrays
 * - State management with automatic dirty tracking
 * - Draw commands with automatic state synchronization
 * - Buffer swapping with in-flight draw wait
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <math.h>

#include "gles11_wrapper.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static volatile bool keep_running = true;

static void handle_sigint(int sig) {
    (void)sig;
    keep_running = false;
}

/* Vertex structure matching PixelForge format (Q16.16 fixed point) */
typedef struct {
    int32_t pos[4];   /* Position (x, y, z, w) */
    int32_t norm[4];  /* Normal (nx, ny, nz, 0) */
    int32_t col[4];   /* Color (r, g, b, a) */
} vertex_t;

static int32_t fp16_16(float v) {
    return (int32_t)(v * 65536.0f);
}

/* Create a simple cube geometry */
static bool create_cube_geometry(GLuint vertex_buffer,
                                 GLuint index_buffer,
                                 uint32_t *vertex_count, uint32_t *index_count) {
    *vertex_count = 24;  /* 4 vertices per face * 6 faces */
    *index_count = 36;   /* 6 indices per face * 6 faces */

    size_t vb_size = *vertex_count * sizeof(vertex_t);
    size_t ib_size = *index_count * sizeof(uint16_t);

    vertex_t *vertices = (vertex_t*)malloc(vb_size);
    uint16_t *indices = (uint16_t*)malloc(ib_size);
    if (!vertices || !indices) {
        fprintf(stderr, "Failed to allocate CPU buffers\n");
        free(vertices);
        free(indices);
        return false;
    }

    /* Define cube vertices (each face separate for proper normals) */
    float positions[24][3] = {
        /* Front face */
        {-1, -1,  1}, { 1, -1,  1}, { 1,  1,  1}, {-1,  1,  1},
        /* Back face */
        {-1, -1, -1}, {-1,  1, -1}, { 1,  1, -1}, { 1, -1, -1},
        /* Top face */
        {-1,  1, -1}, {-1,  1,  1}, { 1,  1,  1}, { 1,  1, -1},
        /* Bottom face */
        {-1, -1, -1}, { 1, -1, -1}, { 1, -1,  1}, {-1, -1,  1},
        /* Right face */
        { 1, -1, -1}, { 1,  1, -1}, { 1,  1,  1}, { 1, -1,  1},
        /* Left face */
        {-1, -1, -1}, {-1, -1,  1}, {-1,  1,  1}, {-1,  1, -1},
    };

    float normals[6][3] = {
        { 0,  0,  1},  /* Front */
        { 0,  0, -1},  /* Back */
        { 0,  1,  0},  /* Top */
        { 0, -1,  0},  /* Bottom */
        { 1,  0,  0},  /* Right */
        {-1,  0,  0},  /* Left */
    };

    float colors[6][4] = {
        {1, 0, 0, 1},  /* Red */
        {0, 1, 0, 1},  /* Green */
        {0, 0, 1, 1},  /* Blue */
        {1, 1, 0, 1},  /* Yellow */
        {1, 0, 1, 1},  /* Magenta */
        {0, 1, 1, 1},  /* Cyan */
    };

    /* Fill vertex buffer */
    for (int face = 0; face < 6; face++) {
        for (int vert = 0; vert < 4; vert++) {
            int idx = face * 4 + vert;

            vertices[idx].pos[0] = fp16_16(positions[idx][0]);
            vertices[idx].pos[1] = fp16_16(positions[idx][1]);
            vertices[idx].pos[2] = fp16_16(positions[idx][2]);
            vertices[idx].pos[3] = fp16_16(1.0f);

            vertices[idx].norm[0] = fp16_16(normals[face][0]);
            vertices[idx].norm[1] = fp16_16(normals[face][1]);
            vertices[idx].norm[2] = fp16_16(normals[face][2]);
            vertices[idx].norm[3] = fp16_16(0.0f);

            vertices[idx].col[0] = fp16_16(colors[face][0]);
            vertices[idx].col[1] = fp16_16(colors[face][1]);
            vertices[idx].col[2] = fp16_16(colors[face][2]);
            vertices[idx].col[3] = fp16_16(colors[face][3]);
        }
    }

    /* Fill index buffer */
    for (int face = 0; face < 6; face++) {
        int base = face * 4;
        int idx = face * 6;

        indices[idx + 0] = base + 0;
        indices[idx + 1] = base + 1;
        indices[idx + 2] = base + 2;
        indices[idx + 3] = base + 0;
        indices[idx + 4] = base + 2;
        indices[idx + 5] = base + 3;
    }

    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, vb_size, vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, ib_size, indices, GL_STATIC_DRAW);

    free(vertices);
    free(indices);

    return true;
}

int main(int argc, char **argv) {
    int frames = 90;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--frames") && i + 1 < argc) {
            frames = atoi(argv[++i]);
        }
    }

    signal(SIGINT, handle_sigint);

    /* Initialize OpenGL ES context */
    if (!glInit()) {
        fprintf(stderr, "Failed to initialize OpenGL ES context\n");
        return 1;
    }

    printf("PixelForge OpenGL ES 1.1 Demo: Rotating Cube\n");
    printf("Rendering %d frames...\n", frames);

    /* Create cube geometry */
    GLuint vertex_buffer = 0;
    GLuint index_buffer = 0;
    uint32_t vertex_count, index_count;

    glGenBuffers(1, &vertex_buffer);
    glGenBuffers(1, &index_buffer);

    if (vertex_buffer == 0 || index_buffer == 0 ||
        !create_cube_geometry(vertex_buffer, index_buffer, &vertex_count, &index_count)) {
        glDestroy();
        return 1;
    }
    (void)vertex_count;

    /* Setup OpenGL state */
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);

    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

    /* Setup lighting */
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    GLfloat light_pos[] = {1.0f, 1.0f, 1.0f, 0.0f};
    GLfloat light_ambient[] = {0.2f, 0.2f, 0.2f};
    GLfloat light_diffuse[] = {1.0f, 1.0f, 1.0f};

    glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);

    GLfloat mat_ambient[] = {1.0f, 1.0f, 1.0f, 1.0f};
    GLfloat mat_diffuse[] = {1.0f, 1.0f, 1.0f, 1.0f};

    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, mat_ambient);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);

    /* Setup vertex arrays */
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glVertexPointer(4, GL_FIXED, sizeof(vertex_t), (void*)offsetof(vertex_t, pos));
    glNormalPointer(GL_FIXED, sizeof(vertex_t), (void*)offsetof(vertex_t, norm));
    glColorPointer(4, GL_FIXED, sizeof(vertex_t), (void*)offsetof(vertex_t, col));

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);

    /* Setup projection matrix */
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    /* Create perspective projection manually */
    float fovy = 45.0f * M_PI / 180.0f;
    float aspect = 640.0f / 480.0f;  /* Assuming default resolution */
    float near = 0.5f;
    float far = 5.0f;

    float f = 1.0f / tanf(fovy / 2.0f);
    glFrustumf(-near * aspect / f, near * aspect / f, -near / f, near / f, near, far);

    /* Animation loop */
    for (int frame = 0; frame < frames && keep_running; frame++) {
        float t = (float)frame / 30.0f;

        /* Clear buffers */
        /* NOTE: clearing depth and stencil at the same time is more efficient than leaving one uncleared */
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

        /* Setup modelview matrix */
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        /* Camera */
        glTranslatef(0.0f, 0.0f, -4.0f);

        /* Rotation */
        glRotatef(t * 40.0f, 0.7f, 1.0f, 0.5f);

        /* Draw the cube
         * State synchronization happens automatically:
         * 1. Checks dirty flags
         * 2. Waits for previous draw to complete
         * 3. Uploads only changed state
         * 4. Issues draw command
         */
        glDrawElements(GL_TRIANGLES, index_count, GL_UNSIGNED_SHORT, (void*)0);

        /* Swap buffers
         * This automatically waits for the draw to complete
         */
        glSwapBuffers();

        if (frame % 30 == 0) {
            printf("Rendered frame %d/%d\n", frame, frames);
        }
    }

    /* Cleanup */
    glDeleteBuffers(1, &vertex_buffer);
    glDeleteBuffers(1, &index_buffer);
    glDestroy();

    printf("Done!\n");
    return 0;
}
