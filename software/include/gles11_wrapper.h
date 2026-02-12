/*
 * OpenGL ES 1.1 Common-Lite Wrapper for PixelForge GPU
 *
 * This header provides an OpenGL ES 1.1 Common-Lite compatible API
 * for the PixelForge GPU architecture.
 */

#ifndef GLES11_WRAPPER_H
#define GLES11_WRAPPER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * OpenGL ES 1.1 Common-Lite Types
 * ============================================================================ */

typedef uint32_t GLenum;
typedef uint32_t GLbitfield;
typedef uint32_t GLuint;
typedef int32_t GLint;
typedef int32_t GLsizei;
typedef int8_t GLbyte;
typedef int16_t GLshort;
typedef uint8_t GLubyte;
typedef uint16_t GLushort;
typedef float GLfloat;
typedef float GLclampf;
typedef void GLvoid;
typedef int32_t GLfixed;
typedef int32_t GLclampx;

/* =========================================================================
 * OpenGL ES 1.1 Common-Lite Constants
 * ============================================================================ */

/* Matrix modes */
#define GL_MODELVIEW                      0x1700
#define GL_PROJECTION                     0x1701
#define GL_TEXTURE                        0x1702

/* Primitives */
#define GL_POINTS                         0x0000
#define GL_LINES                          0x0001
#define GL_LINE_STRIP                     0x0003
#define GL_TRIANGLES                      0x0004
#define GL_TRIANGLE_STRIP                 0x0005
#define GL_TRIANGLE_FAN                   0x0006

/* Depth buffer */
#define GL_NEVER                          0x0200
#define GL_LESS                           0x0201
#define GL_EQUAL                          0x0202
#define GL_LEQUAL                         0x0203
#define GL_GREATER                        0x0204
#define GL_NOTEQUAL                       0x0205
#define GL_GEQUAL                         0x0206
#define GL_ALWAYS                         0x0207
#define GL_DEPTH_TEST                     0x0B71
#define GL_DEPTH_WRITEMASK                0x0B72
#define GL_DEPTH_FUNC                     0x0B74

/* Blending */
#define GL_BLEND                          0x0BE2
#define GL_SRC_ALPHA                      0x0302
#define GL_ONE_MINUS_SRC_ALPHA            0x0303
#define GL_SRC_COLOR                      0x0300
#define GL_ONE_MINUS_SRC_COLOR            0x0301
#define GL_DST_COLOR                      0x0306
#define GL_ONE_MINUS_DST_COLOR            0x0307
#define GL_DST_ALPHA                      0x0304
#define GL_ONE_MINUS_DST_ALPHA            0x0305
#define GL_ZERO                           0
#define GL_ONE                            1

/* Culling */
#define GL_CULL_FACE                      0x0B44
#define GL_FRONT                          0x0404
#define GL_BACK                           0x0405
#define GL_FRONT_AND_BACK                 0x0408
#define GL_CW                             0x0900
#define GL_CCW                            0x0901

/* Stencil */
#define GL_STENCIL_TEST                   0x0B90
#define GL_KEEP                           0x1E00
#define GL_REPLACE                        0x1E01
#define GL_INCR                           0x1E02
#define GL_DECR                           0x1E03
#define GL_INVERT                         0x150A
#define GL_INCR_WRAP                      0x8507
#define GL_DECR_WRAP                      0x8508

/* Lighting */
#define GL_LIGHTING                       0x0B50
#define GL_LIGHT0                         0x4000
#define GL_AMBIENT                        0x1200
#define GL_DIFFUSE                        0x1201
#define GL_SPECULAR                       0x1202
#define GL_POSITION                       0x1203
#define GL_EMISSION                       0x1600
#define GL_SHININESS                      0x1601

/* Buffer bits */
#define GL_COLOR_BUFFER_BIT               0x00004000
#define GL_DEPTH_BUFFER_BIT               0x00000100
#define GL_STENCIL_BUFFER_BIT             0x00000400

/* Vertex arrays */
#define GL_VERTEX_ARRAY                   0x8074
#define GL_NORMAL_ARRAY                   0x8075
#define GL_COLOR_ARRAY                    0x8076
#define GL_TEXTURE_COORD_ARRAY            0x8078

/* Data types */
#define GL_BYTE                           0x1400
#define GL_UNSIGNED_BYTE                  0x1401
#define GL_SHORT                          0x1402
#define GL_UNSIGNED_SHORT                 0x1403
#define GL_FLOAT                          0x1406
#define GL_FIXED                          0x140C

/* Buffer targets/usage */
#define GL_ARRAY_BUFFER                   0x8892
#define GL_ELEMENT_ARRAY_BUFFER           0x8893
#define GL_STATIC_DRAW                    0x88E4
#define GL_DYNAMIC_DRAW                   0x88E8

/* Boolean */
#define GL_FALSE                          0
#define GL_TRUE                           1

/* =========================================================================
 * PixelForge Buffer Objects (Opaque Handles)
 * ============================================================================ */

void glGenBuffers(GLsizei n, GLuint *buffers);
void glDeleteBuffers(GLsizei n, const GLuint *buffers);
void glBindBuffer(GLenum target, GLuint buffer);
void glBufferData(GLenum target, size_t size, const void *data, GLenum usage);
void glBufferSubData(GLenum target, size_t offset, size_t size, const void *data);

/* ============================================================================
 * Context Management
 * ============================================================================ */

/* Initialize the OpenGL ES context with a PixelForge device
 * Must be called before any other GL functions */
bool glInit(void);

/* Cleanup and destroy the OpenGL ES context */
void glDestroy(void);

/* ============================================================================
 * State Management
 * ============================================================================ */

void glEnable(GLenum cap);
void glDisable(GLenum cap);
void glClearColor(GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha);
void glClearDepthf(GLclampf depth);
void glClearStencil(GLint s);
void glClear(GLbitfield mask);
void glViewport(GLint x, GLint y, GLsizei width, GLsizei height);
void glScissor(GLint x, GLint y, GLsizei width, GLsizei height);

/* ============================================================================
 * Depth Testing
 * ============================================================================ */

void glDepthFunc(GLenum func);
void glDepthMask(bool flag);

/* ============================================================================
 * Blending
 * ============================================================================ */

void glBlendFunc(GLenum sfactor, GLenum dfactor);

/* ============================================================================
 * Culling
 * ============================================================================ */

void glCullFace(GLenum mode);
void glFrontFace(GLenum mode);

/* ============================================================================
 * Stencil
 * ============================================================================ */

void glStencilFunc(GLenum func, GLint ref, GLuint mask);
void glStencilOp(GLenum fail, GLenum zfail, GLenum zpass);
void glStencilMask(GLuint mask);

/* ============================================================================
 * Matrix Operations
 * ============================================================================ */

void glMatrixMode(GLenum mode);
void glLoadIdentity(void);
void glLoadMatrixf(const GLfloat *m);
void glMultMatrixf(const GLfloat *m);
void glPushMatrix(void);
void glPopMatrix(void);

/* Transformation functions */
void glTranslatef(GLfloat x, GLfloat y, GLfloat z);
void glRotatef(GLfloat angle, GLfloat x, GLfloat y, GLfloat z);
void glScalef(GLfloat x, GLfloat y, GLfloat z);
void glFrustumf(GLfloat left, GLfloat right, GLfloat bottom, GLfloat top, GLfloat near, GLfloat far);
void glOrthof(GLfloat left, GLfloat right, GLfloat bottom, GLfloat top, GLfloat near, GLfloat far);

/* ============================================================================
 * Lighting
 * ============================================================================ */

void glLightfv(GLenum light, GLenum pname, const GLfloat *params);
void glMaterialfv(GLenum face, GLenum pname, const GLfloat *params);

/* ============================================================================
 * Vertex Arrays
 * ============================================================================ */

void glEnableClientState(GLenum array);
void glDisableClientState(GLenum array);
void glVertexPointer(GLint size, GLenum type, GLsizei stride, const GLvoid *pointer);
void glNormalPointer(GLenum type, GLsizei stride, const GLvoid *pointer);
void glColorPointer(GLint size, GLenum type, GLsizei stride, const GLvoid *pointer);

/* ============================================================================
 * Drawing Commands
 * ============================================================================ */

void glDrawArrays(GLenum mode, GLint first, GLsizei count);
void glDrawElements(GLenum mode, GLsizei count, GLenum type, const GLvoid *indices);

/* ============================================================================
 * Buffer Swap
 * ============================================================================ */

/* Swap front and back buffers (implements the swap wait requirement) */
void glSwapBuffers(void);


#ifdef __cplusplus
}
#endif

#endif /* GLES11_WRAPPER_H */
