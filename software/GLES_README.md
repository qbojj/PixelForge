# OpenGL ES 1.1 Common-Lite Wrapper for PixelForge

This directory contains an OpenGL ES 1.1 Common-Lite compatible wrapper around the PixelForge GPU architecture.

## Overview

The wrapper provides a familiar OpenGL ES 1.1 API while implementing efficient state tracking and synchronization with the underlying PixelForge hardware.

### Key Features

1. **Global State Management**
   - Full OpenGL ES 1.1 state tracking (matrices, materials, lights, depth, blend, stencil, culling)
   - Matrix stacks for modelview, projection, and texture matrices
   - Material and lighting properties

2. **Dirty Flag State Tracking**
   - Each state category has a dirty flag
   - Only changed state is uploaded to GPU
   - Minimizes CSR writes and bus traffic

3. **Automatic State Synchronization**
   - On draw commands: checks dirty flags, waits for GPU readiness, uploads changed data
   - Synchronizes with appropriate pipeline stages
   - Prevents race conditions between CPU and GPU

4. **Buffer Swap with Wait**
   - `glSwapBuffers()` waits for in-flight draws before swapping
   - Uses existing `pixelforge_swap_buffers()` implementation
   - Ensures rendering is complete before display

## API

### Context Management

```c
bool glInit(void);           // Initialize OpenGL ES context
void glDestroy(void);        // Cleanup and destroy context
```

### State Management

```c
void glEnable(GLenum cap);
void glDisable(GLenum cap);
void glClearColor(GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha);
void glClear(GLbitfield mask);
void glViewport(GLint x, GLint y, GLsizei width, GLsizei height);

// Clear controls
void glClearDepthf(GLclampf depth);
void glClearStencil(GLint s);

// Depth testing
void glDepthFunc(GLenum func);
void glDepthMask(bool flag);

// Blending
void glBlendFunc(GLenum sfactor, GLenum dfactor);

// Culling
void glCullFace(GLenum mode);
void glFrontFace(GLenum mode);

// Stencil
void glStencilFunc(GLenum func, GLint ref, GLuint mask);
void glStencilOp(GLenum fail, GLenum zfail, GLenum zpass);
void glStencilMask(GLuint mask);

// Scissor
void glScissor(GLint x, GLint y, GLsizei width, GLsizei height);
```

### Matrix Operations

```c
void glMatrixMode(GLenum mode);
void glLoadIdentity(void);
void glLoadMatrixf(const GLfloat *m);
void glMultMatrixf(const GLfloat *m);
void glPushMatrix(void);
void glPopMatrix(void);

// Transformation functions
void glTranslatef(GLfloat x, GLfloat y, GLfloat z);
void glRotatef(GLfloat angle, GLfloat x, GLfloat y, GLfloat z);
void glScalef(GLfloat x, GLfloat y, GLfloat z);
void glFrustumf(GLfloat left, GLfloat right, GLfloat bottom, GLfloat top,
                GLfloat near, GLfloat far);
void glOrthof(GLfloat left, GLfloat right, GLfloat bottom, GLfloat top,
              GLfloat near, GLfloat far);
```

### Lighting

```c
void glLightfv(GLenum light, GLenum pname, const GLfloat *params);
void glMaterialfv(GLenum face, GLenum pname, const GLfloat *params);
```

### Vertex Arrays

```c
void glEnableClientState(GLenum array);
void glDisableClientState(GLenum array);
void glVertexPointer(GLint size, GLenum type, GLsizei stride, const GLvoid *pointer);
void glNormalPointer(GLenum type, GLsizei stride, const GLvoid *pointer);
void glColorPointer(GLint size, GLenum type, GLsizei stride, const GLvoid *pointer);
```

### Drawing

```c
void glDrawArrays(GLenum mode, GLint first, GLsizei count);
void glDrawElements(GLenum mode, GLsizei count, GLenum type, const GLvoid *indices);
```

### Buffer Management

```c
void glSwapBuffers(void);            // Swap with automatic draw wait
void glGenBuffers(GLsizei n, GLuint *buffers);
void glDeleteBuffers(GLsizei n, const GLuint *buffers);
void glBindBuffer(GLenum target, GLuint buffer);
void glBufferData(GLenum target, size_t size, const void *data, GLenum usage);
void glBufferSubData(GLenum target, size_t offset, size_t size, const void *data);
```

## Implementation Details

### State Tracking Architecture

The wrapper maintains a global context (`gles_context_t`) with:

- **Dirty Flags**: Bitmask tracking which state categories have changed
  - `DIRTY_MATRICES`: Modelview/projection matrices
  - `DIRTY_VIEWPORT`: Viewport settings
  - `DIRTY_MATERIAL`: Material properties
  - `DIRTY_LIGHTS`: Light properties
  - `DIRTY_DEPTH`: Depth test settings
  - `DIRTY_BLEND`: Blend settings
  - `DIRTY_STENCIL`: Stencil test settings
  - `DIRTY_CULL`: Face culling settings
  - `DIRTY_VERTEX_ARRAYS`: Vertex array pointers
  - `DIRTY_FRAMEBUFFER`: Framebuffer configuration

- **Matrix Stacks**: Separate stacks for modelview, projection, and texture
   - Modelview: 32 levels deep
   - Projection: 2 levels deep
   - Texture: 2 levels deep

### Draw Command Flow

1. **State Change**: User calls GL state functions (e.g., `glEnable()`, `glRotatef()`)
   - State updated in context
   - Corresponding dirty flag set

2. **Draw Call**: User calls `glDrawArrays()` or `glDrawElements()`
   - Waits for stages that are required to be idle based on dirty flags
   - Uploads only changed state to GPU via CSR and clears dirty flags
   - Configures topology and vertex attributes
   - Issues `pf_csr_start()`
   - Marks draw as in-flight

3. **Buffer Swap**: User calls `glSwapBuffers()`
   - `wait_for_draw()` ensures GPU finished rendering
   - Calls `pixelforge_swap_buffers()`
   - Marks framebuffer as dirty (address changed)

### GPU Stage Synchronization

The wrapper tracks which pipeline stage needs to complete:
- **GPU_STAGE_IA**: Input Assembly
- **GPU_STAGE_VTX_TRANSFORM**: Vertex Transform
- **GPU_STAGE_PREP_RASTER**: Rasterization Prep
- **GPU_STAGE_PER_PIXEL**: Per-pixel operations

For `glSwapBuffers()` and `glClear()`, the wrapper waits for the whole pipeline to prevent WAW hazards.

### Memory Management

Vertex and index data must be in VRAM (accessible by GPU):
- Use `glGenBuffers()` + `glBindBuffer()` + `glBufferData()` to upload data
- `glVertexPointer()`/`glDrawElements()` treat pointers as offsets into bound buffers
- Use `glBufferSubData()` to update contents (waits for IA stage)
- Client code never sees raw VRAM addresses

### Clear Behavior

- `glClear()` is CPU-side and writes the color buffer directly
- Depth/stencil clears operate on a D16_X8_S8 buffer

## Example Usage

See `demo_gles.c` for a complete example. Basic pattern:

```c
// Initialize
glInit();

// Setup state
glEnable(GL_DEPTH_TEST);
glEnable(GL_CULL_FACE);

// Allocate geometry in VRAM
GLuint vb = 0;
GLuint ib = 0;
glGenBuffers(1, &vb);
glGenBuffers(1, &ib);
glBindBuffer(GL_ARRAY_BUFFER, vb);
glBufferData(GL_ARRAY_BUFFER, vb_size, vertices_init_data, GL_STATIC_DRAW);
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib);
glBufferData(GL_ELEMENT_ARRAY_BUFFER, ib_size, indices_init_data, GL_STATIC_DRAW);

// Setup vertex arrays
glEnableClientState(GL_VERTEX_ARRAY);
glBindBuffer(GL_ARRAY_BUFFER, vb);
glVertexPointer(4, GL_FIXED, sizeof(vertex_t), (void*)(uintptr_t)offsetof(vertex_t, pos));

// Setup projection
glMatrixMode(GL_PROJECTION);
glLoadIdentity();
glFrustumf(...);

// Render loop
while (running) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0, 0, -5);
    glRotatef(angle, 0, 1, 0);

      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib);
      glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_SHORT, (void*)0);
    glSwapBuffers();  // Automatic wait for draw completion
}

// Cleanup
glDeleteBuffers(1, &vb);
glDeleteBuffers(1, &ib);
glDestroy();
```

## Performance Considerations

1. **Minimize State Changes**: Group draws with similar state together
2. **Use Dirty Flags Wisely**: Wrapper only uploads changed state
3. **VRAM Allocation**: Pre-allocate geometry in VRAM, reuse buffers
4. **Matrix Operations**: Use `glPushMatrix()`/`glPopMatrix()` for efficiency

## Limitations

- Single light support (LIGHT0 only)
- No texture mapping (not supported by PixelForge hardware)
- Fixed-point vertex data format (Q16.16)
- Vertex arrays require buffer objects (no client-side arrays)
- `glTexCoordPointer()` is not implemented (texture coordinates are not used)
- `glDeleteBuffers()` does not reclaim VRAM (bump allocator)

## Files

- `include/gles11_wrapper.h`: OpenGL ES 1.1 API header
- `src/gles11_wrapper.c`: Implementation with state tracking
- `src/demo_gles.c`: Example usage demo
