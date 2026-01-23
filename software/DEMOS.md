# PixelForge GPU Feature Demos

This directory contains several demo programs showcasing different features of the PixelForge GPU, along with debugging utilities for inspecting GPU and display controller state.

## Building the Demos

```bash
CROSS_COMPILE=arm-none-linux-gnueabihf-  # Or any suitable ARM linux toolchain
make
```

This will build all demos:
- `pixelforge_demo` - Basic triangle rendering (minimal example)
- `demo_cube` - Rotating colored cube
- `demo_depth` - Multiple cubes demonstrating depth buffer operations
- `demo_obj` - Wavefront OBJ model viewer with optional stencil outline effect

And debugging utilities:
- `dump_gpu_csr` - Display all PixelForge GPU control/status registers
- `dump_vga_dma` - Display VGA Pixel Buffer DMA controller registers

## Demo Descriptions

### 1. pixelforge_demo - Basic Triangle Rendering

**Features showcased:**
- Basic triangle rasterization and color interpolation
- Vertex transformations
- Minimal example of data flow through the rendering pipeline

**Usage:**
```bash
./pixelforge_demo [options]
```

**Options:**
- `--clear-test` - Fill screen with test pattern and exit
- `--xor-test` - Fill screen with XOR pattern and exit
- `--render-triangle` - Render triangle using GPU pipeline
- `--frames N` - Render N frames (default: 1)
- `--verbose` - Enable debug output
- `--throttle` - Throttle debug output with delays
- `--front` - Operate on front buffer instead of back buffer

**What it does:**
Renders a simple colored triangle demonstrating the basic graphics pipeline. By default renders a single frame with test pattern fill. Use `--render-triangle` to actually render geometry. This is the minimal example showing how vertices are transformed, rasterized, and rendered with interpolated colors.

---

### 2. demo_cube - Rotating Colored Cube

**Features showcased:**
- Triangle rasterization and color interpolation
- Rotation animation in 3D space
- Basic vertex transformations

**Usage:**
```bash
./demo_cube [--verbose] [--frames N]
```

**Options:**
- `--verbose` - Enable debug output
- `--frames N` - Render N frames (default: varies)

**What it does:**
Renders a colorful cube rotating in real-time, demonstrating basic 3D transformations and color interpolation across triangle surfaces.

---

### 3. demo_depth - Depth Buffer with Occluding Cubes

**Features showcased:**
- Depth buffer testing and writing
- Multiple objects at different Z-depths
- Proper occlusion (front objects hide back objects)
- Depth compare operations (LESS)
- Animation with moving objects

**Usage:**
```bash
./demo_depth [--verbose] [--frames N]
```

**Options:**
- `--verbose` - Enable debug output
- `--frames N` - Render N frames (default: 120)

**What it does:**
Renders three colored cubes at different depths:

**Performance note:**
Default is 120 frames. Each frame requires 3 draw calls (one per cube). Consider reducing frame count for testing.

---

### 4. demo_obj - Wavefront OBJ Model Viewer

**Features showcased:**
- Loading and rendering Wavefront OBJ files (vertices, normals, faces)
- Vertex duplication for per-face normal support
- Non-indexed rendering mode
- Rotation animation with perspective projection
- Directional diffuse lighting
- Optional stencil buffer outline effect

**Usage:**
```bash
./demo_obj [--verbose] [--frames N] [--stencil-outline] [--obj FILE] <model.obj>
```

**Options:**
- `--verbose` - Enable debug output
- `--frames N` - Number of animation frames (default: 90)
- `--stencil-outline` - Enable two-pass stencil outline rendering
- `--obj FILE` - Alternate way to specify OBJ file
- `<model.obj>` - Path to Wavefront OBJ file (e.g., `sphere.obj`, `tetrahedron.obj`)

**What it does:**
Loads a 3D model from a Wavefront OBJ file and renders it rotating under directional lighting. The demo supports models with per-vertex or per-face normals.

When `--stencil-outline` is enabled, uses a two-pass rendering technique:
- **Pass 1:** Draw the object normally with lighting and mark stencil buffer (value=1)
- **Pass 2:** Draw slightly enlarged object with solid color only where stencil≠1 (creating outline)

This creates a glowing outline effect around the object, commonly used for selection highlights and visual emphasis in games.

**Included models:**
- `sphere.obj` - Smooth sphere (482 vertices, 960 faces, per-vertex normals)
- `sphere_faceted.obj` - Faceted sphere (per-face normals for flat shading)
- `tetrahedron.obj` - Simple 4-sided polyhedron (4 vertices, 4 faces)

**Performance note:**
All provided models achieve 60FPS on PixelForge. As the pipeline is fill-rate limited we don't expect performance to drop significantly with more complex models, but extremely high polygon counts may impact frame time.

---

## Debugging Utilities

### dump_gpu_csr - GPU Register Inspector

Displays all control and status registers of the PixelForge GPU pipeline.

**Usage:**
```bash
./dump_gpu_csr
```

**What it shows:**
- **Index configuration:** Index buffer address, count, format (U8/U16/NOT_INDEXED)
- **Vertex layout:** Attribute offsets, stride, base address
- **Input topology:** Primitive type (triangles/triangle_strip/triangle_fan), restart index
- **Viewport:** X/Y/Z transformation parameters, depth range
- **Depth/stencil test:** Enable flags, compare operations, write masks
- **Stencil operations:** FAIL/ZFAIL/ZPASS actions, reference values
- **Fragment output:** Color/depth/stencil buffer addresses, formats
- **Color blending:** Blend enable, factors, equations, alpha operations
- **Primitive assembly:** Front-face mode, cull mode, polygon mode
- **Transforms:** Model-view-projection matrices (4x4 fixed-point)
- **Lighting:** Light direction, ambient/diffuse colors
- **Draw control:** Start index, primitive count, instance count

**Use cases:**
- Debugging rendering issues (wrong buffer addresses, incorrect state)
- Verifying GPU configuration before/after draw calls
- Understanding current pipeline state when demos hang
- Checking matrix values and lighting parameters

---

### dump_vga_dma - VGA DMA Controller Inspector

Displays registers of the Altera VGA Pixel Buffer DMA controller.

**Usage:**
```bash
./dump_vga_dma
```

**What it shows:**
- **Buffer addresses:** Front buffer and back buffer physical addresses
- **Resolution:** Display width and height in pixels
- **Status:** Buffer status register
- **Control:** DMA control flags

**Use cases:**
- Verifying framebuffer addresses are correctly set
- Checking resolution configuration
- Debugging display issues (no output, wrong buffer)
- Confirming double-buffering setup

---

## Common Options

All demos support:
- `--frames N` - Render N frames (default varies per demo)
- `--verbose` - Enable debug output showing GPU state and operations

`demo_obj` additionally supports:
- `--stencil-outline` - Enable outline effect using stencil buffer
- `--obj FILE` or positional argument - Specify OBJ file

## Performance Considerations

The PixelForge GPU is implemented on an FPGA and runs at a modest clock speed. Rendering is done in software-like fashion through the hardware pipeline. **Each frame can take several seconds to complete.**

**Tips for faster testing:**
- Use `--frames 10` or similar to reduce the number of frames
- Start with low polygon count objects
- The demos are already optimized for the GPU's speed (simple geometry, not too many vertices)

Reduce these if testing on slower hardware or if you want quicker results.

## Technical Details

### Geometry Complexity
- **Triangle** (pixelforge_demo): Basic example
- **Cube** (demo_depth, demo_cube): 8 vertices, 12 triangles (36 indices)
- **Sphere** (sphere.obj): 482 vertices, 960 triangles (smooth per-vertex normals)
- **Tetrahedron** (tetrahedron.obj): 4 vertices, 4 triangles (per-face normals)

**Note:** demo_obj uses non-indexed rendering with vertex duplication, so the actual vertex count sent to GPU is 3× the triangle count.

### Buffer Usage
All demos allocate:
- Front and back color buffers (640x480x4 bytes each)
- Depth/stencil buffer if required (640x480x4 bytes, D16_X8_S8 format)
- Vertex buffers (from VRAM allocator)

### Fixed-Point Format
Vertex data and matrices use formats optimized for DSP blocks:
- Q13.13 for positions, normals and matrices
- Q1.17 for barycentric coordinates, depth, normalized directions
- UQ0.9 for color/alpha channels

## Troubleshooting

**GPU doesn't finish rendering:**
- Check `/dev/mem` permissions (usually need root or `sudo`)
- Verify VRAM is properly mapped at 0x3C000000
- Check GPU CSR base at 0xFF200000
- Use `--verbose` to see where it gets stuck
- Use `dump_gpu_csr` to inspect current GPU state

**Visual artifacts:**
- Make sure depth/stencil buffer is properly cleared between frames
- Check that matrices are correct (especially projection near/far planes)
- Verify vertex normals are normalized
- Use `dump_vga_dma` to verify framebuffer addresses

**No display output:**
- Check VGA DMA configuration with `dump_vga_dma`
- Verify front/back buffer addresses are in valid VRAM range
- Ensure resolution matches display expectations (640×480)
- Reboot the FPGA board to reset VGA controller if needed

**Memory access errors:**
- Check kernel messages (`dmesg`) for memory mapping failures
- Verify physical addresses match your hardware configuration
