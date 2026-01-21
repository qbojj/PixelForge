# PixelForge GPU Feature Demos

This directory contains several demo programs showcasing different features of the PixelForge GPU.

## Building the Demos

```bash
make
```

This will build all demos:
- `pixelforge_demo` - Original demo with basic triangle rendering
- `demo_lighting` - Rotating object with directional diffuse lighting
- `demo_depth` - Multiple cubes demonstrating depth buffer operations
- `demo_stencil` - Object outline/glow effect using stencil buffer

## Demo Descriptions

### 1. demo_lighting - Rotating Object with Lighting

**Features showcased:**
- Vertex transformations (model-view-projection matrices)
- Rotation animation
- Directional diffuse lighting (Phong shading model)
- Normal vector transformation
- 3D icosahedron geometry

**Usage:**
```bash
./demo_lighting [--frames N] [--verbose]
```

**What it does:**
Renders a rotating icosahedron (20-sided polyhedron) with directional diffuse lighting. The object rotates continuously, showing how normals are properly transformed and used for lighting calculations. The light is positioned to create realistic shading on the surface.

**Performance note:**
Default is 60 frames. Each frame takes time to render on the FPGA GPU, so expect this to run slowly. Reduce frame count for quicker testing.

---

### 2. demo_depth - Depth Buffer with Occluding Cubes

**Features showcased:**
- Depth buffer testing and writing
- Multiple objects at different Z-depths
- Proper occlusion (front objects hide back objects)
- Depth compare operations (LESS)
- Animation with moving objects

**Usage:**
```bash
./demo_depth [--frames N] [--verbose]
```

**What it does:**
Renders three colored cubes at different depths:
- **Red cube** (back): Rotates slowly in the background at Z=-3.0
- **Green cube** (middle): Oscillates left-right at Z=-2.0
- **Blue cube** (front): Rotates and moves up-down at Z=-1.2

The depth buffer ensures that closer objects properly occlude farther ones. As the blue cube moves, you can see it hide parts of the green and red cubes behind it.

**Performance note:**
Default is 120 frames. Each frame requires 3 draw calls (one per cube). Consider reducing frame count for testing.

---

### 3. demo_stencil - Object Outline/Glow Effect

**Features showcased:**
- Stencil buffer operations (write, test, masking)
- Two-pass rendering technique
- Stencil compare operations (ALWAYS, NOT_EQUAL)
- Stencil operations (REPLACE, KEEP)
- Creative visual effects using stencil buffer

**Usage:**
```bash
./demo_stencil [--frames N] [--verbose]
```

**What it does:**
Creates a glowing outline effect around a rotating octahedron using a two-pass rendering technique:

**Pass 1:** Draw the object normally (orange color) and mark the stencil buffer with value 1 wherever the object is drawn.

**Pass 2:** Draw a slightly enlarged version of the same object (yellow/orange glow color) but only where the stencil value is NOT 1. This creates a border/glow effect around the original object.

This technique is commonly used in games for:
- Selection highlights
- Outline effects
- Object silhouettes
- Glow effects

**Performance note:**
Default is 90 frames. Each frame requires 2 draw calls (base object + outline). The effect is best seen when the object is rotating.

---

## Common Options

All demos support:
- `--frames N` - Render N frames (default varies per demo)
- `--verbose` - Enable debug output showing GPU state and operations

## Performance Considerations

The PixelForge GPU is implemented on an FPGA and runs at a modest clock speed. Rendering is done in software-like fashion through the hardware pipeline. **Each frame can take several seconds to complete.**

**Tips for faster testing:**
- Use `--frames 10` or similar to reduce the number of frames
- Start with low polygon count objects
- The demos are already optimized for the GPU's speed (simple geometry, not too many vertices)

**Frame counts chosen for each demo:**
- `demo_lighting`: 60 frames (one full rotation at ~2 FPS if each frame takes 0.5s)
- `demo_depth`: 120 frames (longer animation showing occlusion clearly)
- `demo_stencil`: 90 frames (enough to see the outline effect from multiple angles)

Reduce these if testing on slower hardware or if you want quicker results.

## Technical Details

### Geometry Complexity
- **Icosahedron** (demo_lighting): 12 vertices, 20 triangles (60 indices)
- **Cube** (demo_depth): 8 vertices, 12 triangles (36 indices), 3 cubes per frame
- **Octahedron** (demo_stencil): 6 vertices, 8 triangles (24 indices), drawn twice per frame

### Buffer Usage
All demos allocate:
- Front and back color buffers (640x480x4 bytes each)
- Depth/stencil buffer (640x480x4 bytes, D16_X8_S8 format)
- Vertex buffers (from VRAM allocator)

### Fixed-Point Format
All vertex data, matrices, and colors use 16.16 fixed-point format (SQ(16,16)) to match the GPU's internal representation.

## Troubleshooting

**GPU doesn't finish rendering:**
- Check `/dev/mem` permissions (usually need root)
- Verify VRAM is properly mapped at 0x3C000000
- Check GPU CSR base at 0xFF200000
- Use `--verbose` to see where it gets stuck

**Visual artifacts:**
- Make sure depth/stencil buffer is properly cleared between frames
- Check that matrices are correct (especially projection near/far planes)
- Verify vertex normals are normalized

**Performance is very slow:**
- This is expected! The GPU is running on an FPGA at modest clock speeds
- Reduce frame count with `--frames` option
- Simplify geometry if needed (though these demos are already quite simple)

## License

Same as the main PixelForge project.
