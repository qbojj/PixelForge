# PixelForge

**Fixed-pipeline graphics accelerator based on FPGA**

A hardware implementation of a subset of [OpenGL ES 1.1 Common-Lite](https://registry.khronos.org/OpenGL/specs/es/1.1/es_full_spec_1.1.pdf) specification on Intel Cyclone V FPGA using Amaranth HDL.

## 🎓 Bachelor's Thesis

**Title**: Fixed-Pipeline Graphics Accelerator Based on FPGA
**Author**: Jakub Janeczko
**Supervisor**: dr Marek Materzok
**Institution**: Institute of Computer Science, University of Wrocław
**Year**: 2025/2026

## 📚 Documentation

- **[Thesis (Polish)](thesis/thesis_new.tex)** - Full bachelor's thesis in LaTeX
- **[Architecture Documentation](ARCHITECTURE.md)** - Detailed technical architecture
- **[Polish README](README_PL.md)** - Complete guide in Polish
- **[Demo Applications](software/DEMOS.md)** - Documentation of demo programs
- **[Summary](PODSUMOWANIE.md)** - Project summary in Polish

## ⚡ Features

- ✨ Complete 3D graphics pipeline from vertex transform to fragment output
- 🎨 Phong lighting model (ambient, diffuse, emissive - up to 8 lights)
- 🔺 Triangle rasterization with perspective-correct interpolation
- 📊 Depth & stencil buffering
- 🎭 Alpha blending
- 🔧 Configurable topologies (Triangle List, Strip, Fan)
- 🚀 Fixed-point arithmetic dopasowana do bloków DSP (Q13.13 / Q1.17 / UQ0.9)
- 🔌 SoC integration via Wishbone bus and CSR interface

## 🏗️ Pipeline Architecture

```
Index Generation → Input Topology → Input Assembly
    ↓
Vertex Transform → Vertex Shading → Primitive Assembly
    ↓
Primitive Clipping → Perspective Divide → Triangle Prep
    ↓
Triangle Rasterization → Depth/Stencil Test
    ↓
Blending → Framebuffer Output
```

## 🚀 Quick Start

### Requirements
- Python 3.10+
- Amaranth HDL
- Intel Quartus Prime (for FPGA synthesis)
- pytest (for testing)

### Installation

```bash
git clone https://github.com/qbojj/PixelForge.git
cd PixelForge
pip install -e ".[dev]"
```

### Run Tests

```bash
# All tests
pytest tests/

# Parallel execution
pytest -n auto tests/

# Specific module
pytest tests/rasterizer/
```

### Build for FPGA

```bash
cd quartus
make              # Full build (synthesis, fit, asm)
make program      # Program FPGA
```

### Build Demo Applications

```bash
cd software
make              # Build all demos
./demo_lighting   # Run lighting demo
```

## 📊 Resource Usage (Cyclone V)

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| ALMs | 18,542 | 32,070 | 57.8% |
| Registers | 35,821 | 128,280 | 27.9% |
| Block Memory | 89 | 397 | 22.4% |
| DSP Blocks | 67 | 87 | 77.0% |

**Clock Frequency**: 50 MHz

## 📖 Project Structure

```
PixelForge/
├── gpu/                    # Main HDL source (Amaranth)
│   ├── input_assembly/     # Vertex fetch and formatting
│   ├── vertex_transform/   # Geometric transformations
│   ├── vertex_shading/     # Lighting system
│   ├── rasterizer/        # Triangle rasterization
│   ├── pixel_shading/     # Per-fragment operations
│   └── pipeline.py        # Top-level integration
├── tests/                 # Unit and integration tests
├── quartus/              # Intel Quartus Prime project
├── software/             # Demo applications (C)
└── thesis/               # Bachelor's thesis (LaTeX)
```

## 🧪 Testing

91 unit tests covering all major modules:
- Input Assembly & Topology Processing
- Vertex Transformations
- Vertex Shading & Lighting
- Rasterization Pipeline
- Depth/Stencil Tests
- Blending Operations

Visual verification via PPM image generation.

## 🎮 Demo Applications

- **demo_lighting** - Rotating icosahedron with directional lighting
- **demo_cube** - Basic rotating cube
- **demo_depth** - Three cubes at different depths demonstrating depth buffering
- **demo_stencil** - Outline/glow effect using stencil buffer

## 📄 License

[To be determined - e.g., MIT, BSD, GPL]

## 🙏 Acknowledgments

- dr Marek Materzok - academic supervision
- Amaranth HDL team - excellent HDL tools
- Khronos Group - OpenGL ES specification

## 📧 Contact

- **Author**: Jakub Janeczko
- **GitHub**: [@qbojj](https://github.com/qbojj)

## 📝 Citation

```bibtex
@mastersthesis{janeczko2026pixelforge,
  author = {Jakub Janeczko},
  title = {Fixed-Pipeline Graphics Accelerator Based on FPGA},
  school = {University of Wrocław, Institute of Computer Science},
  year = {2026},
  type = {Bachelor's thesis},
  supervisor = {dr Marek Materzok}
}
```

---

**PixelForge** © 2025-2026 Jakub Janeczko
