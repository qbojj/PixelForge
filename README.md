# PixelForge

**Fixed-pipeline graphics accelerator based on FPGA**

A hardware implementation of a subset of [OpenGL ES 1.1 Common-Lite](https://registry.khronos.org/OpenGL/specs/es/1.1/es_full_spec_1.1.pdf) specification on Intel Cyclone V FPGA using Amaranth HDL.

## 🎓 Engineering Thesis

**Title**: Fixed-Pipeline Graphics Accelerator Based on FPGA
**Author**: Jakub Janeczko
**Supervisor**: dr Marek Materzok
**Institution**: Institute of Computer Science, University of Wrocław
**Year**: 2025/2026

## 📚 Documentation

- **[Thesis (Polish)](thesis/thesis_new.tex)** - Full bachelor's thesis in LaTeX
- **[Architecture Documentation](ARCHITECTURE.md)** - Detailed technical architecture
- **[Demo Applications](software/README.md)** - Documentation of demo programs

## ⚡ Features

- Complete 3D graphics pipeline from vertex transform to fragment output
- Phong lighting model (ambient, diffuse)
- Triangle rasterization with perspective-correct interpolation
- Depth & stencil buffering
- Alpha blending
- Configurable topologies (Triangle List, Strip, Fan)
- Fixed-point arithmetic optimized for DE1-SoC's DSP blocks (Q13.13 / Q1.17 / UQ0.9)
- SoC integration via Wishbone bus and CSR interface

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
pytest
```

### Build for FPGA

- Elaborate the Amaranth HDL design:

```bash
python -m gpu.pipeline
```

  - That will regenerate `graphics_pipeline_avalon_csr.sv` and `graphics_pipeline_csr_map.json`.

- Regenerate the CSR mapping with:

```bash
python tools/gen_csr_header.py \
    --json graphics_pipeline_csr_map.json \
    --out software/include/graphics_pipeline_csr.h
```

- Open `quartus/soc_system.qpf` in Intel Quartus.
- Open `quartus/soc_system.qsys` and regenerate the system.
- Compile the project. (This will create .sof file)
- Convert .sof to .rbf:

```bash
quartus_cpf -c quartus/output_files/soc_system.sof quartus/output_files/soc_system.rbf
```

- Upload the `soc_system.rbf` to the root of the first partition of the SD card. (see [INSTALLATION.md](INSTALLATION.md) for details)

- Regenerate the memory map header for software:

```bash
sopcinfo2swinfo --input=quartus/soc_system.sopcinfo --output=quartus/soc_system.swinfo
swinfo2header --swinfo quartus/soc_system.swinfo --single software/include/soc_system.h --module 'hps_arm_a9_0'
```

### Build Demo Applications

```bash
cd software
export CROSS_COMPILE=arm-linux-gnueabihf-
make
```

then you can upload the binaries to the DE1-SoC board.
```bash
cd software
sudo make install DESTDIR=/path/to/sdcard/home/root/
```

### Run Demos

See [software/DEMOS.md](software/DEMOS.md) for detailed instructions on running the demo applications.

## 📊 Resource Usage (Cyclone V)

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| ALMs | 28,702 | 32,070 | 89% |
| DSP Blocks | 67 | 87 | 77% |
| Memory Bits | 552,407 | 4,065,280 | 14% |

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

81 unit and integration tests covering all major modules:
- Input Assembly & Topology Processing
- Vertex Transformations
- Vertex Shading & Lighting
- Rasterization Pipeline
- Depth/Stencil Tests
- Blending Operations
- Full pipeline integration tests

Visual verification via PPM image generation.

## 🎮 Demo Applications

- **demo_cube** - Basic rotating cube
- **demo_depth** - Three cubes at different depths demonstrating depth buffering
- **demo_obj** - Wavefront OBJ model viewer with stencil outline effect
- **pixelforge_demo** - Minimal example rendering simple triangles and test patterns

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- dr Marek Materzok - academic supervision
- Amaranth HDL team - excellent HDL tools
- Khronos Group - OpenGL ES specification

## 📧 Contact

- **Author**: Jakub Janeczko
- **GitHub**: [@qbojj](https://github.com/qbojj)

## 📝 Citation

```bibtex
@mastersthesis{pixelforge2026,
  author = {Jakub Janeczko},
  title = {Fixed-Pipeline Graphics Accelerator Based on FPGA},
  school = {University of Wrocław, Institute of Computer Science},
  year = {2026},
  type = {Engineering thesis},
  supervisor = {dr Marek Materzok}
}
```

---

**PixelForge** © 2025-2026 Jakub Janeczko
