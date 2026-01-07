# PixelForge Software Interface

This directory contains host-side C headers and sources for interacting with the PixelForge CSR window.

## Layout
- include/
  - graphics_pipeline_csr.h: CSR register offsets (32-bit words)
  - graphics_pipeline_formats.h: Host-side enums and structs mirroring RTL layouts
  - graphics_pipeline_csr_access.h: Accessor API (set/get) for all CSRs
- src/
  - graphics_pipeline_csr_access.c: Implementation of the accessor API

## Build
Build a static library with a simple Makefile:

```sh
cd software
make
```

This produces `libpixelforgecsr.a` you can link into your host application.

## Usage
- Include the headers from `software/include` in your application (add `-I software/include`).
- Map your CSR base as a `volatile uint8_t*` and call the `pf_csr_set_*` / `pf_csr_get_*` functions.
- Bitfield configs (stencil/depth/blend) are packed into 32-bit words via helper pack/unpack functions to avoid C bitfield layout issues.

## Notes
- All CSR data words are 32-bit, little-endian. Viewport, matrices, material, and lights use 16.16 fixed-point in CSRs.
- Texture transforms are omitted when `PIXELFORGE_NUM_TEXTURES == 0`.
