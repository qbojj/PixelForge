# PixelForge - Architektura Projektu

## Przegląd

PixelForge to akcelerator graficzny fixed-pipeline implementujący podzbiór funkcjonalności OpenGL ES 1.1 Common-Lite na platformie FPGA. Projekt realizuje pełny potok graficzny od przetwarzania wierzchołków do generowania obrazów rastrowych, z obsługą oświetlenia, transformacji, rasteryzacji, testów głębokości i szablonu oraz mieszania kolorów.

## Technologie i Narzędzia

### Język HDL i Framework
- **Amaranth HDL** (Python-based HDL) - główny język opisu sprzętu
- **Amaranth SoC** - komponenty infrastruktury systemowej (Wishbone, CSR)
- **Python 3.10+** - język testów i narzędzi pomocniczych

### Platforma Docelowa
- **Intel Cyclone V SoC FPGA** (DE1-SoC)
- **Quartus Prime** - narzędzia syntezy i implementacji
- **LiteX/LiteDRAM** - kontrolery pamięci i infrastruktura SoC

### Testy i Weryfikacja
- **pytest** - framework testowy
- **pytest-xdist** - równoległe wykonywanie testów
- **hypothesis** - testy oparte na właściwościach (property-based testing)

## Architektura Potoku Graficznego

PixelForge implementuje klasyczny fixed-pipeline składający się z następujących etapów:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         GRAPHICS PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. INDEX GENERATION           │  6. PRIMITIVE CLIPPING                 │
│     ↓                          │     ↓                                  │
│  2. INPUT TOPOLOGY PROCESSOR   │  7. PERSPECTIVE DIVIDE                 │
│     ↓                          │     ↓                                  │
│  3. INPUT ASSEMBLY             │  8. TRIANGLE PREPARATION               │
│     ↓                          │     ↓                                  │
│  4. VERTEX TRANSFORM           │  9. TRIANGLE RASTERIZATION             │
│     ↓                          │     ↓                                  │
│  5. VERTEX SHADING             │ 10. DEPTH/STENCIL TEST                 │
│                                │     ↓                                  │
│                                │ 11. BLENDING & OUTPUT                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1. Input Assembly (`gpu/input_assembly/`)

Odpowiada za pobieranie i formatowanie danych wierzchołków z pamięci.

**Komponenty:**
- **IndexGenerator** (`cores.py`): Generuje indeksy wierzchołków (generowane lub z bufora indeksów)
- **InputTopologyProcessor** (`cores.py`): Przetwarza topologię primitywów (triangle list, strip, fan)
- **InputAssembly** (`cores.py`): Łączy dane atrybutów wierzchołków z pamięci

**Obsługiwane topologie:**
- Triangle List
- Triangle Strip
- Triangle Fan
- Z opcjonalnym primitive restart

**Formaty danych:**
- Pozycje, normalne, kolory; koordynaty tekstury są opcjonalne (texturing nie
    jest obecnie zaimplementowane)
- Różne formaty atrybutów (float, fixed-point, integer)
- Konfigurowalne offsety i stride

### 2. Vertex Transform (`gpu/vertex_transform/`)

Transformacje geometryczne wierzchołków.

**Komponenty:**
- **VertexTransform** (`cores.py`): Główny moduł transformacji

**Funkcjonalność:**
- Transformacja pozycji (Model-View-Projection)
- Transformacja normalnych (Inverse Transpose Model-View)
- Macierze 4x4 w formacie fixed-point

**Operacje matematyczne:**
- Mnożenie macierz-wektor
- Pipeline mnożeń i akumulacji
- Optymalizacja dla FPGA (DSP blocks)

### 3. Vertex Shading (`gpu/vertex_shading/`)

System oświetlenia zgodny z modelem Phong.

**Komponenty:**
- **VertexShading** (`cores.py`): Obliczenia oświetlenia per-vertex
- **LightPropertyLayout** (`cores.py`): Konfiguracja właściwości świateł
- **MaterialPropertyLayout** (`cores.py`): Właściwości materiału

**Model oświetlenia:**
- Ambient lighting (światło otaczające)
- Diffuse lighting (światło rozproszone, Lambertian)
- Emissive (emisja materiału)
- Wsparcie dla 8 źródeł światła

Składowa specular nie jest zaimplementowana w obecnej wersji.

**Obliczenia:**
- Iloczyn skalarny normalnej i kierunku światła
- Interpolacja kolorów
- Atenuacja światła (odległość)

### 4. Primitive Assembly (`gpu/primitive_assembly/`)

Łączy wierzchołki w prymitywy (trójkąty).

**Komponenty:**
- **PrimitiveAssembly** (`cores.py`): Buduje trójkąty z przetworzonych wierzchołków

**Funkcjonalność:**
- Face culling (wybór widocznych ścian)
  - Front face / Back face / obie
  - CW/CCW winding order
- Przekazywanie trójkątów do rasteryzacji

### 5. Rasterizer (`gpu/rasterizer/`)

Konwersja trójkątów na fragmenty pikseli.

**Komponenty:**
- **PrimitiveClipper** (`cores.py`): Clipping trójkątów do frustum
- **PerspectiveDivide** (`rasterizer.py`): Dzielenie przez W (projekcja perspektywiczna)
- **TrianglePrep** (`rasterizer.py`): Przygotowanie równań krawędzi
- **TriangleRasterizer** (`rasterizer.py`): Skanowanie trójkątów i generowanie fragmentów

**Algorytmy:**
- Edge function (test punktu w trójkącie)
- Barycentric interpolation (atrybuty fragmentów)
- Incremental rasterization (tile-based)
- Viewport transform
- Scissor test

**Interpolowane atrybuty:**
- Kolory RGBA
- Głębokość Z
- Perspective-correct interpolation

### 6. Pixel Shading (`gpu/pixel_shading/`)

Per-fragment operations - testy i mieszanie kolorów.

**Komponenty:**
- **DepthStencilTest** (`cores.py`): Testy głębokości i szablonu
- **BlendConfig** (`cores.py`): Konfiguracja blendingu
- **SwapchainOutput** (`cores.py`): Zapis do framebuffera

Aktualnie brak etapu teksturowania (niezaimplementowane).

**Depth Test:**
- Funkcje porównania: NEVER, LESS, EQUAL, LEQUAL, GREATER, NOTEQUAL, GEQUAL, ALWAYS
- Depth buffer read/write
- Depth range mapping

**Stencil Test:**
- Osobne konfiguracje dla front/back faces
- Operacje: KEEP, ZERO, REPLACE, INCR, DECR, INVERT, INCR_WRAP, DECR_WRAP
- Maska porównania i zapisu
- Reference value

**Blending:**
- Source/Destination blend factors
- Blend equations (ADD, SUBTRACT, REVERSE_SUBTRACT)
- Alpha blending
- Pre-multiplied alpha support

## Interfejsy Systemowe

### Wishbone Bus
Potok graficzny wykorzystuje magistralę Wishbone do dostępu do pamięci:

- **Vertex Data Bus** - pobieranie atrybutów wierzchołków
- **Depth/Stencil Bus** - dostęp do bufora głębokości/szablonu
- **Color Bus** - dostęp do framebuffera kolorów
- **Texture Bus** - planowane; brak jednostki teksturującej w bieżącej wersji

Parametry magistrali:
- Szerokość adresu: 32-bit
- Szerokość danych: 32-bit
- Brak trybu pipelined; pojedyncze transfery

### Control/Status Registers (CSR)

**GraphicsPipelineCSR** (`pipeline.py`):
Interfejs rejestrów konfiguracyjnych i kontrolnych poprzez Wishbone-CSR bridge.

**GraphicsPipelineAvalonCSR** (`graphics_pipeline_avalon_csr.sv`):
Mostek Wishbone→Avalon-MM generowany w Amaranth HDL (bez Qsys/Platform Designer).

**Mapa rejestrów** (`graphics_pipeline_csr_map.json`):
Szczegółowa mapa wszystkich rejestrów konfiguracyjnych dostępnych z poziomu CPU.

## Struktura Katalogów

```
├── gpu/                        # Główny kod źródłowy HDL
│   ├── input_assembly/         # Input assembly stage
│   │   ├── cores.py           # Główne moduły
│   │   └── layouts.py         # Definicje struktur danych
│   ├── vertex_transform/       # Vertex transform stage
│   │   └── cores.py
│   ├── vertex_shading/         # Vertex shading stage
│   │   └── cores.py
│   ├── primitive_assembly/     # Primitive assembly
│   ├── rasterizer/            # Rasterization stage
│   │   ├── cores.py
│   │   ├── rasterizer.py
│   │   └── layouts.py
│   ├── pixel_shading/         # Pixel/fragment operations
│   │   └── cores.py
│   ├── utils/                 # Wspólne utilities
│   │   ├── layouts.py         # Wspólne layouty danych
│   │   ├── types.py           # Typy i enumy
│   │   └── avalon.py          # Avalon interface utils
│   └── pipeline.py            # Top-level pipeline integration
│
├── tests/                      # Testy jednostkowe i integracyjne
│   ├── input_assembly/
│   ├── vertex_transform/
│   ├── vertex_shading/
│   ├── rasterizer/
│   ├── pixel_shading/
│   └── utils/                 # Test utilities
│       ├── testbench.py
│       ├── streams.py
│       └── visualization.py   # PPM image generation
│
├── quartus/                    # Pliki projektu Intel Quartus
│   ├── soc_system.qsf         # Project settings
│   ├── soc_system.qpf         # Project file
│   ├── soc_system.qsys        # Platform Designer SoC
│   ├── ghrd_top.v             # Top-level HDL
│   └── software/              # Bootloader/U-Boot
│
├── software/                   # Aplikacje demo (C)
│   ├── src/                   # Kod źródłowy demo
│   ├── include/               # Nagłówki API
│   ├── demo_cube              # Podstawowy obracający się sześcian
│   ├── demo_lighting          # Demo oświetlenia
│   ├── demo_depth             # Demo depth buffer
│   ├── demo_stencil           # Demo stencil buffer
│   └── DEMOS.md               # Dokumentacja demo
│
├── tools/                      # Narzędzia pomocnicze
│   └── gen_csr_header.py      # Generator nagłówków CSR
│
└── thesis/                     # Praca dyplomowa
    ├── thesis.tex
    └── iithesis.cls           # Szablon WMiI UWr
```

## Przepływ Danych

### 1. Inicjalizacja Draw Call (CPU → GPU)

```
CPU writes CSR registers:
├── Framebuffer config (width, height, addresses)
├── Vertex attribute pointers & formats
├── Transform matrices (MVP, Normal)
├── Material & lighting parameters
├── Render state (depth/stencil/blend)
└── Draw parameters (index buffer, count)

CPU writes START register → GPU starts processing
```

### 2. Vertex Processing

```
IndexGenerator
    ↓ vertex indices
InputTopologyProcessor
    ↓ topology-processed indices
InputAssembly
    ↓ assembled vertex attributes (via Wishbone reads)
VertexTransform
    ↓ transformed positions & normals
VertexShading
    ↓ lit vertices with colors
PrimitiveAssembly
    ↓ complete triangles
```

### 3. Rasterization & Fragment Processing

```
PrimitiveClipper
    ↓ clipped triangles
PerspectiveDivide
    ↓ screen-space coordinates
TrianglePrep
    ↓ edge equations & setup
TriangleRasterizer
    ↓ fragments with interpolated attributes
DepthStencilTest
    ↓ fragments that pass tests (via Wishbone read/write)
BlendOperation & SwapchainOutput
    ↓ final pixel colors (via Wishbone write)
Framebuffer Memory
```

## Typy Danych

### Fixed-Point Arithmetic

Pipeline wykorzystuje głównie arytmetykę fixed-point dostosowaną do bloków DSP
27x27 / 18x18 / 9x9 w Cyclone V:
- **Q13.13** (27-bit) – pozycje, normalne, macierze transformacji
- **Q1.17** (18-bit) – współczynniki barycentryczne, głębokość, znormalizowane
    wektory kierunków
- **UQ0.9** (9-bit, bez znaku) – kanały koloru/alphę w zakresie 0–1

Dobór formatów minimalizuje zużycie bloków DSP i rejestrów przy zachowaniu
akceptowalnej precyzji dla potoku rastrowego 2D/3D.

### Streaming Protocol

Amaranth Stream protocol z sygnałami:
- `valid` - dane są ważne
- `ready` - odbiorca gotowy
- `payload` - struktura danych
- `first` / `last` - oznaczenia początku/końca sekwencji

## Charakterystyka Wydajnościowa

### Teoretyczna Przepustowość

**Vertex Processing:**
- Input Assembly: ~1 wierzchołek / cykl (sekwencyjnie, bez burst/pipeline)
- Vertex Transform: ~10-20 cykli / wierzchołek (w zależności od liczby macierzy)
- Vertex Shading: ~5-15 cykli / wierzchołek (w zależności od liczby świateł)

**Rasterization:**
- Triangle Setup: ~10 cykli / trójkąt
- Fragment Generation: 1 fragment / cykl (teoretycznie)
- Depth/Stencil Test: 2-4 cykle / fragment (memory access)
- Blending: 2-4 cykle / fragment (memory read-modify-write)

### Ograniczenia
- Dostęp do pamięci (Wishbone latency)
- Szerokość magistrali (32-bit)
- Brak cache'owania
- Single-pipeline (no parallelism between stages yet)

## Integracja z SoC

### Platform Designer (Qsys)

Komponent GPU integruje się z systemem poprzez:
- **Avalon Memory-Mapped Slave** - CSR interface
- **Avalon Memory-Mapped Master** (x3) - Vertex, Depth/Stencil, Color memory access
- **Clock/Reset** - synchronizacja z resztą systemu

### Połączenie z HPS (Hard Processor System)

ARM Cortex-A9 w Cyclone V:
- Zapisuje konfigurację poprzez lightweight HPS-to-FPGA bridge
- Alokuje bufory w SDRAM
- Czyta framebuffer do wyświetlenia (VGA/HDMI)
- Może czytać status GPU (idle/busy)

## Testowanie

### Methodology

1. **Unit Tests** - testowanie pojedynczych komponentów
   - Każdy stage potoku ma osobne testy
   - Symulacja Amaranth (Simulator)
   - Weryfikacja protocołu stream

2. **Integration Tests** - testowanie całych ścieżek
   - Rasterizer pipeline (clip → divide → prep → raster)
   - Vertex pipeline (assembly → transform → shading)

3. **Visual Verification** - generowanie obrazów testowych
   - PPM file output z testów rasteryzera
   - Porównanie z reference rendering

4. **Property-Based Testing** - Hypothesis
   - Losowe dane wejściowe
   - Weryfikacja niezmienników (invariants)

### Test Fixtures

`tests/utils/`:
- `testbench.py` - infrastruktura symulacji
- `streams.py` - helpery dla stream protocol
- `visualization.py` - generowanie PPM, statystyki

### Coverage

Testy pokrywają:
- [x] Input assembly (topologies, formats)
- [x] Vertex transform (matrices)
- [x] Vertex shading (lighting)
- [x] Rasterization (triangles, interpolation)
- [x] Depth/stencil tests
- [~] Blending (częściowo)

## Narzędzia Deweloperskie

### gen_csr_header.py

Generator nagłówków C z mapy rejestrów CSR:
```bash
python tools/gen_csr_header.py \
    --json graphics_pipeline_csr_map.json \
    --out software/include/graphics_pipeline_csr.h
```

### Build System

**Python/Amaranth:**
```bash
pip install -e .                    # Install package
pytest tests/                       # Run tests
pytest -n auto tests/               # Parallel execution
```

**Quartus:**
- quartus/soc_system.qpf -> projekt Quartus (gotowy dla DE1-SoC)

Wygenerowane pliki bitstream:
- quartus/output_files/soc_system.sof
- quartus/output_files/soc_system.rbf

**Software:**
```bash
cd software
make                                # Build all demos
make demo_lighting                  # Build specific demo
```

## Rozszerzenia i Planowane Funkcjonalności

### Krótkoterminowe
- [ ] Texturing pipeline
- [ ] Bilinear texture filtering
- [ ] Multiple render targets
- [ ] Antialiasing (MSAA)

### Średnioterminowe
- [ ] Shader pipeline (basic programmable shaders)
- [ ] Geometry instancing
- [ ] Occlusion queries
- [ ] Performance counters

### Długoterminowe
- [ ] Compute shaders
- [ ] Ray tracing acceleration
- [ ] Multi-threading / parallel rasterizers

## Bibliografia Techniczna

- OpenGL ES 1.1 Common-Lite Specification
- Amaranth HDL Documentation
- Intel Cyclone V Handbook
- "Real-Time Rendering" - Akenine-Möller, Haines
- "Computer Graphics: Principles and Practice" - Foley, van Dam
