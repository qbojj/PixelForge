CROSS_COMPILE ?=
CC = $(CROSS_COMPILE)gcc
AR = $(CROSS_COMPILE)ar
CFLAGS ?= -O2 -Wall -Wextra -std=c11 -static
INCLUDE := -Iinclude
SRC := src/graphics_pipeline_csr_access.c
OBJ := $(SRC:.c=.o)
LIB := libpixelforgecsr.a
DEMO := pixelforge_demo
DUMP_VGA := dump_vga_dma
DUMP_GPU_CSR := dump_gpu_csr

all: $(LIB) $(DEMO) $(DUMP_VGA) $(DUMP_GPU_CSR)

$(LIB): $(OBJ)
	$(AR) rcs $@ $(OBJ)

src/%.o: src/%.c include/graphics_pipeline_csr_access.h include/graphics_pipeline_csr.h include/graphics_pipeline_formats.h
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $@ $<

$(DEMO): $(LIB) src/pixelforge_demo.c
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ src/pixelforge_demo.c $(LIB)

$(DUMP_VGA): src/dump_vga_dma.c include/vga_dma.h
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ src/dump_vga_dma.c

$(DUMP_GPU_CSR): $(LIB) src/dump_gpu_csr.c include/graphics_pipeline_csr_access.h
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ src/dump_gpu_csr.c $(LIB)

clean:
	rm -f $(OBJ) $(LIB) $(DEMO) $(DUMP_VGA) $(DUMP_GPU_CSR)

.PHONY: all clean
