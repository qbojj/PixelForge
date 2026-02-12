CROSS_COMPILE ?=
CC = $(CROSS_COMPILE)gcc
AR = $(CROSS_COMPILE)ar
CFLAGS ?= -O2 -Wall -Wextra -std=c11 -static
LDFLAGS ?= -static
LDLIBS ?= -lm

INCLUDE := -Iinclude
SRC := src/graphics_pipeline_csr_access.c src/pixelforge_utils.c src/demo_utils.c src/udma_alloc.c src/obj_loader.c src/frame_capture.c src/gles11_wrapper.c
OBJ := $(SRC:.c=.o)
LIB := libpixelforgecsr.a
DEMO := pixelforge_demo
DEMO_CUBE := demo_cube
DEMO_DEPTH := demo_depth
DEMO_OBJ := demo_obj
DEMO_ALPHA := demo_alpha
DEMO_GLES := demo_gles
DUMP_VGA := dump_vga_dma
DUMP_GPU_CSR := dump_gpu_csr
OBJS := $(shell find . -iname '*.obj')

DEMOS := $(DEMO) $(DEMO_CUBE) $(DEMO_DEPTH) $(DEMO_OBJ) $(DEMO_ALPHA) $(DEMO_GLES)
DUMPS := $(DUMP_GPU_CSR) $(DUMP_VGA)

all: $(LIB) $(DEMOS) $(DUMPS)

$(LIB): $(OBJ)
	$(AR) rcs $@ $(OBJ)

src/%.o: src/%.c
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $@ $< -MMD

$(DEMO): src/pixelforge_demo.o $(LIB)
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(DEMO_CUBE): src/demo_cube.o $(LIB)
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(DEMO_DEPTH): src/demo_depth.o $(LIB)
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(DEMO_OBJ): src/demo_obj.o $(LIB)
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(DEMO_ALPHA): src/demo_alpha.o $(LIB)
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(DEMO_GLES): src/demo_gles.o $(LIB)
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(DUMP_VGA): src/dump_vga_dma.o
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(DUMP_GPU_CSR): src/dump_gpu_csr.o $(LIB)
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)

clean:
	rm -f $(LIB)
	rm -f $(DEMOS) $(DUMPS)
	find src -name '*.o' -delete
	find src -name '*.d' -delete

install: $(DEMOS) $(DUMPS) $(OBJS)
	cp -f $^ $(DESTDIR)

.PHONY: all clean install

-include $(OBJ:.o=.d)
