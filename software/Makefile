CROSS_COMPILE ?=
CC = $(CROSS_COMPILE)gcc
AR = $(CROSS_COMPILE)ar
CFLAGS ?= -O2 -Wall -Wextra -std=c11 -static
INCLUDE := -Iinclude
SRC := src/graphics_pipeline_csr_access.c src/pixelforge_utils.c src/demo_utils.c src/udma_alloc.c src/obj_loader.c
OBJ := $(SRC:.c=.o)
LIB := libpixelforgecsr.a
DEMO := pixelforge_demo
DEMO_CUBE := demo_cube
DEMO_DEPTH := demo_depth
DEMO_OBJ := demo_obj
DUMP_VGA := dump_vga_dma
DUMP_FB2 := dump_fb2
DUMP_GPU_CSR := dump_gpu_csr
DEVMEM2 := devmem2
OBJS := $(shell find . -iname '*.obj')

DEMOS := $(DEMO) $(DEMO_CUBE) $(DEMO_DEPTH) $(DEMO_OBJ)
DUMPS := $(DUMP_FB2) $(DUMP_GPU_CSR) $(DUMP_VGA)
TOOLS := $(DEVMEM2)

all: $(LIB) $(DEMOS) $(DUMPS) $(TOOLS)

$(LIB): $(OBJ)
	$(AR) rcs $@ $(OBJ)

src/%.o: src/%.c
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $@ $< -MMD

$(DEMO): src/pixelforge_demo.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $^ -L. -lpixelforgecsr -lm

$(DEMO_CUBE): src/demo_cube.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $^ -L. -lpixelforgecsr -lm

$(DEMO_DEPTH): src/demo_depth.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $^ -L. -lpixelforgecsr -lm

$(DEMO_OBJ): src/demo_obj.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $^ -L. -lpixelforgecsr -lm

$(DUMP_FB2): src/dump_fb2.o
	$(CC) $(CFLAGS) -o $@ $^

$(DUMP_VGA): src/dump_vga_dma.o
	$(CC) $(CFLAGS) -o $@ $^

$(DUMP_GPU_CSR): src/dump_gpu_csr.o
	$(CC) $(CFLAGS) -o $@ $^ -L. -lpixelforgecsr -lm

$(DEVMEM2): src/devmem2.o
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f $(LIB)
	rm -f $(DEMOS) $(DUMPS) $(TOOLS)
	find src -name '*.o' -delete
	find src -name '*.d' -delete

install: $(DEMOS) $(DUMPS) $(TOOLS) $(OBJS)
	cp -f $^ $(DESTDIR)

.PHONY: all clean install

-include $(OBJ:.o=.d)
