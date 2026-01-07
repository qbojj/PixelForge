CC ?= gcc
AR ?= ar
CFLAGS ?= -O2 -Wall -Wextra -std=c11
INCLUDE := -Iinclude
SRC := src/graphics_pipeline_csr_access.c
OBJ := $(SRC:.c=.o)
LIB := libpixelforgecsr.a

all: $(LIB)

$(LIB): $(OBJ)
	$(AR) rcs $@ $(OBJ)

src/%.o: src/%.c include/graphics_pipeline_csr_access.h include/graphics_pipeline_csr.h include/graphics_pipeline_formats.h
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $@ $<

clean:
	rm -f $(OBJ) $(LIB)

.PHONY: all clean
