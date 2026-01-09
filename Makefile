CC      := gcc
CXX     := g++
AR      := gcc-ar

BUILD ?= debug

CFLAGS_DEBUG := \
	-g -O0 \
	-DDEBUG

CFLAGS_RELEASE := \
	-O3 -DNDEBUG \
	-march=native \
	-flto \
	-fno-plt \
	-fomit-frame-pointer \
	-fstrict-aliasing

CFLAGS   := $(CFLAGS_$(shell echo $(BUILD) | tr a-z A-Z))
CXXFLAGS := $(CFLAGS)
LDFLAGS  := $(CFLAGS)
BUILD_DIR := build/$(BUILD)
LIB_DIR   := $(BUILD_DIR)/lib
BIN_DIR   := $(BUILD_DIR)/bin
CORENDER_SRCS := $(wildcard src/*.c)
EXAMPLE_SRCS  := $(wildcard examples/*.c)

CORENDER_OBJS := $(patsubst src/%.c,$(LIB_DIR)/%.o,$(CORENDER_SRCS))
EXAMPLE_BINS  := $(patsubst examples/%.c,$(BIN_DIR)/examples/%,$(EXAMPLE_SRCS))

EXAMPLE_LIBS_glfw := -lglfw -lGL -lvulkan

all: $(LIB_DIR)/libcorender.a shaders

.PHONY: all clean shaders examples install uninstall

shaders:
	rm -rf ~/.local/state/corender/shaders/*
	mkdir -p bin/shaders/default
	mkdir -p bin/shaders/instanced

	glslc -fshader-stage=vertex shaders/instanced/basic_vert.glsl \
		-o bin/shaders/instanced/basic_vert.spv
	glslc -fshader-stage=fragment shaders/instanced/basic_frag.glsl \
		-o bin/shaders/instanced/basic_frag.spv

	glslc -fshader-stage=vertex shaders/default/basic_vert.glsl \
		-o bin/shaders/default/basic_vert.spv
	glslc -fshader-stage=fragment shaders/default/basic_frag.glsl \
		-o bin/shaders/default/basic_frag.spv

	mkdir -p ~/.local/state/corender
	cp -r bin/shaders ~/.local/state/corender/

$(LIB_DIR)/libcorender.a: $(CORENDER_OBJS)
	mkdir -p $(LIB_DIR)
	$(CXX) $(CXXFLAGS) -c vendor/vma/vma_impl.cpp -o $(LIB_DIR)/vma_impl.o
	$(AR) rcs $@ $^ $(LIB_DIR)/vma_impl.o

$(LIB_DIR)/%.o: src/%.c
	mkdir -p $(LIB_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

examples: $(LIB_DIR)/libcorender.a $(BIN_DIR)/examples $(EXAMPLE_BINS)

$(BIN_DIR)/examples:
	mkdir -p $@

$(BIN_DIR)/examples/%: examples/%.c | $(BIN_DIR)/examples
	$(CC) $(CFLAGS) $< -o $@ \
		-L$(LIB_DIR) -lcorender \
		$(EXAMPLE_LIBS_$*) \
		-lstdc++ \
		$(LDFLAGS)

install:
	install -Dm644 $(LIB_DIR)/libcorender.a /usr/local/lib/libcorender.a
	install -d /usr/local/include/corender
	cp -r include/corender/* /usr/local/include/corender

uninstall:
	rm -f /usr/local/lib/libcorender.a
	rm -rf /usr/local/include/corender

clean:
	rm -rf build bin
