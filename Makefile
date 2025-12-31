CC=gcc

EXAMPLE_SRCS := $(wildcard examples/*.c)
CORENDER_SRCS := $(wildcard src/*.c)
CORENDER_OBJS := $(patsubst src/%.c,lib/%.o,$(CORENDER_SRCS))
EXAMPLE_BINS := $(patsubst examples/%.c,bin/examples/%,$(EXAMPLE_SRCS))
EXAMPLE_LIBS_glfw   := -lglfw -lGL -lvulkan

all: lib/libcorender.a shaders 

.PHONY: shaders

shaders: 
	rm -rf ~/.local/state/corender/shaders/*
	mkdir -p bin/shaders 
	mkdir -p bin/shaders/default 
	mkdir -p bin/shaders/instanced
	glslc -fshader-stage=vertex shaders/instanced/basic_vert.glsl -o bin/shaders/instanced/basic_vert.spv 
	glslc -fshader-stage=fragment shaders/instanced/basic_frag.glsl -o bin/shaders/instanced/basic_frag.spv 
	
	glslc -fshader-stage=vertex shaders/default/basic_vert.glsl -o bin/shaders/default/basic_vert.spv 
	glslc -fshader-stage=fragment shaders/default/basic_frag.glsl -o bin/shaders/default/basic_frag.spv 
	mkdir -p ~/.local/state/corender 
	cp -r ./bin/shaders ~/.local/state/corender/

lib/libcorender.a: $(CORENDER_OBJS) 
	g++ -c vendor/vma/vma_impl.cpp -o lib/vma_impl.o 
	ar rcs $@ $^ lib/vma_impl.o 

lib/%.o: src/%.c | lib
	$(CC) $(CFLAGS) -c $< -o $@ 

lib:
	mkdir -p lib/

clean:
	rm -rf lib bin

install:
	install -Dm644 lib/libcorender.a /usr/local/lib/libcorender.a
	install -d /usr/local/include/corender
	cp -r include/corender/* /usr/local/include/corender

uninstall:
	rm -f /usr/local/lib/libcorender.a 
	rm -rf /usr/local/include/corender

examples: lib/libcorender.a bin/examples $(EXAMPLE_BINS)


clean-examples: 
	rm -rf bin/examples/

bin/examples:
	mkdir -p $@ 

bin/examples/%: examples/%.c | bin/examples
	$(CC) $(CFLAGS) $< -o $@ -Llib -lcorender $(EXAMPLE_LIBS_$*) -lstdc++


