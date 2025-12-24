CC=gcc

EXAMPLE_SRCS := $(wildcard examples/*.c)
CORENDER_SRCS := $(wildcard src/*.c)
CORENDER_OBJS := $(patsubst src/%.c,lib/%.o,$(CORENDER_SRCS))
EXAMPLE_BINS := $(patsubst examples/%.c,bin/examples/%,$(EXAMPLE_SRCS))
EXAMPLE_LIBS_glfw   := -lglfw -lGL -lvulkan

all: lib/libcorender.a 

lib/libcorender.a: $(CORENDER_OBJS) 
	ar rcs $@ $^ 

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
	mkdir -p bin/examples

bin/examples/%: examples/%.c | bin/examples
	$(CC) $(CFLAGS) $< -o $@ -Llib -lcorender $(EXAMPLE_LIBS_$*)


