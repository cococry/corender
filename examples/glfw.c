#include <time.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include "../include/corender/corender.h"
#include <glia/glia.h>


static struct cr_context_t ctx; 
void _glfw_resize(GLFWwindow* window, int width, int height) {
  cr_surface_resize(&ctx, width, height);
}

static inline uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static inline float rand01(uint32_t *state) {
    return (xorshift32(state) >> 8) * (1.0f / 16777216.0f);
}

static bool _glfw_surface_create(
    VkInstance instance,
    struct cr_surface_t* o_surf,
    void* userdata) {
  GLFWwindow* win = (GLFWwindow*)userdata;

 glfwCreateWindowSurface(instance, win, NULL, &o_surf->surf);

  if(o_surf->surf == NULL) {
    printf("Error: failed to create window surface.\n");
  }

  int w, h;
  glfwGetFramebufferSize(win, &w, &h);

  o_surf->width = (uint32_t)w;
  o_surf->height = (uint32_t)h;

  return true;
}
int main() {
  GLFWwindow* window;

  /* Initialize the library */
  if (!glfwInit())
    return -1;

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  /* Create a windowed mode window and its OpenGL context */
  window = glfwCreateWindow(640, 480, "corender - GLFW example", NULL, NULL);
  if (!window) {
    glfwTerminate();
    return -1;
  }

  glfwSetFramebufferSizeCallback(window, _glfw_resize);


  uint32_t n_exts;
  const char** exts = glfwGetRequiredInstanceExtensions(&n_exts);

  const char* validation_layers[1] = {
    "VK_LAYER_KHRONOS_validation"
  };
  struct cr_context_init_info_t info = {
    .enable_validation = true,
    .n_exts = n_exts,
    .exts = exts,
    .enable_time_measuring = false,

    .layers = validation_layers,
    .n_layers = 1,

    .log_verbose = true, 
    .surface_create = _glfw_surface_create,
    .surface_userdata = window
  };
  cr_context_create(&ctx, &info);

  srand(time(0));

  ia_init(window);

    uint32_t rng = 0x12345678;


  float size = 10.0f;
  /* Loop until the user closes the window */
  while (!glfwWindowShouldClose(window)) {
    cr_draw_begin(&ctx);

    cr_draw_segment(&ctx, (struct cr_segment_t){ .p0 = {0,     0},    .p1 = {size,  0} });
cr_draw_segment(&ctx, (struct cr_segment_t){ .p0 = {size,  0},    .p1 = {size / 2.0f,     size} });
cr_draw_segment(&ctx, (struct cr_segment_t){ .p0 = {size /2.0f,     size}, .p1 = {0,     0} });


    if(ia_key_is_released(GLFW_KEY_DOWN)) {
      printf("DOwn!\n");
      size += 2;
    }

    cr_draw_end(&ctx);
    glfwPollEvents();
  }

  cr_context_destroy(&ctx);
  glfwTerminate();
  return 0;
}
