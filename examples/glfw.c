#include <time.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include "../include/corender/corender.h"


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

    .layers = NULL,
    .n_layers = 0,

    .log_verbose = true, 
    .surface_create = _glfw_surface_create,
    .surface_userdata = window
  };
  cr_context_create(&ctx, &info);

  srand(time(0));

    uint32_t rng = 0x12345678;
  /* Loop until the user closes the window */
  while (!glfwWindowShouldClose(window)) {
    cr_draw_begin(&ctx);

    for (uint32_t i = 0; i < 100000; i++) {
      float x = (float)(xorshift32(&rng) % ctx.swapchain.dimensions.width);
      float y = (float)(xorshift32(&rng) % ctx.swapchain.dimensions.height);

      vec4 color = {
        rand01(&rng),
        rand01(&rng),
        rand01(&rng),
        1.0f
      };
      cr_draw_rect(&ctx, (vec2){x, y}, (vec2){8, 8},
                   xorshift32(&rng) % 255,
                   xorshift32(&rng) % 255,
                   xorshift32(&rng) % 255,
                   255);

    }

    cr_draw_end(&ctx);
    glfwPollEvents();
  }

  cr_context_destroy(&ctx);
  glfwTerminate();
  return 0;
}
