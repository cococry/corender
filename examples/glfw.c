#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <corender/corender.h>
#include <stdlib.h>



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


  uint32_t n_exts;
  const char** exts = glfwGetRequiredInstanceExtensions(&n_exts);

  const char* validation_layers[1] = {
    "VK_LAYER_KHRONOS_validation"
  };
  struct cr_context_t ctx; 
  struct cr_context_init_info_t info = {
    .enable_validation = true,
    .n_exts = n_exts,
    .exts = exts,

    .layers = validation_layers,
    .n_layers = 1,

    .log_verbose = true, 
    .surface_create = _glfw_surface_create,
    .surface_userdata = window
  };
  cr_context_create(&ctx, &info);

  /* Loop until the user closes the window */
  while (!glfwWindowShouldClose(window)) {
    cr_draw_frame(&ctx);
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  cr_context_destroy(&ctx);
  glfwTerminate();
  return 0;
}
