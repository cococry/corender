#pragma once 
#include <vulkan/vulkan_core.h>
#include <stdbool.h>
#include <stdio.h>

struct cr_surface_t {
  VkSurfaceKHR surf;
  uint32_t width, height;
};

#define CR_FRAME_COUNT 2

struct cr_frame_t {
  VkCommandPool cmd_pool;
  VkCommandBuffer cmd_buf;

  VkSemaphore image_available;
  VkSemaphore* render_finished_per_image;
  VkFence in_flight_fence;
};

struct cr_swapchain_t {
  VkSwapchainKHR swapchain_handle;
  VkDevice logical_dev;

  VkExtent2D dimensions;
  VkFormat fmt;
  VkSurfaceFormatKHR surf_fmt;
  VkPresentModeKHR present_mode;

  uint32_t n_imgs;
  VkImage* imgs;
  VkImageView* img_views;
 
};

struct cr_frameloop_t {
  struct cr_swapchain_t swapchain;

  VkFramebuffer* fbs;
  uint32_t n_fbs;

  VkRenderPass crnt_pass;
  
  struct cr_frame_t frames[CR_FRAME_COUNT];
  uint32_t frame_idx;

  VkFence* swapchain_image_fences;

};

typedef bool (*cr_surface_create_func_t)(
    VkInstance instance,
    struct cr_surface_t* o_surf,
    void* userdata
    );

struct cr_context_init_info_t {
  const char** exts;
  size_t n_exts;

  const char** layers;
  size_t n_layers;

  bool enable_validation;

  void* surface_userdata;
  cr_surface_create_func_t surface_create;

  bool log_to_file, log_verbose,  log_quiet;
};

struct cr_log_state_t {
  FILE* stream;
  bool verbose, quiet;
};

struct cr_context_t {
  VkInstance instance;
  VkPhysicalDevice phys_dev;
  VkDevice logical_dev;
  uint32_t graphics_queue_family, present_queue_family;
  VkQueue graphics_queue, present_queue;
  VkCommandPool cmd_pool;

  struct cr_surface_t surf;
  struct cr_swapchain_t swapchain;
  struct cr_frameloop_t frameloop;

  struct cr_log_state_t log;
};

bool cr_context_create(struct cr_context_t* ctx, const struct cr_context_init_info_t* info);
bool cr_context_destroy(struct cr_context_t* ctx);
bool cr_draw_frame(struct cr_context_t* ctx);
