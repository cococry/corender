#pragma once 
#include <cglm/types.h>
#include <vulkan/vulkan_core.h>
#include <stdbool.h>
#include <stdio.h>
#include <cglm/cglm.h>

#define CR_FRAME_COUNT 2

struct cr_surface_t {
  VkSurfaceKHR surf;
  uint32_t width, height;
};

struct cr_instance_vertex_t {
  vec2 pos;
};

struct cr_vertex_t {
  vec2 pos;
  vec4 color;
};


struct cr_instance_t {
  _Float16 px, py, sx, sy;
  uint8_t r, g, b, a;
};

struct cr_frame_t {
  VkCommandPool cmd_pool;
  VkCommandBuffer cmd_buf;

  VkSemaphore image_available;
  VkSemaphore* render_finished_per_image;
  VkFence in_flight_fence;
  VkQueryPool timestamp_pool;

  size_t staging_begin; // value of ring.head at frame begin
  size_t staging_end;   // max head reached in this frame
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
  VkImageView* img_views_depth;
  VkImage* depth_images;
};

struct cr_frameloop_t {
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

  bool enable_validation, enable_time_measuring;

  void* surface_userdata;
  cr_surface_create_func_t surface_create;

  bool log_to_file, log_verbose,  log_quiet;

};

struct cr_log_state_t {
  FILE* stream;
  bool verbose, quiet;
};

struct cr_render_pass_info {
  vec4 clear_color;
};

struct cr_draw_command_t {
  uint32_t n_instances;
  uint32_t instance_write_offset;
  uint32_t vertex_write_offset;
  uint32_t n_vertices; 
  bool instanced_draw;
};

struct cr_context_t {
  VkInstance instance;
  VkPhysicalDevice phys_dev;
  VkPhysicalDeviceLimits phys_dev_limits; 
  VkDevice logical_dev;
  uint32_t graphics_queue_family, present_queue_family;
  VkQueue graphics_queue, present_queue;

  VkPipelineLayout pipeline_layout;

  struct cr_surface_t surf;
  struct cr_swapchain_t swapchain;
  struct cr_frameloop_t frameloop;

  struct cr_log_state_t log;

  struct {
    bool pending;  
    uint32_t width, height;
  } pending_resize; 

  uint32_t _swapchain_img_idx;
  bool _skip_render;

  struct cr_render_pass_info _pass_info;

  struct cr_draw_command_t pending_draws[1024];
  uint32_t n_pending_draws;

  bool _have_multi_draw_indirect;

  double ms_gpu, ms_cpu;
  bool enable_time_measuring;
};

bool cr_context_create(struct cr_context_t* ctx, const struct cr_context_init_info_t* info);
bool cr_context_destroy(struct cr_context_t* ctx);

void cr_draw_set_clear_color(struct cr_context_t* ctx, vec4 color);

bool cr_draw_begin(struct cr_context_t* ctx);
void cr_draw_rect(struct cr_context_t* ctx, vec2 pos, vec2 size, uint8_t r, uint8_t g, uint8_t b, uint8_t a);
void cr_draw_vertex(struct cr_context_t* ctx, vec2 pos, vec4 color);
bool cr_draw_end(struct cr_context_t* ctx);

void cr_surface_resize(struct cr_context_t* ctx, uint32_t width, uint32_t height);

