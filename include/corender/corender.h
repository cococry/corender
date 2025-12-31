#pragma once 
#include <cglm/types.h>
#include <vulkan/vulkan_core.h>
#include <stdbool.h>
#include <stdio.h>
#include <cglm/cglm.h>

struct cr_surface_t {
  VkSurfaceKHR surf;
  uint32_t width, height;
};

#define CR_FRAME_COUNT 2
#define CR_MAX_BATCH 5000
#define CR_INITIAL_BATCH_CAP 4

struct cr_vertex_t {
  vec2 pos;
};

struct cr_instance_t {
  vec2 pos, size;
  vec4 color;
};


struct cr_gpu_buffer_t {
  VkBuffer buf;
  size_t buf_size;
  void* mem_handle;

  void* _vma_allocation;
  VkBufferUsageFlags _usage;
  VkMemoryPropertyFlags _mem_props;
};

struct cr_instanced_state_t {
  struct cr_gpu_buffer_t vbo, ibo, instance_buf;
  uint32_t n_instances;

  uint32_t instance_write_offset;
};
struct cr_batch_state_t {
  uint32_t n_vertices;
  uint32_t vert_max;
  
  uint32_t n_indices;
  uint32_t indicies_max;

  struct cr_instanced_state_t instanced;

  uint32_t batch_group_cap;
};

struct cr_frame_t {
  VkCommandPool cmd_pool;
  VkCommandBuffer cmd_buf;

  VkSemaphore image_available;
  VkSemaphore* render_finished_per_image;
  VkFence in_flight_fence;

  struct cr_batch_state_t batch_state;
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

struct cr_render_pass_info {
  vec4 clear_color;
};

struct cr_context_t {
  VkInstance instance;
  VkPhysicalDevice phys_dev;
  VkDevice logical_dev;
  uint32_t graphics_queue_family, present_queue_family;
  VkQueue graphics_queue, present_queue;
  VkCommandPool cmd_pool;
  VkPipeline pipeline;
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
};

bool cr_context_create(struct cr_context_t* ctx, const struct cr_context_init_info_t* info);
bool cr_context_destroy(struct cr_context_t* ctx);

void cr_draw_set_clear_color(struct cr_context_t* ctx, vec4 color);

bool cr_draw_begin(struct cr_context_t* ctx);
void cr_draw_rect(struct cr_context_t* ctx, vec2 pos, vec2 size, vec4 color);
bool cr_draw_end(struct cr_context_t* ctx);

void cr_surface_resize(struct cr_context_t* ctx, uint32_t width, uint32_t height);

