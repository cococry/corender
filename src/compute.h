#pragma once


#include <stdint.h>
#include <stdbool.h>
#include <vulkan/vulkan_core.h>
#include "../include/corender/corender.h"
#include "mem.h"

struct cr_tile_header_t {
  uint32_t n_segments;
};

struct cr_compute_pipeline_push_constant_t {
  uint32_t screen_w,  screen_h;
  uint32_t n_tiles_x, n_tiles_y;
  uint32_t n_macrotiles_x, n_macrotiles_y;
  uint32_t tile_size, macrotile_size;
  
  uint32_t n_segments;
  uint32_t n_paths;

  uint32_t fill_rule;
};

enum cr_compute_pipeline_fill_rule_t {
  CR_COMPUTE_FILL_RULE_EVEN_ODD = 0,
  CR_COMPUTE_FILL_RULE_NON_ZERO
};

struct cr_compute_pipeline_init_info_t {
  uint32_t tile_size;

  uint32_t screen_w, screen_h;

  enum cr_compute_pipeline_fill_rule_t fill_rule;

  char** shader_paths;
  uint32_t n_shaders;
  
  struct cr_compute_pipeline_layout_binding_t* layout_bindings;
  uint32_t n_layout_bindings;
};

struct cr_compute_pipeline_dynamic_state_t {
  struct cr_gpu_buffer_t segment_buf; 
  uint32_t n_segments, n_paths;

  struct cr_segment_t* segment_data;
};

struct cr_compute_kernel_t {
  char* shader_path;
  uint32_t hash;
  VkPipeline kernel_pipeline;
};

struct cr_compute_pipeline_layout_binding_t {
  const char* name;
  size_t buffer_size;
};

struct cr_compute_pipeline_layout_buffer_t {
  uint32_t hash;
  struct cr_gpu_buffer_t buf;
};

struct cr_compute_pipeline_t {
  struct cr_compute_kernel_t* kernels;

  struct cr_compute_pipeline_init_info_t info;

  struct cr_compute_pipeline_dynamic_state_t* dynamic;

  struct cr_storage_image_t storage_img;

  struct cr_compute_pipeline_layout_buffer_t* buffers;
  uint32_t n_buffers;
};

bool cr_compute_pipeline_init(struct cr_context_t* ctx, struct cr_compute_pipeline_init_info_t* info, struct cr_compute_pipeline_t* pipeline);

bool cr_compute_pipeline_resize(struct cr_context_t* ctx, struct cr_compute_pipeline_t* pipeline, uint32_t w, uint32_t h);

bool cr_compute_pipeline_dispatch(struct cr_context_t* ctx, struct cr_compute_pipeline_t* pipeline, uint32_t frame_idx,
    uint32_t swapchain_image_idx);

bool cr_compute_pipeline_get_internal_shader_paths(struct cr_context_t* ctx, const char* subpath, char*** o_paths, uint32_t* o_n_paths); 

bool cr_compute_pipeline_insert_segment(struct cr_context_t* ctx, struct cr_compute_pipeline_t* pipeline,
    struct cr_segment_t segment, uint32_t swapchain_idx);
