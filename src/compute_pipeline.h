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
  uint32_t tile_size;
  
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
};


typedef struct cr_storage_image_t {
    VkImage         image;
    VkImageView     view;
    VmaAllocation   allocation;
    uint32_t        width;
    uint32_t        height;
} cr_storage_image_t;


struct cr_compute_pipeline_dynamic_state_t {
  struct cr_gpu_buffer_t segment_buf; 
  uint32_t n_segments, n_paths;

  struct cr_segment_t* segment_data;
};

struct cr_compute_pipeline_t {
  VkPipeline pipeline_bin, pipeline_fill, pipeline_base_parity, pipeline_prefix_per_sg, pipeline_prefix_all_sgs,
             pipeline_prefix_final;

  struct cr_compute_pipeline_init_info_t info;

  struct cr_gpu_buffer_t tile_buf, 
                         segment_indices_buf, 
                         prefix_parity_buf, 
                         prefix_sg_tmp;

  struct cr_compute_pipeline_dynamic_state_t* dynamic;

  struct cr_storage_image_t storage_img;
};

bool cr_compute_pipeline_init(struct cr_context_t* ctx, struct cr_compute_pipeline_init_info_t* info, struct cr_compute_pipeline_t* pipeline);

bool cr_compute_pipeline_resize(struct cr_context_t* ctx, struct cr_compute_pipeline_t* pipeline, uint32_t w, uint32_t h);

bool cr_compute_pipeline_dispatch(struct cr_context_t* ctx, struct cr_compute_pipeline_t* pipeline, uint32_t frame_idx,
    uint32_t swapchain_image_idx);

bool cr_compute_pipeline_get_internal_shader_paths(struct cr_context_t* ctx, const char* subpath, char*** o_paths, uint32_t* o_n_paths); 

bool cr_compute_pipeline_insert_segment(struct cr_context_t* ctx, struct cr_compute_pipeline_t* pipeline,
    struct cr_segment_t segment, uint32_t swapchain_idx);
