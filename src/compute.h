#pragma once


#include <stdint.h>
#include <stdbool.h>
#include <vulkan/vulkan_core.h>
#include "../include/corender/corender.h"
#include "mem.h"

struct cr_tile_header_t {
  uint32_t n_segments;
};

struct cr_segment_range_t {
    uint min_xy; // x | y << 16
    uint max_xy; // x | y << 16
};

struct cr_compute_pipeline_push_constant_t {
  uint32_t screen_w,  screen_h;
  uint32_t n_tiles_x, n_tiles_y, n_tiles;

  uint32_t tile_size; 

  uint32_t max_tile_storage;
  
  uint32_t n_segments;
  uint32_t n_paths;
};

struct cr_compute_bump_t {
    uint32_t n_touches;
    uint32_t n_active_tiles;
    uint32_t n_tile_segment_slots;
    uint32_t failed;
};

struct cr_compute_tile_info_t {
    uint32_t base;
    uint32_t count;
    int32_t winding;
    uint32_t flags;
};

struct cr_compute_tile_edge_t {
    uint32_t p0;   // x: low 16, y: high 16, tile-local 8.8 fixed
    uint32_t p1;   // x: low 16, y: high 16, tile-local 8.8 fixed
    uint32_t meta; // bit 0 valid, bit 1 negative winding delta
    uint32_t y_edge;
};

struct cr_compute_tile_touch_record_t {
    uint32_t seg_id;
    uint32_t pack;
};

struct cr_compute_indirect_dispatch_cmd_t {
    uint32_t dispatch_x;
    uint32_t dispatch_y;
    uint32_t dispatch_z;
};


enum cr_compute_pipeline_fill_rule_t {
  CR_COMPUTE_FILL_RULE_EVEN_ODD = 0,
  CR_COMPUTE_FILL_RULE_NON_ZERO
};

struct cr_compute_macrotile_metadata_t {
    uint32_t offset, count;
};

struct cr_compute_draw_path_t {
    uint32_t segment_offset;
    uint32_t segment_count;

    vec2 min, max;

    uint32_t id;
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

struct cr_compute_pipeline_layout_binding_t {
  const char* name;
  size_t buffer_size;
};

struct cr_compute_pipeline_layout_buffer_t {
  const char* name;
  uint32_t hash;
  size_t buffer_size;
  struct cr_gpu_buffer_t buf;
};

struct cr_compute_pipeline_dynamic_state_t {
  struct cr_gpu_buffer_t segment_buf, path_buf, indirect_buf; 
  uint32_t n_segments, n_paths;

  struct cr_segment_t* segment_data;
  struct cr_compute_draw_path_t* path_data;
  struct cr_compute_indirect_dispatch_cmd_t* indirect_data;

  size_t segment_capacity;
  size_t path_capacity;

  size_t n_segments_in_path;

  struct cr_compute_pipeline_layout_buffer_t* buffers;
  uint32_t n_buffers;
};

struct cr_compute_kernel_t {
  char* shader_path;
  uint32_t hash;
  VkPipeline kernel_pipeline;
};

struct cr_compute_pipeline_t {
  struct cr_compute_kernel_t* kernels;

  struct cr_compute_pipeline_init_info_t info;

  struct cr_compute_pipeline_dynamic_state_t* dynamic;

  struct cr_storage_image_t storage_img;

  VkDescriptorPool descriptor_pool;

  uint32_t n_buffers;
};

bool cr_compute_pipeline_init(struct cr_context_t* ctx, struct cr_compute_pipeline_init_info_t* info, struct cr_compute_pipeline_t* pipeline);

bool cr_compute_pipeline_resize(struct cr_context_t* ctx, struct cr_compute_pipeline_t* pipeline, uint32_t w, uint32_t h);

bool cr_compute_pipeline_dispatch(struct cr_context_t* ctx, struct cr_compute_pipeline_t* pipeline, uint32_t frame_idx,
    uint32_t swapchain_image_idx);

bool cr_compute_pipeline_get_internal_shader_paths(struct cr_context_t* ctx, const char* subpath, char*** o_paths, uint32_t* o_n_paths); 

bool cr_compute_pipeline_insert_segment(struct cr_context_t* ctx, struct cr_compute_pipeline_t* pipeline,
    struct cr_segment_t segment, uint32_t swapchain_idx);

bool cr_compute_pipeline_insert_path(struct cr_context_t* ctx, struct cr_compute_pipeline_t* pipeline,
    struct cr_compute_draw_path_t path, uint32_t swapchain_idx);

bool cr_compute_pipeline_begin_path(struct cr_context_t* ctx, struct cr_compute_pipeline_t* pipeline, uint32_t swapchain_idx);

bool cr_compute_pipeline_end_path(struct cr_context_t* ctx, struct cr_compute_pipeline_t* pipeline, uint32_t swapchain_idx);
