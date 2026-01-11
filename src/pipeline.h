#pragma once
#include <vulkan/vulkan_core.h>

#include "util.h"
#include "mem.h"

struct cr_pipeline_input_state_t {
  VkVertexInputBindingDescription binding_desc[CR_MAX_BINDING_DESC];
  uint32_t n_binding_descs;
  VkVertexInputAttributeDescription vert_attrs[CR_MAX_VERT_ATTRS];
  uint32_t n_vert_attr;
  VkPipelineVertexInputStateCreateInfo input_state;
};

struct cr_pipeline_batch_state_t {
  void* data;

  uint32_t n_elements, write_offset;
           size_t element_stride;
  uint32_t element_cap, _element_cap_cpu;

  struct cr_gpu_buffer_t gpu_buffer; 

  void* emitted_draws;
  uint32_t n_emitted_draws, emitted_draws_cap, _emitted_draws_cap_cpu;
  
  struct cr_gpu_buffer_t _indirect_buffer;
};

struct cr_pipeline_push_constant_t {
  vec2 scale, offset;
};

struct cr_pipeline_t {
  VkPipeline pipeline;

  struct cr_pipeline_input_state_t input;
  struct cr_pipeline_batch_state_t batching[CR_FRAME_COUNT];

  
  struct cr_gpu_buffer_t static_buffers[CR_MAX_DYNAMIC_BUFS]; 
  uint32_t n_static_buffers;

  uint32_t indices_per_instance; 
  uint32_t vertices_per_instance;
  bool use_device_local_buffer;

  uint32_t _total_elements_uploaded;
};

struct cr_pipeline_init_info_t {
    const char* vertex_path; 
    const char* fragment_path; 
    uint32_t elements_per_batch; 
    size_t batch_element_size; 

    uint32_t indices_per_instance; 
    uint32_t vertices_per_instance;

    bool use_device_local_buffer;
};

bool cr_pipeline_init(struct cr_context_t* ctx, 
    struct cr_pipeline_t* o_pipeline, 
    const struct cr_pipeline_init_info_t* info
    );

bool cr_pipeline_add_binding_desc(struct cr_pipeline_t* pipeline, 
    VkVertexInputBindingDescription binding_desc);

bool cr_pipeline_add_vertex_input_attribute(
    struct cr_pipeline_t* pipeline, 
    VkVertexInputAttributeDescription vert_attr);

bool cr_pipeline_get_internal_shader_paths(const char* subpath, char** o_vertex_path, char** o_fragment_path);

bool cr_pipeline_add_static_buffer(struct cr_context_t* ctx, struct cr_pipeline_t* pipeline, void* data, size_t size, enum cr_gpu_buffer_type_t type);

bool cr_pipeline_batching_allocate_buffer(struct cr_context_t* ctx, struct cr_pipeline_t* pipeline,size_t size, enum cr_gpu_buffer_type_t type);

bool cr_pipeline_batching_write_to_batch(struct cr_context_t* ctx, struct cr_pipeline_t* pipeline, const void* element, uint32_t frame_idx);

bool cr_pipeline_batching_flush(struct cr_context_t* ctx, struct cr_pipeline_t* pipeline, uint32_t frame_idx); 

bool cr_pipeline_batching_begin(struct cr_context_t* ctx, struct cr_pipeline_t* pipeline, uint32_t frame_idx);

bool cr_pipeline_batching_upload(struct cr_context_t* ctx, struct cr_pipeline_t* pipeline, uint32_t frame_idx);

bool cr_pipeline_batching_commit(struct cr_context_t* ctx, struct cr_pipeline_t* pipeline, uint32_t frame_idx);

uint32_t cr_pipeline_batching_get_write_idx(struct cr_pipeline_t* pipeline, uint32_t frame_idx);

bool cr_pipeline_batching_ensure_batch_size(struct 
    cr_context_t* ctx, struct cr_pipeline_t* pipeline, uint32_t write_idx, uint32_t frame_idx);
