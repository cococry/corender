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

  uint32_t n_elements, write_offset, element_stride;
  uint32_t element_cap, _element_cap_cpu;

  struct cr_gpu_buffer_t dynamic_buffers[CR_MAX_DYNAMIC_BUFS];
  uint32_t n_dynamic_buffers;

  void* emitted_draws;
  uint32_t n_emitted_draws, emitted_draws_cap;
  
  struct cr_gpu_buffer_t _indirect_buffer;
  uint32_t _indirect_buffer_cap;
};

struct cr_pipeline_t {
  VkPipeline pipeline;

  struct cr_pipeline_input_state_t input;
  struct cr_pipeline_batch_state_t batching[CR_FRAME_COUNT];

  
  struct cr_gpu_buffer_t static_buffers[CR_MAX_DYNAMIC_BUFS]; 
  uint32_t n_static_buffers;

  bool draw_indexed; 
};

bool cr_pipeline_init(struct cr_context_t* ctx, const char* vertex_path, const char* fragment_path, struct cr_pipeline_t* pipeline, uint32_t elements_per_batch, size_t batch_element_size, bool draw_indexed);

bool cr_pipeline_add_binding_desc(struct cr_pipeline_t* pipeline, 
    VkVertexInputBindingDescription binding_desc);

bool cr_pipeline_add_vertex_input_attribute(
    struct cr_pipeline_t* pipeline, 
    VkVertexInputAttributeDescription vert_attr);

bool cr_pipeline_get_internal_shader_paths(const char* subpath, char* o_vertex_path, char* o_fragment_path);

bool cr_pipeline_add_static_buffer(struct cr_context_t* ctx, struct cr_pipeline_t* pipeline, void* data, size_t size, enum cr_gpu_buffer_type_t type);


bool cr_pipeline_add_dynamic_buffer(struct cr_context_t* ctx, struct cr_pipeline_t* pipeline,size_t size, enum cr_gpu_buffer_type_t type);


bool cr_pipeline_flush(struct cr_context_t* ctx, struct cr_pipeline_t* pipeline,
    uint32_t n_instances, uint32_t n_indices); 

uint32_t cr_pipeline_get_batch_write_idx(struct cr_pipeline_t* pipeline);

bool cr_pipeline_ensure_batch_data_size(struct 
    cr_context_t* ctx, struct cr_pipeline_t* pipeline, uint32_t write_idx);
