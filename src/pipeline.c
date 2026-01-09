#include "pipeline.h"
#include "../vendor/vma/vk_mem_alloc.h"
#include "mem.h"
#include "util.h"
#include <pipeline.h>
#include <string.h>
#include <linux/limits.h>
#include <stdbool.h>
#include <vulkan/vulkan_core.h>

#define _SUBSYS_NAME "PIPELINE"

#define _PARAM_CHECK_FAIL()                                               \
  do {                                                                    \
    fprintf(stderr, "corender: Fatal: Did not pass parameter check.");    \
    exit(1);                                                              \
  } while (0);                                                            \


static bool     
_vk_create_pipeline(
  struct cr_context_t* ctx, VkPipelineVertexInputStateCreateInfo vertex_input_state,
  const char* vertex_path, const char* fragment_path,  VkPipeline* o_pipeline);

bool 
_create_shader_module(struct 
  cr_context_t* ctx, const char* filepath, VkShaderModule* o_module);

bool 
_create_shader_module(struct 
  cr_context_t* ctx, const char* filepath, VkShaderModule* o_module) {
  size_t file_size;
  unsigned char* file_data = cr_util_read_file(filepath, &file_size);
  if(!file_data || !file_size) goto err; 

  VkShaderModuleCreateInfo shader_info = {0};
  shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shader_info.codeSize = file_size;
  shader_info.pCode = (const uint32_t*)file_data;

  _VK_CHECK(ctx, vkCreateShaderModule(ctx->logical_dev, &shader_info, NULL, o_module));

  CR_TRACE(ctx->log, "Successfully created Vulkan shader module for file '%s'", filepath)

  free(file_data);

  return true;

err:
  return false;
}

bool 
_vk_create_pipeline(
  struct cr_context_t* ctx, VkPipelineVertexInputStateCreateInfo vertex_input_state,
  const char* vertex_path, const char* fragment_path, VkPipeline* o_pipeline) {
  VkShaderModule vert_mod, frag_mod;

  if(!_create_shader_module(ctx, vertex_path, &vert_mod)) {
    CR_ERROR(ctx->log, "Failed to create vertex shader byte code for file '%s'", 
             vertex_path);
    goto err;
  }
  if(!_create_shader_module(ctx, fragment_path, &frag_mod)) {
    CR_ERROR(ctx->log, "Failed to create fragment shader byte code for file '%s'", 
             fragment_path);
    goto err;
  }

  VkPipelineShaderStageCreateInfo shader_stages[2] = {
    (VkPipelineShaderStageCreateInfo){
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_VERTEX_BIT,
      .pName = "main",
      .module = vert_mod,
    },
    (VkPipelineShaderStageCreateInfo){
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
      .pName = "main",
      .module = frag_mod,
    }
  };


  VkPipelineInputAssemblyStateCreateInfo assmebly_state = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
    .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
  };

  VkPipelineRasterizationStateCreateInfo raster_state = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
    .polygonMode = VK_POLYGON_MODE_FILL,
    .cullMode = VK_CULL_MODE_NONE,
    .frontFace = VK_FRONT_FACE_CLOCKWISE,
    .lineWidth = 1.0f,
  };


  VkPipelineMultisampleStateCreateInfo msaa_state = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
    .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT
  };

  VkPipelineColorBlendAttachmentState blend = {
    .blendEnable = VK_TRUE,
    .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
    .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
    .colorBlendOp = VK_BLEND_OP_ADD,
    .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
    .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
    .alphaBlendOp = VK_BLEND_OP_ADD,
    .colorWriteMask = 0xF
  };

  VkPipelineColorBlendStateCreateInfo blend_state = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
    .attachmentCount = 1,
    .pAttachments = &blend
  };

  VkDynamicState dynamic_states[] = {
    VK_DYNAMIC_STATE_VIEWPORT,
    VK_DYNAMIC_STATE_SCISSOR
  };

  VkPipelineDynamicStateCreateInfo dynamic_state = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
    .dynamicStateCount = 2,
    .pDynamicStates = dynamic_states
  };

  VkPipelineViewportStateCreateInfo viewport_state = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
    .viewportCount = 1,
    .pViewports = NULL,
    .scissorCount = 1,
    .pScissors = NULL,
  };

  VkGraphicsPipelineCreateInfo pipeline_info = {
    .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
    .stageCount = 2,
    .pStages = shader_stages,
    .pVertexInputState = &vertex_input_state,
    .pInputAssemblyState = &assmebly_state,
    .pColorBlendState = &blend_state,
    .pMultisampleState = &msaa_state,
    .pRasterizationState = &raster_state,
    .pDynamicState = &dynamic_state,
    .pViewportState = &viewport_state,
    .layout = ctx->pipeline_layout,
    .renderPass = ctx->frameloop.crnt_pass,
    .subpass = 0
  };

  _VK_CHECK(ctx, vkCreateGraphicsPipelines(ctx->logical_dev, VK_NULL_HANDLE, 1, &pipeline_info, NULL, o_pipeline));

  vkDestroyShaderModule(ctx->logical_dev, vert_mod, NULL);
  vkDestroyShaderModule(ctx->logical_dev, frag_mod, NULL);

  CR_TRACE(ctx->log, "Initialized graphics pipeline for surface %p (vertex shader: %s, fragment shader: %s)", ctx->surf.surf,
           vertex_path, fragment_path);


  return true;
err:
  return false;
}

bool 
cr_pipeline_get_internal_shader_paths(const char* subpath, char* o_vertex_path, char* o_fragment_path) {
  const char* state_dir = cr_util_get_state_folder();
  char shader_dir[PATH_MAX];
  snprintf(shader_dir, sizeof(shader_dir), "%s/%s/shaders/%s", state_dir, _CR_BRAND_NAME, subpath);

  char* vert_src = malloc(PATH_MAX);
  if(!vert_src) return false;
  snprintf(vert_src, PATH_MAX, "%s/basic_vert.spv", shader_dir);
  char* frag_src = malloc(PATH_MAX);
  if(!frag_src) return false;
  snprintf(frag_src, PATH_MAX, "%s/basic_frag.spv", shader_dir);

  o_vertex_path = vert_src;
  o_fragment_path = frag_src;

  return true;
}



bool 
cr_pipeline_init(struct cr_context_t* ctx, const char* vertex_path, const char* fragment_path, struct cr_pipeline_t* pipeline, uint32_t elements_per_batch, size_t batch_element_size, bool draw_indexed) {
  if(!pipeline || !fragment_path || !vertex_path) _PARAM_CHECK_FAIL();


  bool indirect_draw_size = draw_indexed ? sizeof(VkDrawIndexedIndirectCommand) : sizeof(VkDrawIndirectCommand);



  pipeline->input.input_state = (VkPipelineVertexInputStateCreateInfo){
    .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    .vertexBindingDescriptionCount = pipeline->input.n_binding_descs,
    .pVertexBindingDescriptions = pipeline->input.binding_desc,
    .vertexAttributeDescriptionCount = pipeline->input.n_vert_attr,
    .pVertexAttributeDescriptions = pipeline->input.vert_attrs
  };



  if(!_vk_create_pipeline(ctx, pipeline->input.input_state, 
                      vertex_path, fragment_path, &pipeline->pipeline)) {
    CR_ERROR(ctx->log, "Failed to create Vulkan Pipeline (fragment shader: %s, vertex shader: %s)", fragment_path, vertex_path);
    goto err;
  }
    
  CR_TRACE(ctx->log, "Created Vulkan Pipeline (fragment shader: %s, vertex shader: %s)", fragment_path, vertex_path);

  for(uint32_t i = 0; i < CR_FRAME_COUNT; i++) {
    struct cr_pipeline_batch_state_t* batching = &pipeline->batching[i];
    batching->_indirect_buffer_cap = CR_INITIAL_INDIRECT_DRAW_CAP;
    cr_mem_create_gpu_buffer(
      ctx, 
      indirect_draw_size * batching->_indirect_buffer_cap, CR_GPU_BUFFER_INDIRECT, 
      CR_GPU_BUFFER_MEM_DEVICE_LOCAL, &batching->_indirect_buffer);


    batching->emitted_draws = cr_util_alloc(
      ctx, batching->_indirect_buffer_cap, indirect_draw_size); 

    batching->element_cap = elements_per_batch * CR_INITIAL_BATCH_CAP;
    batching->element_stride = batch_element_size; 
    batching->data = cr_util_alloc(ctx, elements_per_batch, batch_element_size);
  }

  pipeline->draw_indexed = draw_indexed; 

  return true;

err:
  return false;
}

bool 
cr_pipeline_add_binding_desc(struct cr_pipeline_t* pipeline, 
    VkVertexInputBindingDescription binding_desc) {
  if(pipeline->input.n_binding_descs >= CR_MAX_BINDING_DESC) return false;
  pipeline->input.binding_desc[pipeline->input.n_binding_descs++] = binding_desc;

  return true;
}

bool
cr_pipeline_add_vertex_input_attribute(
    struct cr_pipeline_t* pipeline, 
    VkVertexInputAttributeDescription vert_attr) {
  if(pipeline->input.n_vert_attr >= CR_MAX_VERT_ATTRS) return false;
  pipeline->input.vert_attrs[pipeline->input.n_vert_attr++] = vert_attr;

  return true;
}

bool cr_pipeline_add_static_buffer(struct cr_context_t* ctx, struct cr_pipeline_t* pipeline, void* data, size_t size, enum cr_gpu_buffer_type_t type) {

  if(pipeline->n_static_buffers >= CR_MAX_STATIC_BUFS) return false;
  if(!cr_mem_upload_to_device_local_gpu_buffer(
    ctx, data, size, type, &pipeline->static_buffers[pipeline->n_static_buffers++])) {
    CR_ERROR(ctx->log, "Failed to upload static buffer to device local GPU buffer.");
    return false;
  }

  return true;
}

bool 
cr_pipeline_add_dynamic_buffer(struct cr_context_t* ctx, struct cr_pipeline_t* pipeline,size_t size, enum cr_gpu_buffer_type_t type) {

  for(uint32_t i = 0; i < CR_MAX_BATCH; i++) {
    if(pipeline->batching[i].n_dynamic_buffers >= CR_MAX_DYNAMIC_BUFS) return false;

    if(!cr_mem_create_gpu_buffer(
      ctx, size, 
      type,
      CR_GPU_BUFFER_MEM_DEVICE_LOCAL,
      &pipeline->batching[i].dynamic_buffers[pipeline->batching[i].n_dynamic_buffers++])) {
      CR_ERROR(ctx->log, "Failed to create dynamic GPU Buffer.");
      return false;
    }
  }

  return true;
}

bool 
cr_pipeline_flush(struct cr_context_t* ctx, struct cr_pipeline_t* pipeline,
                  uint32_t n_instances, uint32_t n_indices){ 
  if(ctx->_skip_render) return true;
  struct cr_frame_t* frame = &ctx->frameloop.frames[ctx->frameloop.frame_idx];
  
  struct cr_pipeline_batch_state_t* batch = &pipeline->batching[ctx->frameloop.frame_idx];

  bool indirect_draw_size = pipeline->draw_indexed ? sizeof(VkDrawIndexedIndirectCommand) : sizeof(VkDrawIndirectCommand);

  if(batch->n_emitted_draws >= batch->emitted_draws_cap) {
    batch->emitted_draws_cap *= 2;
    batch->emitted_draws = realloc(batch->emitted_draws,
                                     batch->emitted_draws_cap * 
                                       indirect_draw_size);

    if(!frame->_indirect_cmds_instanced) {
      CR_ERROR(ctx->log, "Failed to reallocate CPU Indirect Command Buffer.");
      goto err;
    }
  }

  if(pipeline->draw_indexed) {
    VkDrawIndexedIndirectCommand* draws =  (VkDrawIndexedIndirectCommand*)
      batch->emitted_draws;

    draws[batch->n_emitted_draws++] = (VkDrawIndexedIndirectCommand){
      .indexCount    = n_indices,
      .instanceCount = n_instances, 
      .firstIndex    = n_instances > 1 ? 0 : batch->write_offset,
      .vertexOffset  = 0,
      .firstInstance = n_instances > 1 ? batch->write_offset : 0,
    };
  } else {
    VkDrawIndirectCommand* draws =  (VkDrawIndirectCommand*)
      batch->emitted_draws;
    draws[batch->n_emitted_draws++] = (
        VkDrawIndirectCommand
      ) {
        .instanceCount = n_instances,
        .firstInstance = n_instances > 1 ? batch->write_offset : 0,
        .firstVertex = n_instances > 1 ? 0 : batch->write_offset,
        .vertexCount = batch->n_elements 
      };
  }

  batch->write_offset += batch->n_elements;
  batch->n_elements = 0;

  return true;

err: 
  return false;

}


uint32_t 
cr_pipeline_get_batch_write_idx(struct cr_pipeline_t* pipeline) {
  return pipeline->batching->write_offset + pipeline->batching->n_elements;
}

bool 
cr_pipeline_ensure_batch_data_size(
  struct cr_context_t* ctx, 
  struct cr_pipeline_t* pipeline,
uint32_t write_idx) {
  if(write_idx >= pipeline->batching->_element_cap_cpu) {
    pipeline->batching->_element_cap_cpu = 
      CR_MAX(pipeline->batching->_element_cap_cpu * 2,
             write_idx);

    void* new = realloc(pipeline->batching->data,
                        pipeline->batching->_element_cap_cpu * pipeline->batching->element_stride);
    if(!new) {
      CR_ERROR(ctx->log, "Failed to resize CPU buffer for instance data.\n");
      return false;
    }
    pipeline->batching->data  = new;
  }

  return true;
}
