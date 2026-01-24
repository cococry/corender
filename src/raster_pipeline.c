#include "raster_pipeline.h"
#include "../vendor/vma/vk_mem_alloc.h"
#include "mem.h"
#include "util.h"
#include "raster_pipeline.h"

#include <string.h>
#include <linux/limits.h>
#include <stdbool.h>
#include <vulkan/vulkan_core.h>
#include <assert.h>


#define _SUBSYS_NAME "RASTER"

static void _vk_render_set_dynamic_state(struct cr_context_t* ctx);

static bool _create_shader_module(struct 
                                  cr_context_t* ctx, const char* filepath, VkShaderModule* o_module);

static bool 
_create_shader_module(struct 
                      cr_context_t* ctx, const char* filepath, VkShaderModule* o_module);

static bool     
_vk_create_raster_pipeline(
  struct cr_context_t* ctx, VkPipelineVertexInputStateCreateInfo vertex_input_state,
  const char* vertex_path, const char* fragment_path,  VkPipeline* o_raster_pipeline);

void _vk_render_set_dynamic_state(struct cr_context_t* ctx) {
  if(!ctx) _PARAM_CHECK_FAIL();

  struct cr_frame_t* frame = &ctx->frameloop.frames[ctx->frameloop.frame_idx];
  VkViewport viewport = {
    .x = 0.0f,
    .y = 0.0f,
    .width = (float)ctx->swapchain.dimensions.width,
    .height = (float)ctx->swapchain.dimensions.height,
    .minDepth = 0.0f,
    .maxDepth = 1.0f
  };

  vkCmdSetViewport(frame->cmd_buf, 0, 1, &viewport);

  VkRect2D scissor_rect = {
    .offset = {0, 0},
    .extent = ctx->swapchain.dimensions
  };

  vkCmdSetScissor(frame->cmd_buf, 0, 1, &scissor_rect);

  struct cr_raster_pipeline_push_constant_t pc = {
    .scale = { 2.0f / ctx->swapchain.dimensions.width,  2.0f / ctx->swapchain.dimensions.height},
    .offset = { -1.0f, -1.0f},
  };

  vkCmdPushConstants(
    frame->cmd_buf, 
    ctx->pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc); 
}


bool 
_create_shader_module(struct 
                      cr_context_t* ctx, const char* filepath, VkShaderModule* o_module);

bool 

_create_shader_module(struct 
                      cr_context_t* ctx, const char* filepath, VkShaderModule* o_module) {
  if(!ctx || !filepath || !o_module) _PARAM_CHECK_FAIL();
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
_vk_create_raster_pipeline(
  struct cr_context_t* ctx, VkPipelineVertexInputStateCreateInfo vertex_input_state,
  const char* vertex_path, const char* fragment_path, VkPipeline* o_raster_pipeline) {
  if(!ctx || !vertex_path || !fragment_path || !o_raster_pipeline) _PARAM_CHECK_FAIL();

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
    .blendEnable = VK_FALSE,
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

  VkPipelineDepthStencilStateCreateInfo depth_state = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
    .depthTestEnable = VK_FALSE,
    .depthWriteEnable = VK_FALSE,
    .depthCompareOp = VK_COMPARE_OP_LESS,
    .depthBoundsTestEnable = VK_FALSE,
    .stencilTestEnable = VK_FALSE
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
    .pDepthStencilState = &depth_state,
    .layout = ctx->pipeline_layout,
    .renderPass = ctx->frameloop.crnt_pass,
    .subpass = 0
  };

  _VK_CHECK(ctx, vkCreateGraphicsPipelines(ctx->logical_dev, VK_NULL_HANDLE, 1, &pipeline_info, NULL, o_raster_pipeline));

  vkDestroyShaderModule(ctx->logical_dev, vert_mod, NULL);
  vkDestroyShaderModule(ctx->logical_dev, frag_mod, NULL);

  CR_TRACE(ctx->log, "Initialized graphics pipeline for surface %p (vertex shader: %s, fragment shader: %s)", ctx->surf.surf,
           vertex_path, fragment_path);


  return true;
err:
  return false;
}

bool 
cr_raster_pipeline_get_internal_shader_paths(const char* subpath, char** o_vertex_path, char** o_fragment_path) {
  const char* state_dir = cr_util_get_state_folder();
  char shader_dir[PATH_MAX];
  snprintf(shader_dir, sizeof(shader_dir), "%s/%s/shaders/%s", state_dir, _CR_BRAND_NAME, subpath);

  char* vert_src = malloc(PATH_MAX);
  if(!vert_src) return false;
  snprintf(vert_src, PATH_MAX, "%s/basic_vert.spv", shader_dir);
  char* frag_src = malloc(PATH_MAX);
  if(!frag_src) return false;
  snprintf(frag_src, PATH_MAX, "%s/basic_frag.spv", shader_dir);

  *o_vertex_path = vert_src;
  *o_fragment_path = frag_src;

  return true;
}



bool 
cr_raster_pipeline_init(struct cr_context_t* ctx, 
                 struct cr_raster_pipeline_t* o_raster_pipeline, 
                 const struct cr_raster_pipeline_init_info_t* info
                 ) {
  if(!ctx || !o_raster_pipeline || !info) _PARAM_CHECK_FAIL();

  size_t indirect_draw_size = info->indices_per_instance > 0 ? sizeof(VkDrawIndexedIndirectCommand) : sizeof(VkDrawIndirectCommand);

  o_raster_pipeline->input.input_state = (VkPipelineVertexInputStateCreateInfo){
    .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    .vertexBindingDescriptionCount = o_raster_pipeline->input.n_binding_descs,
    .pVertexBindingDescriptions = o_raster_pipeline->input.binding_desc,
    .vertexAttributeDescriptionCount = o_raster_pipeline->input.n_vert_attr,
    .pVertexAttributeDescriptions = o_raster_pipeline->input.vert_attrs
  };

  if(!_vk_create_raster_pipeline(ctx, o_raster_pipeline->input.input_state, 
                          info->vertex_path, info->fragment_path, &o_raster_pipeline->pipeline)) {
    CR_ERROR(ctx->log, "Failed to create Vulkan Pipeline (fragment shader: %s, vertex shader: %s)", info->fragment_path, info->vertex_path);
    goto err;
  }

  CR_TRACE(ctx->log, "Created Vulkan Pipeline (fragment shader: %s, vertex shader: %s)", info->fragment_path, info->vertex_path);

  for(uint32_t i = 0; i < CR_FRAME_COUNT; i++) {
    struct cr_raster_pipeline_batch_state_t* batching = &o_raster_pipeline->batching[i];
    batching->emitted_draws_cap = CR_INITIAL_INDIRECT_DRAW_CAP;
    batching->_emitted_draws_cap_cpu = CR_INITIAL_INDIRECT_DRAW_CAP;

    cr_mem_create_gpu_buffer(
      ctx, 
      indirect_draw_size * batching->emitted_draws_cap, CR_GPU_BUFFER_INDIRECT, 
      CR_GPU_BUFFER_MEM_DEVICE_LOCAL, &batching->_indirect_buffer);

    batching->emitted_draws = cr_util_alloc(
      ctx, batching->emitted_draws_cap, indirect_draw_size); 

    batching->element_cap = info->elements_per_batch * CR_INITIAL_BATCH_CAP;
    batching->_element_cap_cpu = info->elements_per_batch * CR_INITIAL_BATCH_CAP;

    batching->element_stride = info->batch_element_size; 
    if(info->use_device_local_buffer) {
      batching->data = cr_util_alloc(ctx, batching->element_cap, info->batch_element_size);
    }
  }

  o_raster_pipeline->vertices_per_instance = info->vertices_per_instance; 
  o_raster_pipeline->indices_per_instance = info->indices_per_instance;
  o_raster_pipeline->use_device_local_buffer = info->use_device_local_buffer;

  return true;

err:
  return false;
}

bool 
cr_raster_pipeline_add_binding_desc(struct cr_raster_pipeline_t* pipeline, 
                             VkVertexInputBindingDescription binding_desc) {
  if(!pipeline) _PARAM_CHECK_FAIL();

  if(pipeline->input.n_binding_descs >= CR_MAX_BINDING_DESC) return false;
  pipeline->input.binding_desc[pipeline->input.n_binding_descs++] = binding_desc;

  return true;
}

bool
cr_raster_pipeline_add_vertex_input_attribute(
  struct cr_raster_pipeline_t* pipeline, 
  VkVertexInputAttributeDescription vert_attr) {
  if(!pipeline) _PARAM_CHECK_FAIL();

  if(pipeline->input.n_vert_attr >= CR_MAX_VERT_ATTRS) return false;
  pipeline->input.vert_attrs[pipeline->input.n_vert_attr++] = vert_attr;

  return true;
}

bool cr_raster_pipeline_add_static_buffer(struct cr_context_t* ctx, struct cr_raster_pipeline_t* pipeline, void* data, size_t size, enum cr_gpu_buffer_type_t type) {
  if(!pipeline || !ctx) _PARAM_CHECK_FAIL();

  if(pipeline->n_static_buffers >= CR_MAX_STATIC_BUFS) return false;
  if(!cr_mem_upload_to_device_local_gpu_buffer(
    ctx, data, size, type, &pipeline->static_buffers[pipeline->n_static_buffers++])) {
    CR_ERROR(ctx->log, "Failed to upload static buffer to device local GPU buffer.");
    return false;
  }

  return true;
}

bool 
cr_raster_pipeline_batching_allocate_buffer(struct cr_context_t* ctx, struct cr_raster_pipeline_t* pipeline,size_t size, enum cr_gpu_buffer_type_t type) {
  if(!ctx|| !pipeline) _PARAM_CHECK_FAIL();
  for(uint32_t i = 0; i < CR_FRAME_COUNT; i++) {
    struct cr_raster_pipeline_batch_state_t* batching = &pipeline->batching[i];
    if(batching->gpu_buffer.buf) continue;
    if(!cr_mem_create_gpu_buffer(
      ctx, size, 
      type,
      pipeline->use_device_local_buffer ? CR_GPU_BUFFER_MEM_DEVICE_LOCAL : CR_GPU_BUFFER_MEM_MAPPED,
      &batching->gpu_buffer)) {
      CR_ERROR(ctx->log, "Failed to create dynamic GPU Buffer.");
      return false;
    }
    if(!pipeline->use_device_local_buffer) {
      batching->data = batching->gpu_buffer.mem_handle;
    }
  }

  return true;
}

bool 
cr_raster_pipeline_batching_begin(struct cr_context_t* ctx, struct cr_raster_pipeline_t* pipeline, uint32_t frame_idx) {
  (void)ctx;
  if(!pipeline) _PARAM_CHECK_FAIL();
  if(frame_idx >= CR_FRAME_COUNT) return false;

  struct cr_raster_pipeline_batch_state_t* batching = &pipeline->batching[frame_idx];

  batching->n_elements = 0;
  batching->write_offset = 0;

  return true;
}

bool 
cr_raster_pipeline_batching_upload(struct cr_context_t* ctx, struct cr_raster_pipeline_t* pipeline, uint32_t frame_idx) {
  if(!ctx || !pipeline) _PARAM_CHECK_FAIL();
  if(frame_idx >= CR_FRAME_COUNT) return false;

  uint32_t total_elements = cr_raster_pipeline_batching_get_write_idx(pipeline, frame_idx); 
  struct cr_raster_pipeline_batch_state_t* batching = &pipeline->batching[frame_idx];

  if(total_elements > 0) {
    if(!cr_raster_pipeline_batching_flush(ctx, pipeline, frame_idx)) {
      CR_ERROR(ctx->log, "Failed to flush the renderer.");
      return false;
    }

    if(pipeline->use_device_local_buffer) {
      if(total_elements > batching->element_cap) {
        batching->element_cap = CR_MAX(batching->element_cap * 2,
                                       total_elements);

        if(!cr_mem_resize_gpu_buffer(ctx, &batching->gpu_buffer, batching->element_cap * batching->element_stride)) {
          CR_ERROR(ctx->log, "Failed to resize batching buffer.");
          return false;
        }
      }

      cr_mem_transfer_to_device_local_gpu_buffer(
        ctx,
        &ctx->frameloop.frames[frame_idx],
        batching->data,
        total_elements * batching->element_stride,
        &batching->gpu_buffer
      );

    }
    pipeline->_total_elements_uploaded = total_elements;
  }

  if(batching->n_emitted_draws > 0) {

    size_t indirect_draw_size = pipeline->indices_per_instance > 0 ? sizeof(VkDrawIndexedIndirectCommand) : sizeof(VkDrawIndirectCommand);
    if(batching->n_emitted_draws > batching->emitted_draws_cap) {
      batching->emitted_draws_cap = CR_MAX(batching->emitted_draws_cap * 2,
                                           batching->n_emitted_draws);

      if(!cr_mem_resize_gpu_buffer(ctx, &batching->_indirect_buffer, batching->emitted_draws_cap * indirect_draw_size)) {

        CR_ERROR(ctx->log, "Failed to resize indirect drawing buffer.");
        return false;
      }
    }
    cr_mem_transfer_to_device_local_gpu_buffer(
      ctx,
      &ctx->frameloop.frames[frame_idx],
      batching->emitted_draws,
      batching->n_emitted_draws * indirect_draw_size, 
      &batching->_indirect_buffer
    );
  }

  return true;

}

bool 
cr_raster_pipeline_batching_commit(struct cr_context_t* ctx, struct cr_raster_pipeline_t* pipeline, uint32_t frame_idx) {
  if(!ctx || !pipeline) _PARAM_CHECK_FAIL();
  if(frame_idx >= CR_FRAME_COUNT) return false;

  uint32_t total_elements = pipeline->_total_elements_uploaded;
  struct cr_raster_pipeline_batch_state_t* batching = &pipeline->batching[frame_idx];

  struct cr_frame_t* frame = &ctx->frameloop.frames[frame_idx];

  if(total_elements <= 0) return true;

  vkCmdBindPipeline(frame->cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->pipeline);

  _vk_render_set_dynamic_state(ctx);


  uint32_t n_vbos = batching->gpu_buffer.buf ?  1 : 0;
  for(uint32_t i = 0; i < pipeline->n_static_buffers; i++) {
    if(pipeline->static_buffers[i].type != CR_GPU_BUFFER_VERTEX) continue;
    n_vbos++;
  }
  if(n_vbos != pipeline->input.n_binding_descs) {
    CR_FATAL(ctx->log, "Pipeline vertex buffer count does not match te number of binding descriptions in the pipelines's vertex input.\n");
    return false;
  }

  VkBuffer vbos[n_vbos]; 

  bool instanced_raster_pipeline = pipeline->indices_per_instance > 0 || pipeline->vertices_per_instance > 0;
  for(uint32_t i = 0; i < pipeline->input.n_binding_descs; i++) {
    if(pipeline->input.binding_desc[i].inputRate == VK_VERTEX_INPUT_RATE_VERTEX) {
      bool input_rate_means_static_buffer = pipeline->n_static_buffers > 0 && i < n_vbos && pipeline->static_buffers[i].type == CR_GPU_BUFFER_VERTEX;
      vbos[pipeline->input.binding_desc[i].binding] = input_rate_means_static_buffer ? pipeline->static_buffers[i].buf : batching->gpu_buffer.buf; 
    } 
    else if(pipeline->input.binding_desc[i].inputRate == VK_VERTEX_INPUT_RATE_INSTANCE) {
      if(!instanced_raster_pipeline) {
        CR_FATAL(ctx->log,  "A binding description with VK_VERTEX_INPUT_RATE_INSTANCE "
                 "exists in the pipeline's (%p) input state but "
                 "pipeline->indices_per_instance & pipeline->vertices_per_instance are both 0.", pipeline);
      }
      vbos[pipeline->input.binding_desc[i].binding] = batching->gpu_buffer.buf; 
    }
  }
  VkDeviceSize vbo_offsets[n_vbos];
  memset(vbo_offsets, 0, sizeof(vbo_offsets));

  bool draw_indexed = pipeline->indices_per_instance > 0;

  vkCmdBindVertexBuffers(frame->cmd_buf, 0, n_vbos, vbos, vbo_offsets);

  if(ctx->enable_time_measuring) {
    vkCmdResetQueryPool(frame->cmd_buf, frame->timestamp_pool, 0, 2);

    vkCmdWriteTimestamp(
      frame->cmd_buf,
      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
      frame->timestamp_pool,
      0
    );
  }

  if(draw_indexed) {
    for(uint32_t i = 0; i < pipeline->n_static_buffers; i++) {
      if(pipeline->static_buffers[i].type != CR_GPU_BUFFER_INDEX) continue;
      vkCmdBindIndexBuffer(frame->cmd_buf, pipeline->static_buffers[i].buf, 0, VK_INDEX_TYPE_UINT32);
      break;
    }
    vkCmdDrawIndexedIndirect(
      frame->cmd_buf,
      batching->_indirect_buffer.buf,
      0,
      batching->n_emitted_draws,
      sizeof(VkDrawIndexedIndirectCommand)
    );

  } else {
if (!pipeline->use_device_local_buffer) {
    VkBufferMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = batching->gpu_buffer.buf,
        .offset = 0,
        .size = VK_WHOLE_SIZE
    };

    vkCmdPipelineBarrier(
        frame->cmd_buf,
        VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
        0,
        0, NULL,
        1, &barrier,
        0, NULL
    );
}
    vkCmdDrawIndirect(
      frame->cmd_buf,
      batching->_indirect_buffer.buf,
      0,
      batching->n_emitted_draws,
      sizeof(VkDrawIndirectCommand)
    );
  }

  if(ctx->enable_time_measuring) {
    vkCmdWriteTimestamp(
      frame->cmd_buf,
      VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
      frame->timestamp_pool,
      1
    );
  }


  batching->n_emitted_draws = 0;

  return true;
}

bool 
cr_raster_pipeline_batching_flush(struct cr_context_t* ctx, struct cr_raster_pipeline_t* pipeline,
                           uint32_t frame_idx){ 
  if(!ctx || !pipeline) _PARAM_CHECK_FAIL();
  if(frame_idx >= CR_FRAME_COUNT) return false;

  if(ctx->_skip_render) return true;

  struct cr_raster_pipeline_batch_state_t* batch = &pipeline->batching[frame_idx];

  if(batch->n_emitted_draws >= batch->_emitted_draws_cap_cpu) {
    size_t indirect_draw_size = pipeline->indices_per_instance > 0 ? sizeof(VkDrawIndexedIndirectCommand) : sizeof(VkDrawIndirectCommand);

    batch->_emitted_draws_cap_cpu *= 2;
    batch->emitted_draws = realloc(batch->emitted_draws,
                                   batch->_emitted_draws_cap_cpu * 
                                   indirect_draw_size);

    if(!batch->emitted_draws) {
      CR_ERROR(ctx->log, "Failed to reallocate CPU Indirect Command Buffer.");
      goto err;
    }
  }

  if(pipeline->indices_per_instance > 0) {
    bool instanced_raster_pipeline = pipeline->indices_per_instance > 0;

    VkDrawIndexedIndirectCommand* draws =  (VkDrawIndexedIndirectCommand*)
      batch->emitted_draws;


    draws[batch->n_emitted_draws++] = (VkDrawIndexedIndirectCommand){
      .firstIndex    = instanced_raster_pipeline ? 0 : batch->write_offset, 
      .indexCount    = instanced_raster_pipeline ? pipeline->indices_per_instance : batch->n_elements,

      .firstInstance = instanced_raster_pipeline ? batch->write_offset : 0,
      .instanceCount = instanced_raster_pipeline ? batch->n_elements : 1,
    };

  } else {
    VkDrawIndirectCommand* draws =  (VkDrawIndirectCommand*)
      batch->emitted_draws;

    bool instanced_raster_pipeline = pipeline->vertices_per_instance > 0;

    draws[batch->n_emitted_draws++] = (VkDrawIndirectCommand) {
      .firstVertex = instanced_raster_pipeline ? 0 : batch->write_offset, 
      .vertexCount = instanced_raster_pipeline ? pipeline->vertices_per_instance : batch->n_elements,

      .firstInstance = instanced_raster_pipeline ? batch->write_offset : 0, 
      .instanceCount = instanced_raster_pipeline ? batch->n_elements : 1,
    };

    uint32_t draw_idx = batch->n_emitted_draws - 1;
    VkDrawIndirectCommand* d = &draws[draw_idx];
  }

  batch->write_offset += batch->n_elements;
  batch->n_elements = 0;

  return true;

err: 
  return false;

}

bool 
cr_raster_pipeline_batching_write_to_batch(struct cr_context_t* ctx, struct cr_raster_pipeline_t* pipeline, const void* element, uint32_t frame_idx) {
  if(!ctx || !pipeline || !element) _PARAM_CHECK_FAIL();

  if(frame_idx >= CR_FRAME_COUNT) return false;

  struct cr_raster_pipeline_batch_state_t* batching = &pipeline->batching[frame_idx];

  if(batching->n_elements >= CR_MAX_BATCH) {
    cr_raster_pipeline_batching_flush(ctx, pipeline, frame_idx);
  }

  uint32_t write_idx = cr_raster_pipeline_batching_get_write_idx(pipeline, frame_idx);

  if(!cr_raster_pipeline_batching_ensure_batch_size(ctx, pipeline, write_idx, frame_idx)) return false;

  size_t bytes_needed = ((size_t)write_idx + 1) * batching->element_stride;
  assert(batching->data != NULL);
  assert(bytes_needed <= batching->gpu_buffer.buf_size);
  void* dest = (char*)batching->data + write_idx * batching->element_stride;
  memcpy(dest, element, batching->element_stride);

  batching->n_elements++;

  return true;

}

uint32_t 
cr_raster_pipeline_batching_get_write_idx(struct cr_raster_pipeline_t* pipeline, uint32_t frame_idx) {
  if(!pipeline) _PARAM_CHECK_FAIL();
  if(frame_idx >= CR_FRAME_COUNT) _PARAM_CHECK_FAIL();
  return pipeline->batching[frame_idx].write_offset + pipeline->batching[frame_idx].n_elements;
}

bool 
cr_raster_pipeline_batching_ensure_batch_size(
  struct cr_context_t* ctx, 
  struct cr_raster_pipeline_t* pipeline,
  uint32_t write_idx,
  uint32_t frame_idx) {
  if(!pipeline || !ctx || frame_idx >= CR_FRAME_COUNT) _PARAM_CHECK_FAIL();

  struct cr_raster_pipeline_batch_state_t* batching = &pipeline->batching[frame_idx];
  if(write_idx >= batching->_element_cap_cpu) {
    batching->_element_cap_cpu = 
      CR_MAX(batching->_element_cap_cpu * 2,
             write_idx);

    if(!pipeline->use_device_local_buffer) {
      if(!cr_mem_resize_gpu_buffer(ctx, &batching->gpu_buffer, batching->_element_cap_cpu * batching->element_stride)) {
        CR_ERROR(ctx->log, "Failed to resize batching buffer.");
        return false;
      }
      batching->element_cap = batching->_element_cap_cpu;
      batching->data = batching->gpu_buffer.mem_handle;
    }
    else {
      void* new = realloc(batching->data,
                          batching->_element_cap_cpu * batching->element_stride);
      if(!new) {
        CR_ERROR(ctx->log, "Failed to resize CPU buffer for instance data.\n");
        return false;
      }
      batching->data  = new;
    }
  }

  return true;
}
