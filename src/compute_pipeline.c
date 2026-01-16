#include "compute_pipeline.h"
#include "mem.h"
#include "util.h"
#include <linux/limits.h>
#include <string.h>
#include <vulkan/vulkan_core.h>

#define _SUBSYS_NAME "COMPUTE"

static VkPipelineLayout _compute_pipeline_layout;
static VkDescriptorSetLayout _set_layout;
static VkDescriptorSet* _global_sets;

static bool _create_shader_module(struct  cr_context_t* ctx, const char* filepath, VkShaderModule* o_module);
static bool _vk_create_compute_pipeline(struct cr_context_t* ctx, const char* compute_path, VkPipeline* o_pipeline);
static bool _vk_pipeline_layout_init(struct cr_context_t* ctx, struct cr_compute_pipeline_t* pipeline);

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
_vk_create_compute_pipeline(
  struct cr_context_t* ctx, const char* compute_path, VkPipeline* o_pipeline) {
  if(!ctx || !compute_path) _PARAM_CHECK_FAIL();

  VkShaderModule comp_mod;

  if(!_create_shader_module(ctx, compute_path, &comp_mod)) {
    CR_ERROR(ctx->log, "Failed to create compute shader byte code for file '%s'", compute_path);
    goto err;
  }

  VkPipelineShaderStageCreateInfo stage_info = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    .stage = VK_SHADER_STAGE_COMPUTE_BIT,
    .module = comp_mod,
    .pName = "main"
  };

  VkComputePipelineCreateInfo pipeline_info = {
    .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    .stage = stage_info,
    .layout = _compute_pipeline_layout 
  };

  _VK_CHECK(ctx, vkCreateComputePipelines(
    ctx->logical_dev,
    VK_NULL_HANDLE,
    1,
    &pipeline_info,
    NULL,
    o_pipeline
  ));

  vkDestroyShaderModule(ctx->logical_dev, comp_mod, NULL);

  return true;
err:
  return false;
}

bool _vk_create_storage_image(
    struct cr_context_t* ctx,
    uint32_t width,
    uint32_t height,
    cr_storage_image_t* out_image
) {
    if (!ctx || !out_image) return false;

    VmaAllocator allocator = cr_mem_get_allocator();

    VkImageCreateInfo image_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .extent = {
            .width  = width,
            .height = height,
            .depth  = 1
        },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage =
            VK_IMAGE_USAGE_STORAGE_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT, // useful for clears
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
    };

    VmaAllocationCreateInfo alloc_info = {
        .usage = VMA_MEMORY_USAGE_GPU_ONLY
    };

    VkResult res = vmaCreateImage(
        allocator,
        &image_info,
        &alloc_info,
        &out_image->image,
        &out_image->allocation,
        NULL
    );

    if (res != VK_SUCCESS)
        return false;

    // Create image view
    VkImageViewCreateInfo view_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = out_image->image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        }
    };

    res = vkCreateImageView(
        ctx->logical_dev,
        &view_info,
        NULL,
        &out_image->view
    );

    if (res != VK_SUCCESS) {
        vmaDestroyImage(allocator, out_image->image, out_image->allocation);
        return false;
    }

    out_image->width  = width;
    out_image->height = height;

    return true;
}

bool _vk_destroy_storage_image(struct cr_context_t* ctx, struct cr_storage_image_t* img) {
  vkDestroyImageView(ctx->logical_dev, img->view, NULL);
  vmaDestroyImage(cr_mem_get_allocator(), img->image, img->allocation);
  *img = (struct cr_storage_image_t){0};

  return true;
}

bool 
_vk_pipeline_layout_init(struct cr_context_t* ctx, struct cr_compute_pipeline_t* pipeline) {
  VkPushConstantRange pc_range = {
    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    .offset = 0,
    .size = sizeof(struct cr_compute_pipeline_push_constant_t)
  };

  VkDescriptorSetLayoutBinding bindings[6] = {};

  bindings[0] = (VkDescriptorSetLayoutBinding){
    .binding = 0,
    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    .descriptorCount = 1,
    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
  };
  bindings[1] =  (VkDescriptorSetLayoutBinding){
    .binding = 1,
    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    .descriptorCount = 1,
    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
  };
  bindings[2] =  (VkDescriptorSetLayoutBinding){
    .binding = 2,
    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    .descriptorCount = 1,
    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
  };
  bindings[3] = (VkDescriptorSetLayoutBinding){
    .binding = 3,
    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
    .descriptorCount = 1,
    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
  };
  bindings[4] =  (VkDescriptorSetLayoutBinding){
    .binding = 4,
    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    .descriptorCount = 1,
    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
  };
  bindings[5] =  (VkDescriptorSetLayoutBinding){
    .binding = 5,
    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    .descriptorCount = 1,
    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
  };

  VkDescriptorSetLayoutCreateInfo desc_layout_info = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    .bindingCount = 6,
    .pBindings = bindings
  };

  vkCreateDescriptorSetLayout(ctx->logical_dev, &desc_layout_info, NULL, &_set_layout);

  VkDescriptorPoolSize pool_sizes[2] = {
    {
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 5 * ctx->swapchain.n_imgs 
    },
    {
      .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .descriptorCount = 1 * ctx->swapchain.n_imgs 
    }
  };

  VkDescriptorPoolCreateInfo pool_info = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    .maxSets = ctx->swapchain.n_imgs,
    .poolSizeCount = 2,
    .pPoolSizes = pool_sizes
  };


  VkDescriptorPool descriptor_pool;
  vkCreateDescriptorPool(ctx->logical_dev, &pool_info, NULL, &descriptor_pool);

  VkDescriptorSetLayout layouts[ctx->swapchain.n_imgs];
  for (uint32_t i = 0; i < ctx->swapchain.n_imgs; i++)
    layouts[i] = _set_layout;

  VkDescriptorSetAllocateInfo alloc_info = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    .descriptorPool = descriptor_pool,
    .descriptorSetCount = ctx->swapchain.n_imgs,
    .pSetLayouts = layouts
  };

  vkAllocateDescriptorSets(
    ctx->logical_dev,
    &alloc_info,
    _global_sets
  );
   

  struct cr_compute_pipeline_init_info_t i = pipeline->info;
  uint32_t tiles_x = (i.screen_w + i.tile_size - 1) / i.tile_size;
  uint32_t tiles_y = (i.screen_h + i.tile_size - 1) / i.tile_size;

  cr_mem_create_gpu_buffer(
    ctx, 
    sizeof(struct cr_tile_header_t) * tiles_x * tiles_y, 
    CR_GPU_BUFFER_SSBO, 
    CR_GPU_BUFFER_MEM_DEVICE_LOCAL, &pipeline->tile_buf);

  cr_mem_create_gpu_buffer(
    ctx, 
    sizeof(uint32_t) * tiles_x * tiles_y * 32, 
    CR_GPU_BUFFER_SSBO, 
    CR_GPU_BUFFER_MEM_DEVICE_LOCAL, &pipeline->segment_indices_buf);

  cr_mem_create_gpu_buffer(
    ctx, 
    sizeof(uint32_t) * tiles_x * tiles_y, 
    CR_GPU_BUFFER_SSBO, 
    CR_GPU_BUFFER_MEM_DEVICE_LOCAL, &pipeline->parity_in_buf);
  
  cr_mem_create_gpu_buffer(
    ctx, 
    sizeof(uint32_t) * tiles_x * tiles_y, 
    CR_GPU_BUFFER_SSBO, 
    CR_GPU_BUFFER_MEM_DEVICE_LOCAL, &pipeline->tile_parity_buf);


  cr_compute_pipeline_resize(ctx, pipeline, ctx->swapchain.dimensions.width, ctx->swapchain.dimensions.height);

  
  pipeline->dynamic = cr_util_alloc(ctx, ctx->swapchain.n_imgs, sizeof(struct cr_compute_pipeline_dynamic_state_t));
  for (uint32_t i = 0; i < ctx->swapchain.n_imgs; i++) {
    pipeline->dynamic[i].segment_data = cr_util_alloc(ctx, 8000, sizeof(struct cr_segment_t));
    cr_mem_create_gpu_buffer(ctx, 
                             sizeof(struct cr_segment_t) * 8000, 
                             CR_GPU_BUFFER_SSBO, 
                             CR_GPU_BUFFER_MEM_DEVICE_LOCAL, &pipeline->dynamic[i].segment_buf);


    VkDescriptorBufferInfo segment_info = {
      .buffer = pipeline->dynamic[i].segment_buf.buf, 
      .offset = 0,
      .range  = VK_WHOLE_SIZE
    };

    VkDescriptorBufferInfo tile_info = {
      .buffer = pipeline->tile_buf.buf,
      .offset = 0,
      .range  = VK_WHOLE_SIZE
    };

    VkDescriptorBufferInfo segment_indices_info = {
      .buffer = pipeline->segment_indices_buf.buf,
      .offset = 0,
      .range  = VK_WHOLE_SIZE
    };
    
    VkDescriptorBufferInfo parity_in_info = {
      .buffer = pipeline->parity_in_buf.buf,
      .offset = 0,
      .range  = VK_WHOLE_SIZE
    };
    VkDescriptorBufferInfo tile_parity_info = {
      .buffer = pipeline->tile_parity_buf.buf,
      .offset = 0,
      .range  = VK_WHOLE_SIZE
    };
    VkDescriptorImageInfo image_info = {
      .imageView = pipeline->storage_img.view, 
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL
    };

    VkWriteDescriptorSet writes[6] = {
      {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = _global_sets[i],
        .dstBinding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .pBufferInfo = &segment_info
      },
      {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = _global_sets[i],
        .dstBinding = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .pBufferInfo = &tile_info
      },
      {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = _global_sets[i],
        .dstBinding = 2,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .pBufferInfo = &segment_indices_info
      },
      {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = _global_sets[i],
        .dstBinding = 3,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 1,
        .pImageInfo = &image_info
      },
      {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = _global_sets[i],
        .dstBinding = 4,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .pBufferInfo = &tile_parity_info
      },
      {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = _global_sets[i],
        .dstBinding = 5,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .pBufferInfo = &parity_in_info
      },
    };

    vkUpdateDescriptorSets(ctx->logical_dev, 4, writes, 0, NULL);
  }

  VkPipelineLayoutCreateInfo layout_info = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    .setLayoutCount = 1,
    .pSetLayouts  = &_set_layout,
    .pushConstantRangeCount = 1,
    .pPushConstantRanges = &pc_range
  };

  _VK_CHECK(ctx, vkCreatePipelineLayout(ctx->logical_dev, &layout_info, NULL, &_compute_pipeline_layout));

  return true;

err:
  return false;
}


bool 
cr_compute_pipeline_init(
  struct cr_context_t* ctx, struct cr_compute_pipeline_init_info_t* info, 
  struct cr_compute_pipeline_t* pipeline) {
  if(!info || !pipeline || !ctx || !info->shader_path_bin || !info->shader_path_fill) return false;

  memset(pipeline, 0, sizeof(struct cr_compute_pipeline_t));

  pipeline->info = *info;

  _global_sets = cr_util_alloc(ctx, ctx->swapchain.n_imgs, sizeof(VkDescriptorSet));
  if(!_vk_pipeline_layout_init(ctx, pipeline)) {
    CR_ERROR(ctx->log, "Failed to create Vulkan-Compute pipeline layout with shader paths: ('%s', '%s')", info->shader_path_fill,
             info->shader_path_bin);
    return false;
  }

  if(!_vk_create_compute_pipeline(ctx, info->shader_path_fill, &pipeline->pipeline_fill)) {
    CR_ERROR(ctx->log, "Failed to create Vulkan-Compute pipeline with shader path: '%s'", info->shader_path_fill);
    return false;
  }

  if(!_vk_create_compute_pipeline(ctx, info->shader_path_bin, &pipeline->pipeline_bin)) {
    CR_ERROR(ctx->log, "Failed to create Vulkan-Compute pipeline with shader path: '%s'", info->shader_path_bin);
    return false;
  }
  
  if(!_vk_create_compute_pipeline(ctx, info->shader_path_prefix, &pipeline->pipeline_prefix)) {
    CR_ERROR(ctx->log, "Failed to create Vulkan-Compute pipeline with shader path: '%s'", info->shader_path_prefix);
    return false;
  }

  if(!_vk_create_compute_pipeline(ctx, info->shader_path_parity, &pipeline->pipeline_parity)) {
    CR_ERROR(ctx->log, "Failed to create Vulkan-Compute pipeline with shader path: '%s'", info->shader_path_parity);
    return false;
  }

  return true;
}


bool 
cr_compute_pipeline_resize(struct cr_context_t* ctx, struct cr_compute_pipeline_t* pipeline, uint32_t w, uint32_t h) {
  if(pipeline->storage_img.image != VK_NULL_HANDLE)  {
    _vk_destroy_storage_image(ctx, &pipeline->storage_img);
  }
  _vk_create_storage_image(ctx, w, h, &pipeline->storage_img);

  VkDescriptorImageInfo img_info = {
    .imageView   = pipeline->storage_img.view,
    .imageLayout = VK_IMAGE_LAYOUT_GENERAL
  };

  for (uint32_t i = 0; i < ctx->swapchain.n_imgs; ++i) {
    VkWriteDescriptorSet write = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _global_sets[i],
      .dstBinding = 3,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .pImageInfo = &img_info
    };

    vkUpdateDescriptorSets(
      ctx->logical_dev,
      1, &write,
      0, NULL
    );
  }
  return true;
}

bool
cr_compute_pipeline_dispatch(
  struct cr_context_t* ctx,
  struct cr_compute_pipeline_t* pipeline,
  uint32_t frame_idx,
  uint32_t swapchain_image_idx
) {
  struct cr_frame_t* frame = &ctx->frameloop.frames[frame_idx];

  if(pipeline->dynamic[swapchain_image_idx].n_segments > 0) 
    cr_mem_transfer_to_device_local_gpu_buffer(ctx, frame, pipeline->dynamic[swapchain_image_idx].segment_data,
                                               pipeline->dynamic[swapchain_image_idx].n_segments * 
                                               sizeof(struct cr_segment_t),
                                               &pipeline->dynamic[swapchain_image_idx].segment_buf);


  uint32_t screen_w = ctx->swapchain.dimensions.width;
  uint32_t screen_h = ctx->swapchain.dimensions.height;
  uint32_t tile_size = 32; 

  uint32_t tiles_x = (screen_w + tile_size - 1) / tile_size;
  uint32_t tiles_y = (screen_h + tile_size - 1) / tile_size;

  uint32_t n_segments = pipeline->dynamic[swapchain_image_idx].n_segments;

  struct cr_compute_pipeline_push_constant_t pc = {
    .tile_size = tile_size,
    .n_tiles_x = tiles_x,
    .n_tiles_y = tiles_y,
    .n_segments = n_segments,
    .fill_rule = CR_COMPUTE_FILL_RULE_EVEN_ODD,
    .screen_w = screen_w,
    .screen_h = screen_h,
    .n_paths = pipeline->dynamic[swapchain_image_idx].n_paths
  };

  {
    // clear/reset tile buffer per frame 
    vkCmdFillBuffer(
      frame->cmd_buf,
      pipeline->tile_buf.buf,
      0,
      VK_WHOLE_SIZE,
      0
    );

    // prevent read/write hazard - barrier from TRANSFER/WRITE -> COMPUTE/READWRITE
    VkBufferMemoryBarrier2 clear_barrier = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
      .buffer = pipeline->tile_buf.buf,
      .size = VK_WHOLE_SIZE,
      .offset        = 0,
    };

    VkDependencyInfo clear_dep = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .bufferMemoryBarrierCount = 1,
      .pBufferMemoryBarriers = &clear_barrier
    };

    vkCmdPipelineBarrier2(frame->cmd_buf, &clear_dep);
  }
  {
    // clear/reset tile buffer per frame 
    vkCmdFillBuffer(
      frame->cmd_buf,
      pipeline->tile_parity_buf.buf,
      0,
      VK_WHOLE_SIZE,
      0
    );

    // prevent read/write hazard - barrier from TRANSFER/WRITE -> COMPUTE/READWRITE
    VkBufferMemoryBarrier2 clear_barrier = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
      .buffer = pipeline->tile_parity_buf.buf,
      .size = VK_WHOLE_SIZE,
      .offset        = 0,
    };

    VkDependencyInfo clear_dep = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .bufferMemoryBarrierCount = 1,
      .pBufferMemoryBarriers = &clear_barrier
    };

    vkCmdPipelineBarrier2(frame->cmd_buf, &clear_dep);
  }
  {
    // clear/reset tile buffer per frame 
    vkCmdFillBuffer(
      frame->cmd_buf,
      pipeline->parity_in_buf.buf,
      0,
      VK_WHOLE_SIZE,
      0
    );

    // prevent read/write hazard - barrier from TRANSFER/WRITE -> COMPUTE/READWRITE
    VkBufferMemoryBarrier2 clear_barrier = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
      .buffer = pipeline->parity_in_buf.buf,
      .size = VK_WHOLE_SIZE,
      .offset        = 0,
    };

    VkDependencyInfo clear_dep = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .bufferMemoryBarrierCount = 1,
      .pBufferMemoryBarriers = &clear_barrier
    };

    vkCmdPipelineBarrier2(frame->cmd_buf, &clear_dep);
  }

  {
    // clear/reset tile buffer per frame 
    vkCmdFillBuffer(
      frame->cmd_buf,
      pipeline->segment_indices_buf.buf,
      0,
      VK_WHOLE_SIZE,
      0
    );

    // prevent read/write hazard - barrier from TRANSFER/WRITE -> COMPUTE/READWRITE
    VkBufferMemoryBarrier2 clear_barrier = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
      .buffer = pipeline->segment_indices_buf.buf,
      .size = VK_WHOLE_SIZE,
      .offset        = 0,
    };

    VkDependencyInfo clear_dep = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .bufferMemoryBarrierCount = 1,
      .pBufferMemoryBarriers = &clear_barrier
    };

    vkCmdPipelineBarrier2(frame->cmd_buf, &clear_dep);
  }
  VkImageMemoryBarrier2 to_clear = {
    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
    .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
    .srcAccessMask = 0,
    .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
    .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
    .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    .image = pipeline->storage_img.image,
    .subresourceRange = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .levelCount = 1,
      .layerCount = 1
    }
  };

  VkDependencyInfo dep = {
    .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
    .imageMemoryBarrierCount = 1,
    .pImageMemoryBarriers = &to_clear
  };

  vkCmdPipelineBarrier2(frame->cmd_buf, &dep);

  VkClearColorValue clear = { .float32 = { 0.1f, 0.1f, 0.1f, 1.0f } };

  vkCmdClearColorImage(
    frame->cmd_buf,
    pipeline->storage_img.image,
    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    &clear,
    1,
    &(VkImageSubresourceRange){
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .levelCount = 1,
      .layerCount = 1
    }
  );




  // making sure the storage image is in GENERAL layout before any
  // shader writes happen as the image is initially in UNDEFINED, 
  // which would cause UB.
  VkImageMemoryBarrier2 to_compute = {
    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
    .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
    .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
    .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    .dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT |
    VK_ACCESS_2_SHADER_READ_BIT,
    .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
    .image = pipeline->storage_img.image,
    .subresourceRange = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .levelCount = 1,
      .layerCount = 1
    }
  };

  VkDependencyInfo dep_compute = {
    .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
    .imageMemoryBarrierCount = 1,
    .pImageMemoryBarriers = &to_compute
  };

  vkCmdPipelineBarrier2(frame->cmd_buf, &dep_compute);


  vkCmdBindPipeline(
    frame->cmd_buf,
    VK_PIPELINE_BIND_POINT_COMPUTE,
    pipeline->pipeline_bin
  );

  vkCmdBindDescriptorSets(
    frame->cmd_buf,
    VK_PIPELINE_BIND_POINT_COMPUTE,
    _compute_pipeline_layout,
    0,
    1,
    &_global_sets[swapchain_image_idx],
    0,
    NULL
  );

  vkCmdPushConstants(
    frame->cmd_buf,
    _compute_pipeline_layout,
    VK_SHADER_STAGE_COMPUTE_BIT,
    0,
    sizeof(pc),
    &pc
  );

  const uint32_t LOCAL_SIZE = 32;
  uint32_t bin_groups =
    (n_segments + LOCAL_SIZE - 1) / LOCAL_SIZE;

  vkCmdDispatch(frame->cmd_buf, bin_groups, 1, 1);

  VkBufferMemoryBarrier2 shader_barriers[3] = {
    {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      .srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
      .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
      .buffer        = pipeline->tile_buf.buf,
      .offset        = 0,
      .size          = VK_WHOLE_SIZE
    },
    {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      .srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
      .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
      .buffer        = pipeline->segment_indices_buf.buf,
      .offset        = 0,
      .size          = VK_WHOLE_SIZE
    },
    {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
      .buffer = pipeline->dynamic[swapchain_image_idx].segment_buf.buf,
      .offset = 0,
      .size = VK_WHOLE_SIZE
    }
  };

  VkDependencyInfo dep_shader = {
    .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
    .bufferMemoryBarrierCount = 3,
    .pBufferMemoryBarriers    = shader_barriers
  };

  vkCmdPipelineBarrier2(frame->cmd_buf, &dep_shader);

  vkCmdBindPipeline(frame->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                    pipeline->pipeline_prefix);

  vkCmdBindDescriptorSets(frame->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                         _compute_pipeline_layout , 0, 1,
                          &_global_sets[swapchain_image_idx], 0, NULL);

  vkCmdPushConstants(frame->cmd_buf, _compute_pipeline_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT,
                     0, sizeof(pc), &pc);

  vkCmdDispatch(frame->cmd_buf, 1, tiles_y * tile_size, 1);

  VkBufferMemoryBarrier2 prefix_barrier = {
  .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
  .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
  .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
  .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
  .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
  .buffer = pipeline->tile_parity_buf.buf,
  .offset = 0,
  .size = VK_WHOLE_SIZE
  };
  VkDependencyInfo dep_prefix = {
    .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
    .bufferMemoryBarrierCount = 1,
    .pBufferMemoryBarriers    = &prefix_barrier 
  };

  vkCmdPipelineBarrier2(frame->cmd_buf, &dep_prefix);

  vkCmdBindPipeline(frame->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                    pipeline->pipeline_parity);

  vkCmdBindDescriptorSets(frame->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                         _compute_pipeline_layout , 0, 1,
                          &_global_sets[swapchain_image_idx], 0, NULL);

  vkCmdPushConstants(frame->cmd_buf, _compute_pipeline_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT,
                     0, sizeof(pc), &pc);

  vkCmdDispatch(frame->cmd_buf, tiles_y, tile_size, 1);

  VkBufferMemoryBarrier2 parity_barrier = {
    .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
    .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
    .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
    .buffer = pipeline->parity_in_buf.buf,
    .offset = 0,
    .size = VK_WHOLE_SIZE
  };

  VkDependencyInfo parity_dep = {
    .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
    .bufferMemoryBarrierCount = 1,
    .pBufferMemoryBarriers    = &parity_barrier
  };


  vkCmdPipelineBarrier2(frame->cmd_buf, &parity_dep); 

  vkCmdBindPipeline(frame->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                    pipeline->pipeline_fill);

  vkCmdBindDescriptorSets(frame->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                         _compute_pipeline_layout , 0, 1,
                          &_global_sets[swapchain_image_idx], 0, NULL);

  vkCmdPushConstants(frame->cmd_buf, _compute_pipeline_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT,
                     0, sizeof(pc), &pc);

  vkCmdDispatch(frame->cmd_buf, tiles_x, tiles_y, 1);


  // Barrier to prevent hazard of COMPUTE/WRITE -> TRANSFER/READ in the storage image.
  // Because the image storage data needs to  be in VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL 
  // for vkCmdBlitImage(), we need to also prevent the layout hazard from 
  // VK_IMAGE_LAYOUT_GENERAL/WRITE ->  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL/READ.
  VkImageMemoryBarrier2 storage_to_src = {
    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
    .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
    .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
    .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
    .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
    .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
    .image = pipeline->storage_img.image,
    .subresourceRange = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .levelCount = 1,
      .layerCount = 1
    }
  };

  VkImageMemoryBarrier2 swapchain_to_dst = {
    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
    .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
    .srcAccessMask = 0,
    .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
    .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
    .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    .image = ctx->swapchain.imgs[swapchain_image_idx],
    .subresourceRange = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .levelCount = 1,
      .layerCount = 1
    }
  };

  VkImageMemoryBarrier2 barriers[] = {
    storage_to_src,
    swapchain_to_dst
  };

  VkDependencyInfo swapchain_deps = {
    .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
    .imageMemoryBarrierCount = 2,
    .pImageMemoryBarriers = barriers
  };

  vkCmdPipelineBarrier2(frame->cmd_buf, &swapchain_deps);

  VkImageBlit blit = {
    .srcSubresource = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .layerCount = 1
    },
    .dstSubresource = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .layerCount = 1
    },
    .srcOffsets = {
      { 0, 0, 0 },
      { (int32_t)screen_w, (int32_t)screen_h, 1 }
    },
    .dstOffsets = {
      { 0, 0, 0 },
      { (int32_t)screen_w, (int32_t)screen_h, 1 }
    }
  };

  vkCmdBlitImage(
    frame->cmd_buf,
    pipeline->storage_img.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
    ctx->swapchain.imgs[swapchain_image_idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    1,
    &blit,
    VK_FILTER_NEAREST
  );

  VkImageMemoryBarrier2 present_barrier = {
    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
    .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
    .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
    .dstStageMask = VK_PIPELINE_STAGE_2_NONE,
    .dstAccessMask = 0,
    .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    .image = ctx->swapchain.imgs[swapchain_image_idx],
    .subresourceRange = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .levelCount = 1,
      .layerCount = 1
    }
  };

  VkDependencyInfo present_dep = {
    .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
    .imageMemoryBarrierCount = 1,
    .pImageMemoryBarriers = &present_barrier
  };

  vkCmdPipelineBarrier2(frame->cmd_buf, &present_dep);

  pipeline->dynamic[swapchain_image_idx].n_segments = 0;
  return true;
}

bool 
cr_compute_pipeline_get_internal_shader_paths(const char* subpath, char** o_bin_path, char** o_fill_path, char** o_prefix_xor_path, char** o_tile_parity_path) {
  const char* state_dir = cr_util_get_state_folder();
  char shader_dir[PATH_MAX];
  snprintf(shader_dir, sizeof(shader_dir), "%s/%s/shaders/%s", state_dir, _CR_BRAND_NAME, subpath);

  char* bin_src = malloc(PATH_MAX);
  if(!bin_src) return false;
  snprintf(bin_src, PATH_MAX, "%s/bin_compute.spv", shader_dir);
  char* fill_src = malloc(PATH_MAX);
  if(!fill_src) return false;
  snprintf(fill_src, PATH_MAX, "%s/fill_compute.spv", shader_dir);
  char* prefix_xor_src = malloc(PATH_MAX);
  if(!prefix_xor_src) return false;
  snprintf(prefix_xor_src, PATH_MAX, "%s/prefix_xor_compute.spv", shader_dir);
  char* tile_parity_src = malloc(PATH_MAX);
  if(!tile_parity_src) return false;
  snprintf(tile_parity_src, PATH_MAX, "%s/tile_parity_compute.spv", shader_dir);

  *o_bin_path = bin_src;
  *o_fill_path = fill_src;
  *o_prefix_xor_path = prefix_xor_src;
  *o_tile_parity_path = tile_parity_src;

  return true;
}

bool 
cr_compute_pipeline_insert_segment(struct cr_context_t* ctx, struct cr_compute_pipeline_t* pipeline,
    struct cr_segment_t segment, uint32_t swapchain_idx) {
  pipeline->dynamic[swapchain_idx].segment_data[pipeline->dynamic[swapchain_idx].n_segments++] = segment;

  return true;
}
