#include "compute_pipeline.h"
#include "mem.h"
#include "util.h"
#include <errno.h>
#include <linux/limits.h>
#include <stdint.h>
#include <string.h>
#include <dirent.h>
#include <sys/types.h>
#include <vulkan/vulkan_core.h>

#define _SUBSYS_NAME "COMPUTE"

static VkPipelineLayout _compute_pipeline_layout;
static VkDescriptorSetLayout _set_layout;
static VkDescriptorSet* _global_sets;

static bool _create_shader_module(struct  cr_context_t* ctx, const char* filepath, VkShaderModule* o_module);
static bool _vk_create_compute_pipeline(struct cr_context_t* ctx, const char* compute_path, VkPipeline* o_pipeline);
static bool _vk_pipeline_layout_init(struct cr_context_t* ctx, struct cr_compute_pipeline_t* pipeline);

#define _VK_DESCRIPTOR_LAYOUT_BINDING_BUF(buf, binding, swapchain_idx, name, writes)   \
VkDescriptorBufferInfo (name) = {                                     \
  .buffer = (buf),                                                    \
  .offset = 0,                                                        \
  .range  = VK_WHOLE_SIZE                                             \
};                                                                    \
(writes)[(binding)] = (VkWriteDescriptorSet){                         \
  .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,                    \
  .dstSet = _global_sets[(swapchain_idx)],                            \
  .dstBinding = (binding),                                            \
  .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,                \
  .descriptorCount = 1,                                               \
  .pBufferInfo = &(name)};                                            

#define _VK_DESCRIPTOR_LAYOUT_BINDING_IMG(img, binding, swapchain_idx, name, writes) \
VkDescriptorImageInfo (name) = {                                      \
  .imageView= (img),                                                  \
  .imageLayout = VK_IMAGE_LAYOUT_GENERAL                                \
};                                                                    \
(writes)[(binding)] = (VkWriteDescriptorSet){                         \
  .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,                    \
  .dstSet = _global_sets[i],                                          \
  .dstBinding = (binding),                                            \
  .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,                 \
  .descriptorCount = 1,                                               \
  .pImageInfo = &(name)};                                            

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

void 
_vk_descriptor_layout_binding(VkShaderStageFlagBits stage, VkDescriptorType type, uint32_t binding, VkDescriptorSetLayoutBinding* bindings) {
  bindings[binding] = (VkDescriptorSetLayoutBinding){
    .binding = binding,
    .descriptorType = type,
    .descriptorCount = 1,
    .stageFlags =  stage
  };
}

bool 
_vk_pipeline_layout_init(struct cr_context_t* ctx, struct cr_compute_pipeline_t* pipeline) {
  VkPushConstantRange pc_range = {
    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    .offset = 0,
    .size = sizeof(struct cr_compute_pipeline_push_constant_t)
  };

  VkDescriptorSetLayoutBinding bindings[6] = {};
  _vk_descriptor_layout_binding(VK_SHADER_STAGE_COMPUTE_BIT, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, bindings);
  _vk_descriptor_layout_binding(VK_SHADER_STAGE_COMPUTE_BIT, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, bindings);
  _vk_descriptor_layout_binding(VK_SHADER_STAGE_COMPUTE_BIT, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2, bindings);
  _vk_descriptor_layout_binding(VK_SHADER_STAGE_COMPUTE_BIT, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3, bindings);
  _vk_descriptor_layout_binding(VK_SHADER_STAGE_COMPUTE_BIT, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4, bindings);
  _vk_descriptor_layout_binding(VK_SHADER_STAGE_COMPUTE_BIT, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  5, bindings);

  VkDescriptorSetLayoutCreateInfo desc_layout_info = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    .bindingCount = sizeof(bindings) / sizeof(bindings[0]),
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
    CR_GPU_BUFFER_MEM_DEVICE_LOCAL, &pipeline->prefix_parity_buf);

  uint32_t subgroup_size = 32;

  uint32_t n_rows = tiles_y * i.tile_size;

  uint32_t subgroups_per_row =
    (tiles_x + subgroup_size - 1) / subgroup_size;

  uint32_t total = n_rows * subgroups_per_row;

  cr_mem_create_gpu_buffer(
    ctx,
    total * sizeof(uint32_t),
    CR_GPU_BUFFER_SSBO,
    CR_GPU_BUFFER_MEM_DEVICE_LOCAL,
    &pipeline->prefix_sg_tmp);

  cr_compute_pipeline_resize(ctx, pipeline, ctx->swapchain.dimensions.width, ctx->swapchain.dimensions.height);

  pipeline->dynamic = cr_util_alloc(ctx, ctx->swapchain.n_imgs, sizeof(struct cr_compute_pipeline_dynamic_state_t));
  for (uint32_t i = 0; i < ctx->swapchain.n_imgs; i++) {
    pipeline->dynamic[i].segment_data = cr_util_alloc(ctx, 8000, sizeof(struct cr_segment_t));
    cr_mem_create_gpu_buffer(ctx, 
                             sizeof(struct cr_segment_t) * 8000, 
                             CR_GPU_BUFFER_SSBO, 
                             CR_GPU_BUFFER_MEM_DEVICE_LOCAL, &pipeline->dynamic[i].segment_buf);


    VkWriteDescriptorSet writes[6] = {0};


    _VK_DESCRIPTOR_LAYOUT_BINDING_BUF(pipeline->dynamic[i].segment_buf.buf,       0, i, seg_info, writes);
    _VK_DESCRIPTOR_LAYOUT_BINDING_BUF(pipeline->tile_buf.buf,                     1, i, tile_info, writes);
    _VK_DESCRIPTOR_LAYOUT_BINDING_BUF(pipeline->segment_indices_buf.buf,          2, i, seg_ind_info, writes);
    _VK_DESCRIPTOR_LAYOUT_BINDING_BUF(pipeline->prefix_parity_buf.buf,            3, i, prefix_info, writes);
    _VK_DESCRIPTOR_LAYOUT_BINDING_BUF(pipeline->prefix_sg_tmp.buf,                4, i, prefix_sg_tmp_info, writes);
    _VK_DESCRIPTOR_LAYOUT_BINDING_IMG(pipeline->storage_img.view,                 5, i, img_info, writes);

    vkUpdateDescriptorSets(ctx->logical_dev, sizeof(writes) / sizeof(writes[0]), writes, 0, NULL);
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
  if(!info || !pipeline || !ctx || !info->shader_paths || !info->n_shaders) return false;

  memset(pipeline, 0, sizeof(struct cr_compute_pipeline_t));

  pipeline->info = *info;

  _global_sets = cr_util_alloc(ctx, ctx->swapchain.n_imgs, sizeof(VkDescriptorSet));
  if(!_vk_pipeline_layout_init(ctx, pipeline)) {
    CR_ERROR(ctx->log, "Failed to create Vulkan-Compute pipeline layout."); 
    return false;
  }
  for (uint32_t i = 0; i < info->n_shaders; i++) {
    VkPipeline* vk_pipe = NULL;

    const char* path = info->shader_paths[i];

    if (strstr(path, "bin") != NULL)
      vk_pipe = &pipeline->pipeline_bin;
    else if (strstr(path, "base_parity") != NULL)
      vk_pipe = &pipeline->pipeline_base_parity;
    else if (strstr(path, "prefix_per_sub") != NULL)
      vk_pipe = &pipeline->pipeline_prefix_per_sg;
    else if (strstr(path, "prefix_all_subs") != NULL)
      vk_pipe = &pipeline->pipeline_prefix_all_sgs;
    else if (strstr(path, "prefix_final") != NULL)
      vk_pipe = &pipeline->pipeline_prefix_final;
    else if (strstr(path, "fill") != NULL)
      vk_pipe = &pipeline->pipeline_fill;
    else {
      CR_ERROR(ctx->log,
               "Invalid compute-shader stage in path: %s",
               path);
      continue;
    }

    if (!_vk_create_compute_pipeline(ctx, path, vk_pipe)) {
      CR_ERROR(ctx->log,
               "Failed to create Vulkan compute pipeline with shader path: '%s'",
               path);
      return false;
    }
  }


  CR_TRACE(ctx->log, "Successfully initialized compute state.");
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
    .dstBinding = 5,
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

static void _vk_clear_compute_buffer(struct cr_frame_t* frame, struct cr_gpu_buffer_t* buf) {
// clear/reset tile buffer per frame 
vkCmdFillBuffer(
  frame->cmd_buf,
  buf->buf,
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
  .buffer = buf->buf, 
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

static void _vk_clear_storage_image(const struct cr_frame_t* frame, const struct cr_storage_image_t* img, const float clear_col[4]) {
VkImageMemoryBarrier2 to_clear = {
  .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
  .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
  .srcAccessMask = 0,
  .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
  .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
  .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
  .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
  .image = img->image,
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

VkClearColorValue clear;
memcpy(clear.float32, clear_col, sizeof(clear.float32));

vkCmdClearColorImage(
  frame->cmd_buf,
  img->image,
  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
  &clear,
  1,
  &(VkImageSubresourceRange){
    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
    .levelCount = 1,
    .layerCount = 1
  }
);
}

struct _vk_access_masks {
VkAccessFlags2 src_access; 
VkAccessFlags2 dst_access;
};

static void _vk_dispatch_compute(
struct cr_frame_t* frame, VkPipeline pipeline, 
uint32_t swap_idx, struct cr_compute_pipeline_push_constant_t* pc,
uint32_t gc_x, uint32_t gc_y, uint32_t gc_z, const struct cr_gpu_buffer_t barrier_buffers[], const struct _vk_access_masks access_mask[], uint32_t n_barrier_buffers) {
// making sure the storage image is in GENERAL layout before any
// shader writes happen as the image is initially in UNDEFINED, 
// which would cause UB.

if(n_barrier_buffers) {
  VkBufferMemoryBarrier2 shader_barriers[n_barrier_buffers];

  for(uint32_t i = 0; i < n_barrier_buffers; i++) {
    shader_barriers[i] = (VkBufferMemoryBarrier2){
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      .srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .srcAccessMask = access_mask[i].src_access,
      .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .dstAccessMask = access_mask[i].dst_access,
      .buffer        = barrier_buffers[i].buf,
      .offset        = 0,
      .size          = VK_WHOLE_SIZE
    };
  }

  VkDependencyInfo dep_shader = {
    .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
    .bufferMemoryBarrierCount = n_barrier_buffers,
    .pBufferMemoryBarriers    = shader_barriers
  };

  vkCmdPipelineBarrier2(frame->cmd_buf, &dep_shader);
}

vkCmdBindPipeline(
  frame->cmd_buf,
  VK_PIPELINE_BIND_POINT_COMPUTE,
  pipeline
);

vkCmdBindDescriptorSets(
  frame->cmd_buf,
  VK_PIPELINE_BIND_POINT_COMPUTE,
  _compute_pipeline_layout,
  0,
  1,
  &_global_sets[swap_idx],
  0,
  NULL
);

vkCmdPushConstants(
  frame->cmd_buf,
  _compute_pipeline_layout,
  VK_SHADER_STAGE_COMPUTE_BIT,
  0,
  sizeof(*pc),
  pc
);
vkCmdDispatch(frame->cmd_buf, gc_x, gc_y, gc_z);

}

static void _vk_compute_present(struct cr_context_t* ctx, struct cr_frame_t* frame, struct cr_compute_pipeline_t* pipeline, uint32_t swapchain_image_idx, 
                              uint32_t screen_w, uint32_t screen_h) {
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
uint32_t tile_size = pipeline->info.tile_size;

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

// Clear temporary buffers
  _vk_clear_compute_buffer(frame, &pipeline->tile_buf);
  _vk_clear_compute_buffer(frame, &pipeline->prefix_sg_tmp);
  _vk_clear_compute_buffer(frame, &pipeline->prefix_parity_buf);
  _vk_clear_compute_buffer(frame, &pipeline->segment_indices_buf);

  // Clear the storage image pixels 
  _vk_clear_storage_image(frame, &pipeline->storage_img, (float[4]){0.1f, 0.1f, 0.1f, 1.0f});

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

  // Dispatch binning pipeline 
  uint32_t gx = (n_segments + tile_size - 1) / tile_size; 


  _vk_dispatch_compute(frame, pipeline->pipeline_bin, 
                       swapchain_image_idx, &pc, gx, 1, 1,
                       (struct cr_gpu_buffer_t[]){pipeline->segment_indices_buf, pipeline->tile_buf},
                       (struct _vk_access_masks[]){
                       {.src_access = VK_ACCESS_2_SHADER_WRITE_BIT, .dst_access = VK_ACCESS_2_SHADER_READ_BIT},
                       {.src_access = VK_ACCESS_2_SHADER_WRITE_BIT, .dst_access = VK_ACCESS_2_SHADER_READ_BIT}},
                       2); 

  _vk_dispatch_compute(frame, pipeline->pipeline_base_parity, 
                       swapchain_image_idx, &pc, pc.n_tiles_x, pc.n_tiles_y, 1, 
                       (struct cr_gpu_buffer_t[]){pipeline->segment_indices_buf, pipeline->tile_buf, pipeline->prefix_parity_buf},
                       (struct _vk_access_masks[]) {
                       {.src_access = VK_ACCESS_2_SHADER_WRITE_BIT, .dst_access = VK_ACCESS_2_SHADER_READ_BIT},
                       {.src_access = VK_ACCESS_2_SHADER_WRITE_BIT, .dst_access = VK_ACCESS_2_SHADER_READ_BIT},
                       {.src_access = VK_ACCESS_2_SHADER_WRITE_BIT, .dst_access = VK_ACCESS_2_SHADER_READ_BIT},
                       },
                       3
                       );

  _vk_dispatch_compute(
    frame,
    pipeline->pipeline_prefix_per_sg,
    swapchain_image_idx,
    &pc,
    pc.n_tiles_y, 
    1,
    1,
    (struct cr_gpu_buffer_t[]){
      pipeline->prefix_sg_tmp,
      pipeline->prefix_parity_buf,
    },
    (struct _vk_access_masks[]){
      { .src_access = VK_ACCESS_2_SHADER_WRITE_BIT,
        .dst_access = VK_ACCESS_2_SHADER_READ_BIT },
      { .src_access = VK_ACCESS_2_SHADER_WRITE_BIT,
        .dst_access = VK_ACCESS_2_SHADER_READ_BIT },
    },
    2
  );

  uint32_t subgroups_per_row = (pc.n_tiles_x + 31) / 32;

  if(subgroups_per_row > 1) {
  _vk_dispatch_compute(
    frame,
    pipeline->pipeline_prefix_all_sgs,
    swapchain_image_idx,
    &pc,
    pc.n_tiles_y, 
    1,
    1,
    (struct cr_gpu_buffer_t[]){
      pipeline->prefix_sg_tmp,
    },
    (struct _vk_access_masks[]){
      { .src_access = VK_ACCESS_2_SHADER_WRITE_BIT,
        .dst_access = VK_ACCESS_2_SHADER_READ_BIT },
    },
    1
  );
  _vk_dispatch_compute(
    frame,
    pipeline->pipeline_prefix_final,
    swapchain_image_idx,
    &pc,
    pc.n_tiles_y, 
    1,
    1,
    (struct cr_gpu_buffer_t[]){
      pipeline->prefix_parity_buf,
      pipeline->prefix_sg_tmp,
    },
    (struct _vk_access_masks[]){
      { .src_access = VK_ACCESS_2_SHADER_WRITE_BIT,
        .dst_access = VK_ACCESS_2_SHADER_READ_BIT },
      { .src_access = VK_ACCESS_2_SHADER_WRITE_BIT,
        .dst_access = VK_ACCESS_2_SHADER_READ_BIT }
    },
    2
  );
  }

  _vk_dispatch_compute(
    frame,
    pipeline->pipeline_fill,
    swapchain_image_idx,
    &pc,
    pc.n_tiles_x,
    pc.n_tiles_y,
    1,
    (struct cr_gpu_buffer_t[]){
      pipeline->prefix_parity_buf, 
    },
    (struct _vk_access_masks[]) {
      {.src_access = VK_ACCESS_2_SHADER_WRITE_BIT, .dst_access = VK_ACCESS_2_SHADER_READ_BIT},
    },
    1
  );



  _vk_compute_present(ctx, frame, pipeline, swapchain_image_idx, screen_w, screen_h);

  pipeline->dynamic[swapchain_image_idx].n_segments = 0;
  return true;
}

bool cr_compute_pipeline_get_internal_shader_paths(struct cr_context_t* ctx, const char* subpath, char*** o_paths, uint32_t* o_n_paths) { 
  const char* state_dir = cr_util_get_state_folder();
  char shader_dir[PATH_MAX];

  snprintf(shader_dir, sizeof(shader_dir),
           "%s/%s/shaders/%s", state_dir, "corender", subpath);

  struct dirent **namelist;
  int n = scandir(shader_dir, &namelist, NULL, alphasort);

  if (n < 0) {
    CR_ERROR(ctx->log, "Failed to scan directory %s: %s", shader_dir, strerror(errno)); 
    *o_paths = NULL;
    *o_n_paths = 0;
    return false;
  }

  *o_paths = malloc(n * sizeof(char*));

  int i = 0;
  uint32_t final_n = n;
  for (int j = 0; j < n; j++) {
    if(strcmp(namelist[j]->d_name, ".") == 0) {final_n--; continue;}
    if(strcmp(namelist[j]->d_name, "..") == 0) {final_n--; continue;}
    char fullpath[PATH_MAX];
    snprintf(fullpath, sizeof(fullpath),
             "%s/%s", shader_dir, namelist[j]->d_name);
    (*o_paths)[i++] = strdup(fullpath);

    free(namelist[j]);
  }

  *o_n_paths = final_n;

  free(namelist);
  return true;

}

bool 
cr_compute_pipeline_insert_segment(struct cr_context_t* ctx, struct cr_compute_pipeline_t* pipeline,
    struct cr_segment_t segment, uint32_t swapchain_idx) {
  pipeline->dynamic[swapchain_idx].segment_data[pipeline->dynamic[swapchain_idx].n_segments++] = segment;

  return true;
}
