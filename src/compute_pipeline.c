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

#define DIV_UP(x, y) (((x) + (y) - 1) / (y))

static VkPipelineLayout _compute_pipeline_layout;
static VkDescriptorSetLayout _set_layout;
static VkDescriptorSet* _global_sets;

static bool _create_shader_module(struct  cr_context_t* ctx, const char* filepath, VkShaderModule* o_module);
static bool _vk_create_compute_pipeline(struct cr_context_t* ctx, const char* compute_path, VkPipeline* o_pipeline);
static bool _vk_pipeline_layout_init(struct cr_context_t* ctx, struct cr_compute_pipeline_t* pipeline, struct cr_compute_pipeline_layout_binding_t* binding, uint32_t n_bindings);
static bool _extract_kernel_name(
  const char* path,
  char* out,
  size_t out_size);

static VkPipeline               _kernel_by_name(struct cr_context_t* ctx, struct cr_compute_pipeline_t* pipeline, const char* kernel_name);
static struct cr_gpu_buffer_t*  _buffer_by_name(struct cr_context_t* ctx, struct cr_compute_pipeline_t* pipeline, const char* buffer_name);

static void _vk_dispatch_compute(
  struct cr_frame_t* frame, VkPipeline pipeline, 
  uint32_t swap_idx, struct cr_compute_pipeline_push_constant_t* pc,
  uint32_t gc_x, uint32_t gc_y, uint32_t gc_z, const struct cr_gpu_buffer_t barrier_buffers[], uint32_t n_barrier_buffers);

static void _vk_compute_present(struct cr_context_t* ctx, struct cr_frame_t* frame, struct cr_compute_pipeline_t* pipeline, uint32_t swapchain_image_idx, 
                                uint32_t screen_w, uint32_t screen_h);

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

  VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT subgroupInfo = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
    .pNext = NULL,
    .requiredSubgroupSize = 32
  };
  VkPipelineShaderStageCreateInfo stage_info = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    .stage = VK_SHADER_STAGE_COMPUTE_BIT,
    .module = comp_mod,
    .pName = "main",
    .pNext  = &subgroupInfo
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


static bool _extract_kernel_name(
  const char* path,
  char* out,
  size_t out_size) {

  char temp[PATH_MAX];
  strncpy(temp, path, sizeof(temp));
  temp[sizeof(temp) - 1] = '\0';

  char* p = strchr(temp, '_');
  if (!p) goto fail;

  p = strchr(p + 1, '_');
  if (!p) goto fail;

  char* name_start = p + 1;

  char* compute = strstr(name_start, "_compute");
  if (!compute) goto fail;
  *compute = '\0';

  strncpy(out, name_start, out_size);
  out[out_size - 1] = '\0';
  return true;

fail:
  return false;
};


VkPipeline _kernel_by_name(struct cr_context_t* ctx, struct cr_compute_pipeline_t* pipeline, const char* kernel_name) {
  uint32_t name_hash = cr_util_djb2_hash((char*)kernel_name);
  for(uint32_t i = 0; i < pipeline->info.n_shaders; i++) {
    if(pipeline->kernels[i].hash == name_hash) return pipeline->kernels[i].kernel_pipeline;
  }
  CR_ERROR(ctx->log, "Kernel name '%s' did not match any pipelines.", kernel_name);
  return VK_NULL_HANDLE;
}

struct cr_gpu_buffer_t* 
_buffer_by_name(struct cr_context_t* ctx, struct cr_compute_pipeline_t* pipeline, const char* buffer_name) {
  uint32_t name_hash = cr_util_djb2_hash((char*)buffer_name);
  for(uint32_t i = 0; i < pipeline->n_buffers; i++) {
    if(pipeline->buffers[i].hash == name_hash) return &pipeline->buffers[i].buf;
  }
  CR_ERROR(ctx->log, "Buffer name '%s' did not match any bindings.", buffer_name);
  return NULL; 
}

void 
_vk_dispatch_compute(
  struct cr_frame_t* frame, VkPipeline pipeline, 
  uint32_t swap_idx, struct cr_compute_pipeline_push_constant_t* pc,
  uint32_t gc_x, uint32_t gc_y, uint32_t gc_z, const struct cr_gpu_buffer_t barrier_buffers[], uint32_t n_barrier_buffers) {

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

  if(n_barrier_buffers) {
    VkBufferMemoryBarrier2 shader_barriers[n_barrier_buffers];
    for(uint32_t i = 0; i < n_barrier_buffers; i++) {
      shader_barriers[i] = (VkBufferMemoryBarrier2){
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT, 
        .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT, 
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
}

void 
_vk_compute_present(struct cr_context_t* ctx, struct cr_frame_t* frame, struct cr_compute_pipeline_t* pipeline, uint32_t swapchain_image_idx, 
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
_vk_pipeline_layout_init(
  struct cr_context_t* ctx, 
  struct cr_compute_pipeline_t* pipeline,
  struct cr_compute_pipeline_layout_binding_t* bindings,
  uint32_t n_bindings
) {
  VkPushConstantRange pc_range = {
    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    .offset = 0,
    .size = sizeof(struct cr_compute_pipeline_push_constant_t)
  };

  VkDescriptorSetLayoutBinding set_bindings[n_bindings + 2]; // +1 for the storage image, +1 for the segment buffer
  memset(set_bindings, 0, sizeof(set_bindings));
  for(uint32_t i = 0; i < n_bindings + 1; i++) {
    _vk_descriptor_layout_binding(VK_SHADER_STAGE_COMPUTE_BIT, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, i, set_bindings);
  }

  _vk_descriptor_layout_binding(VK_SHADER_STAGE_COMPUTE_BIT, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  n_bindings + 1, set_bindings);

  VkDescriptorSetLayoutCreateInfo desc_layout_info = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    .bindingCount = sizeof(set_bindings) / sizeof(set_bindings[0]),
    .pBindings = set_bindings
  };

  vkCreateDescriptorSetLayout(ctx->logical_dev, &desc_layout_info, NULL, &_set_layout);

  VkDescriptorPoolSize pool_sizes[2] = {
    {
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = (n_bindings + 1) * ctx->swapchain.n_imgs 
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

  pipeline->buffers = cr_util_alloc(ctx, n_bindings, sizeof(*pipeline->buffers));
  pipeline->n_buffers = n_bindings;
  for(uint32_t i = 0; i < n_bindings; i++) {
    struct cr_compute_pipeline_layout_buffer_t* buf = &pipeline->buffers[i];
    CR_TRACE(ctx->log, "Created buffer '%s' with size %lu, binding at location: %i in layout.", bindings[i].name, bindings[i].buffer_size, i + 1);

    buf->hash = cr_util_djb2_hash((char*)bindings[i].name);
    cr_mem_create_gpu_buffer(ctx, bindings[i].buffer_size, CR_GPU_BUFFER_SSBO, CR_GPU_BUFFER_MEM_DEVICE_LOCAL,
                             &buf->buf);
  }

  cr_compute_pipeline_resize(ctx, pipeline, ctx->swapchain.dimensions.width, ctx->swapchain.dimensions.height);

  pipeline->dynamic = cr_util_alloc(ctx, ctx->swapchain.n_imgs, sizeof(struct cr_compute_pipeline_dynamic_state_t));
  for (uint32_t i = 0; i < ctx->swapchain.n_imgs; i++) {

    pipeline->dynamic[i].segment_data = cr_util_alloc(ctx, 8000, sizeof(struct cr_segment_t));
    cr_mem_create_gpu_buffer(
      ctx, 
      sizeof(struct cr_segment_t) * 8000, 
      CR_GPU_BUFFER_SSBO, 
      CR_GPU_BUFFER_MEM_DEVICE_LOCAL, &pipeline->dynamic[i].segment_buf);


    VkWriteDescriptorSet writes[n_bindings + 2]; 
    memset(writes, 0, sizeof(writes));


    VkDescriptorBufferInfo infos[n_bindings + 1]; 

    infos[0] = (VkDescriptorBufferInfo){ 
      .buffer = pipeline->dynamic[i].segment_buf.buf,
      .offset = 0,
      .range  = VK_WHOLE_SIZE,
    };
    writes[0] = (VkWriteDescriptorSet){
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _global_sets[i],
      .dstBinding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .pBufferInfo = &infos[0]};

    for(uint32_t j = 1; j <= n_bindings; j++) {
      infos[j] = (VkDescriptorBufferInfo){ 
        .buffer = pipeline->buffers[j - 1].buf.buf,
        .offset = 0,
        .range  = VK_WHOLE_SIZE,
      };
      writes[j] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = _global_sets[i],
        .dstBinding = j,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .pBufferInfo = &infos[j]};
    }

    VkDescriptorImageInfo image_info = {
      .imageView = pipeline->storage_img.view,
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL
    };


    writes[n_bindings + 1] = (VkWriteDescriptorSet){
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _global_sets[i],
      .dstBinding = n_bindings + 1, 
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .descriptorCount = 1,
      .pImageInfo = &image_info};


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
  pipeline->kernels = cr_util_alloc(ctx, info->n_shaders, sizeof(struct cr_compute_kernel_t));


  uint32_t screen_w = info->screen_w; 
  uint32_t screen_h = info->screen_h; 
  uint32_t tile_size = info->tile_size;

  uint32_t tiles_x = (screen_w + tile_size - 1) / tile_size;
  uint32_t tiles_y = (screen_h + tile_size - 1) / tile_size;

  uint32_t subgroup_size = 32;

  uint32_t n_rows = tiles_y * info->tile_size;

  uint32_t subgroups_per_row =
    (tiles_x + subgroup_size - 1) / subgroup_size;

  uint32_t total_prefix_parity_size = n_rows * subgroups_per_row;

  struct cr_compute_pipeline_layout_binding_t bindings[] = {
    (struct cr_compute_pipeline_layout_binding_t){.buffer_size = sizeof(uint32_t) * tiles_x * tiles_y,        .name = "tile_n_segments"       }, // 1
    (struct cr_compute_pipeline_layout_binding_t){.buffer_size = sizeof(uint32_t) * tiles_x * tiles_y,        .name = "tile_offsets"          }, // 2
    (struct cr_compute_pipeline_layout_binding_t){.buffer_size = sizeof(uint32_t) * tiles_x * tiles_y,        .name = "subgroup_tmp_binning"  }, // 3
    (struct cr_compute_pipeline_layout_binding_t){.buffer_size = sizeof(uint32_t) * tiles_x * tiles_y * 32,   .name = "tile_segments"         }, // 4
    (struct cr_compute_pipeline_layout_binding_t){.buffer_size = sizeof(uint32_t) * tiles_x * tiles_y,        .name = "tile_cursor"           }, // 5
    (struct cr_compute_pipeline_layout_binding_t){.buffer_size = sizeof(uint32_t) * tiles_x * tiles_y,        .name = "prefix_parity"         }, // 6
    (struct cr_compute_pipeline_layout_binding_t){.buffer_size = sizeof(uint32_t) * total_prefix_parity_size, .name = "subgroup_tmp"          }, // 7
  };

  if(!_vk_pipeline_layout_init(ctx, pipeline, bindings, sizeof(bindings) / sizeof(bindings[0]))) {
    CR_ERROR(ctx->log, "Failed to create Vulkan-Compute pipeline layout."); 
    return false;
  }

  for (uint32_t i = 0; i < info->n_shaders; i++) {
    const char* path = info->shader_paths[i];

    char kernel_name[256];
    if(!_extract_kernel_name(path, kernel_name, sizeof(kernel_name))) {
      CR_WARN(ctx->log, "Kernel shader at path '%s' uses invalid naming convention. Kernel will not be created correctly.",
              path);
      continue;
    }
    struct cr_compute_kernel_t* kernel =  &pipeline->kernels[i];

    kernel->hash = cr_util_djb2_hash(kernel_name);
    kernel->shader_path = info->shader_paths[i];

    if(!_vk_create_compute_pipeline(ctx, path, &kernel->kernel_pipeline)) {
      CR_ERROR(ctx->log,
               "Failed to create Vulkan compute pipeline with shader path: '%s'",
               path);
      continue;;
    }
  }


  CR_TRACE(ctx->log, "Successfully initialized compute state.");
  return true;
}


bool 
cr_compute_pipeline_resize(struct cr_context_t* ctx, struct cr_compute_pipeline_t* pipeline, uint32_t w, uint32_t h) {
  if(pipeline->storage_img.image != VK_NULL_HANDLE)  {
    cr_mem_destroy_storage_image(ctx, &pipeline->storage_img);
  }
  cr_mem_create_storage_image(ctx, w, h, &pipeline->storage_img);

  VkDescriptorImageInfo img_info = {
    .imageView   = pipeline->storage_img.view,
    .imageLayout = VK_IMAGE_LAYOUT_GENERAL
  };

  for (uint32_t i = 0; i < ctx->swapchain.n_imgs; ++i) {
    VkWriteDescriptorSet write = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _global_sets[i],
      .dstBinding = pipeline->n_buffers + 1,
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

struct _vk_access_masks {
  VkAccessFlags2 src_access; 
  VkAccessFlags2 dst_access;
};

bool
cr_compute_pipeline_dispatch(
  struct cr_context_t* ctx,
  struct cr_compute_pipeline_t* pipeline,
  uint32_t frame_idx,
  uint32_t swapchain_image_idx
) {
  struct cr_frame_t* frame = &ctx->frameloop.frames[frame_idx];

  if(pipeline->dynamic[swapchain_image_idx].n_segments > 0) {
    cr_mem_transfer_to_device_local_gpu_buffer(ctx, frame, pipeline->dynamic[swapchain_image_idx].segment_data,
                                               pipeline->dynamic[swapchain_image_idx].n_segments * 
                                               sizeof(struct cr_segment_t),
                                               &pipeline->dynamic[swapchain_image_idx].segment_buf);
  }


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
  for(uint32_t i = 0; i < pipeline->n_buffers; i++) {
    cr_mem_clear_gpu_buffer(ctx, frame, &pipeline->buffers[i].buf);
  }

  // Clear the storage image pixels 
  cr_mem_clear_storage_image_color(ctx, frame, &pipeline->storage_img, (float[4]){0.1f, 0.1f, 0.1f, 1.0f});

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
  const uint32_t WG = 256;

  uint32_t gx_bin_per = DIV_UP(tiles_x * tiles_y, WG); 
  uint32_t gx_bin_all = DIV_UP(gx_bin_per, WG); 
  uint32_t gx_bin_count =
    (n_segments + WG - 1) / WG;

  _vk_dispatch_compute(frame, _kernel_by_name(ctx, pipeline, "bin_count"), 
                       swapchain_image_idx, &pc, gx_bin_count, 1, 1,
                       (struct cr_gpu_buffer_t[]){*_buffer_by_name(ctx, pipeline, "tile_n_segments")},
                       1); 

  _vk_dispatch_compute(frame, _kernel_by_name(ctx, pipeline, "bin_prefix_per_sub"), 
                       swapchain_image_idx, &pc, gx_bin_per, 1, 1,
                       (struct cr_gpu_buffer_t[]){
                       *_buffer_by_name(ctx, pipeline, "subgroup_tmp_binning"),
                       *_buffer_by_name(ctx, pipeline, "tile_offsets")
                       },
                       2); 

  _vk_dispatch_compute(frame, _kernel_by_name(ctx, pipeline, "bin_prefix_all_subs"), 
                       swapchain_image_idx, &pc, gx_bin_all, 1, 1,
                       (struct cr_gpu_buffer_t[]){
                       *_buffer_by_name(ctx, pipeline, "subgroup_tmp_binning")}, 1); 

  _vk_dispatch_compute(frame, _kernel_by_name(ctx, pipeline, "bin_prefix_final") ,
                       swapchain_image_idx, &pc, gx_bin_per, 1, 1,
                       (struct cr_gpu_buffer_t[]){
                       *_buffer_by_name(ctx, pipeline, "subgroup_tmp_binning"),
                       *_buffer_by_name(ctx, pipeline, "tile_offsets")
                       },
                       2); 

  _vk_dispatch_compute(frame, _kernel_by_name(ctx, pipeline, "bin_scatter"),
                       swapchain_image_idx, &pc, gx_bin_count, 1, 1,
                       (struct cr_gpu_buffer_t[]){
                       *_buffer_by_name(ctx, pipeline, "tile_segments"),
                       *_buffer_by_name(ctx, pipeline, "tile_cursor")
                       }, 2);

  _vk_dispatch_compute(frame, _kernel_by_name(ctx, pipeline, "base_parity"), 
                       swapchain_image_idx, &pc, pc.n_tiles_x, pc.n_tiles_y, 1, 
                       (struct cr_gpu_buffer_t[]){ 
                       *_buffer_by_name(ctx, pipeline, "prefix_parity"),
                       },
                       1
                       );

  _vk_dispatch_compute(
    frame,
    _kernel_by_name(ctx, pipeline, "parity_prefix_per_sub"), 
    swapchain_image_idx,
    &pc,
    DIV_UP(pc.n_tiles_x, WG),
    pc.n_tiles_y, 
    1,
    (struct cr_gpu_buffer_t[]){
      *_buffer_by_name(ctx, pipeline, "prefix_parity"),
      *_buffer_by_name(ctx, pipeline, "subgroup_tmp"),
    },
    2
  );

  if(pc.n_tiles_x > 256) {
  _vk_dispatch_compute(
    frame,
    _kernel_by_name(ctx, pipeline, "parity_prefix_all_subs"), 
    swapchain_image_idx,
    &pc,
    DIV_UP(pc.n_tiles_x, WG),
    pc.n_tiles_y, 
    1,
    (struct cr_gpu_buffer_t[]){
      *_buffer_by_name(ctx, pipeline, "subgroup_tmp"),
    },
    1
  );
  _vk_dispatch_compute(
    frame,
    _kernel_by_name(ctx, pipeline, "parity_prefix_final"), 
    swapchain_image_idx,
    &pc,
    DIV_UP(pc.n_tiles_x, WG),
    pc.n_tiles_y,
    1,
    (struct cr_gpu_buffer_t[]){
      *_buffer_by_name(ctx, pipeline, "prefix_parity"),
      *_buffer_by_name(ctx, pipeline, "subgroup_tmp"),
    },
    2
  );
  }

  _vk_dispatch_compute(
    frame,
    _kernel_by_name(ctx, pipeline, "fill"), 
    swapchain_image_idx,
    &pc,
    pc.n_tiles_x,
    pc.n_tiles_y,
    1,
    NULL, 0 
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
