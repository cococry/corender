#include "mem.h"
#include "util.h"
#include <string.h>
#include <vulkan/vulkan_core.h>
#include "../vendor/vma/vk_mem_alloc.h"
#include "../include/corender/corender.h"

#define _SUBSYS_NAME "MEMORY"
#define _MAX_STAGING_RING_MEM 1024 * 1024 * 256

static VmaAllocator _vma_allocator;

struct cr_staging_ring_t _staging_ring;
struct cr_upload_context_t _upload;

static struct cr_gpu_buffer_t _deferred_buffer_destroys[CR_FRAME_COUNT];
static uint32_t _n_deferred_buffer_destroys = 0;

static bool 
_create_upload_context(struct cr_context_t* ctx, struct cr_upload_context_t* o_upload, uint32_t graphics_queue_family);

bool 
_create_upload_context(struct cr_context_t* ctx, struct cr_upload_context_t* o_upload, uint32_t graphics_queue_family) {
  if(!ctx || !o_upload) _PARAM_CHECK_FAIL();

  VkCommandPoolCreateInfo pool_info = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    .queueFamilyIndex = graphics_queue_family, 
    .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
  };

  _VK_CHECK(ctx, vkCreateCommandPool(
    ctx->logical_dev, &pool_info, NULL, &o_upload->cmd_pool));

  VkCommandBufferAllocateInfo buf_info = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    .commandPool = o_upload->cmd_pool,
    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    .commandBufferCount = 1
  };

  _VK_CHECK(ctx, vkAllocateCommandBuffers(ctx->logical_dev, &buf_info, &o_upload->cmd_buf));

  VkFenceCreateInfo fence_info = {
    .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
    .flags = VK_FENCE_CREATE_SIGNALED_BIT
  };

  _VK_CHECK(ctx, vkCreateFence(
    ctx->logical_dev, &fence_info, NULL, &o_upload->fence));

  return true;
err:
  return false;
}

bool 
cr_mem_init(struct cr_context_t* ctx) {
  if(!ctx) _PARAM_CHECK_FAIL();

  if(!_create_upload_context(ctx, &_upload, ctx->graphics_queue_family)) {
    CR_ERROR(ctx->log, "Failed to GPU buffer create upload context.");
    goto err;
  }
  _staging_ring.capacity = _MAX_STAGING_RING_MEM;


  VmaAllocatorCreateInfo allocator_info = {
    .device = ctx->logical_dev,
    .physicalDevice = ctx->phys_dev,
    .instance = ctx->instance 
  };

  _VK_CHECK(ctx, vmaCreateAllocator(&allocator_info, &_vma_allocator));


  if(!cr_mem_create_gpu_buffer(
    ctx, _MAX_STAGING_RING_MEM, 
    CR_GPU_BUFFER_NO_TYPE,
    CR_GPU_BUFFER_MEM_STAGING,
    &_staging_ring.buf)) {
    CR_ERROR(ctx->log, "Failed to create staging ring GPU Buffer.");
    return false;
  }

  CR_TRACE(ctx->log, "Initialized GPU memory context."); 

  return true;
err:
  return false;
}

bool 
cr_mem_create_gpu_buffer(
  struct 
  cr_context_t* ctx, 
  VkDeviceSize size,
  enum cr_gpu_buffer_type_t type,
  enum cr_gpu_buffer_memory_type_t mem_type,
  struct cr_gpu_buffer_t* o_buf
) {
  if(!o_buf || !ctx) _PARAM_CHECK_FAIL(); 

  *o_buf = (struct cr_gpu_buffer_t){0};

  VkBufferUsageFlags usage = 0;
  VkMemoryPropertyFlags mem_props;
  switch(mem_type) {
    case CR_GPU_BUFFER_MEM_DEVICE_LOCAL:
      usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      mem_props = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

      break;
    case CR_GPU_BUFFER_MEM_MAPPED:
      mem_props =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
      break;
    case CR_GPU_BUFFER_MEM_STAGING:
      usage |= 
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

      mem_props = 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
      break;
    default: {
      CR_FATAL(ctx->log, "Invalid buffer memory type for GPU buffer creation specified: %i. Use enumeration values of cr_gpu_buffer_type_t.", 
               type);
      break;
    }
  }

  switch(type) {
    case CR_GPU_BUFFER_INDEX: usage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT; break;
    case CR_GPU_BUFFER_VERTEX: usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT; break;
    case CR_GPU_BUFFER_INDIRECT: usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT; break;
    case CR_GPU_BUFFER_SSBO: usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; break;
    case CR_GPU_BUFFER_NO_TYPE: break;
    default: {
      CR_FATAL(ctx->log, "Invalid buffer type for GPU buffer creation specified: %i. Use enumeration values of cr_gpu_buffer_memory_type_t.", 
               type);
      break;
    }
  }

  VkBufferCreateInfo buffer_info = {
    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    .size = size,
    .usage = usage,
  };

  bool want_mapping = mem_type == CR_GPU_BUFFER_MEM_STAGING || mem_type == CR_GPU_BUFFER_MEM_MAPPED;

  VmaAllocationCreateInfo alloc_info = {};
  alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
  alloc_info.requiredFlags = mem_props;
  if(want_mapping) 
    alloc_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

  VkBuffer buffer;
  VmaAllocation allocation; 
  _VK_CHECK(ctx, vmaCreateBuffer(_vma_allocator, &buffer_info, &alloc_info, &buffer, &allocation, NULL));

  if(want_mapping)  {
    VmaAllocationInfo retrieved_info;
    vmaGetAllocationInfo(_vma_allocator, allocation, &retrieved_info); 
    o_buf->mem_handle = retrieved_info.pMappedData;
  }

  o_buf->buf = buffer;
  o_buf->_vma_allocation = allocation;
  o_buf->buf_size = size;
  o_buf->_mem_props = mem_props;
  o_buf->_usage = usage;

  o_buf->type = type;
  o_buf->mem_type = mem_type;
  CR_TRACE(ctx->log, "Successfully allocated GPU buffer %p with size %li", buffer, size);

  return true;
err:
  return false;

}

bool 
cr_mem_destroy_gpu_buffer(
  struct cr_context_t* ctx, 
  struct cr_gpu_buffer_t* buf
) {
  if(!ctx || !buf || !buf->_vma_allocation) _PARAM_CHECK_FAIL(); 

  vmaDestroyBuffer(_vma_allocator, buf->buf, (VmaAllocation)buf->_vma_allocation);

  CR_TRACE(ctx->log, "Successfully destroyed GPU buffer %p of size %lu.\n", buf->buf, buf->buf_size);

  memset(buf, 0, sizeof(*buf));

  return true;

err:
  return false;
} 

bool     
cr_mem_resize_gpu_buffer(
  struct cr_context_t* ctx, 
  struct cr_gpu_buffer_t* buf,
  VkDeviceSize new_size 
) {
  if(!ctx || !buf) _PARAM_CHECK_FAIL();

  struct cr_gpu_buffer_t new_buf;
  if(!cr_mem_create_gpu_buffer(ctx, new_size, buf->type,  buf->mem_type, &new_buf)) {
    return false;
  }

    if (buf->mem_handle && new_buf.mem_handle) {
    memcpy(
      new_buf.mem_handle,
      buf->mem_handle,
      CR_MIN(new_size, buf->buf_size)
    );
  }

  CR_TRACE(ctx->log, "Successfully resized GPU buffer %p from size %lu to size %lu (new buffer => %p)\n", buf->buf, buf->buf_size, new_size, new_buf.buf);

  struct cr_frame_t* frame = &ctx->frameloop.frames[ctx->frameloop.frame_idx];


  *buf = new_buf;

  return true;
}

bool 
cr_mem_upload_to_device_local_gpu_buffer(
  struct cr_context_t* ctx, 
  void* data, VkDeviceSize size, 
  enum cr_gpu_buffer_type_t type, 
  struct cr_gpu_buffer_t* o_buf) {
  if(!ctx || !o_buf) _PARAM_CHECK_FAIL();

  cr_mem_create_gpu_buffer(ctx, size, type, CR_GPU_BUFFER_MEM_DEVICE_LOCAL, o_buf);

  vkWaitForFences(ctx->logical_dev, 1, &_upload.fence, VK_TRUE, UINT64_MAX);

  _VK_CHECK(ctx, vkResetFences(ctx->logical_dev, 1, &_upload.fence));

  _VK_CHECK(ctx, vkResetCommandPool(ctx->logical_dev, _upload.cmd_pool, 0));

  VkCommandBufferBeginInfo begin_info = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
  };

  _VK_CHECK(ctx, vkBeginCommandBuffer(_upload.cmd_buf, &begin_info));

  size_t offset = cr_mem_staging_ring_alloc(
    &_staging_ring,
    size,
    ctx->phys_dev_limits.optimalBufferCopyOffsetAlignment
  );
  if (offset == SIZE_MAX) {
    CR_FATAL(ctx->log, "Staging ring out of memory this frame");
  }

  memcpy((uint8_t*)_staging_ring.buf.mem_handle + offset, data, size);

  VkBufferCopy copy = {
    .srcOffset = offset,
    .dstOffset = 0,
    .size      = size
  };

  vkCmdCopyBuffer(
    _upload.cmd_buf,
    _staging_ring.buf.buf,
    o_buf->buf,
    1,
    &copy
  );

  VkAccessFlags dst_access_mask = 0;
  switch(o_buf->type) {
    case CR_GPU_BUFFER_VERTEX:
      dst_access_mask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
      break;
    case CR_GPU_BUFFER_INDEX:
      dst_access_mask = VK_ACCESS_INDEX_READ_BIT;
      break;
    case CR_GPU_BUFFER_INDIRECT:
      dst_access_mask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT; 
      break;
    default:
      CR_FATAL(ctx->log, "Invalid GPU buffer type.");
      break;
  }
  VkBufferMemoryBarrier barrier = {
    .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
    .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
    .dstAccessMask = dst_access_mask, 
    .buffer = o_buf->buf,
    .offset = 0,
    .size   = size
  };

  vkCmdPipelineBarrier(
    _upload.cmd_buf,
    VK_PIPELINE_STAGE_TRANSFER_BIT,
    o_buf->type != CR_GPU_BUFFER_INDIRECT ? 
    VK_PIPELINE_STAGE_VERTEX_INPUT_BIT : VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
    0,
    0, NULL,
    1, &barrier,
    0, NULL
  );

  vkEndCommandBuffer(_upload.cmd_buf);

  VkSubmitInfo submit = {
    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
    .commandBufferCount = 1,
    .pCommandBuffers = &_upload.cmd_buf
  };

  vkQueueSubmit(ctx->graphics_queue, 1, &submit,
                _upload.fence);

  return true;

err:
  return false;
}


bool 
cr_mem_transfer_to_device_local_gpu_buffer(
    struct cr_context_t* ctx,
    struct cr_frame_t* frame,
    void* data,
    size_t size, 
    const struct cr_gpu_buffer_t* buf
) {
  if (!ctx || !data || !frame || !buf) _PARAM_CHECK_FAIL();

  size_t offset = cr_mem_staging_ring_alloc(
    &_staging_ring,
    size,
    ctx->phys_dev_limits.optimalBufferCopyOffsetAlignment
  );

  if (offset == SIZE_MAX) {
    CR_FATAL(ctx->log, "Staging ring out of memory this frame");
  }

  memcpy(
    (uint8_t*)_staging_ring.buf.mem_handle + offset,
    data,
    size
  );

  VkBufferCopy copy = {
    .srcOffset = offset,
    .dstOffset = 0,
    .size      = size
  };

  vkCmdCopyBuffer(
    frame->cmd_buf,
    _staging_ring.buf.buf,
    buf->buf,
    1,
    &copy
  );

  /* ---- Barrier (TRANSFER → device-local usage) ---- */

  VkPipelineStageFlags2 dst_stage_mask = 0;
  VkAccessFlags2        dst_access_mask = 0;

  switch (buf->type) {
    case CR_GPU_BUFFER_VERTEX:
      dst_stage_mask  = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
      dst_access_mask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
      break;

    case CR_GPU_BUFFER_INDEX:
      dst_stage_mask  = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
      dst_access_mask = VK_ACCESS_2_INDEX_READ_BIT;
      break;

    case CR_GPU_BUFFER_INDIRECT:
      dst_stage_mask  = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
      dst_access_mask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
      break;

    case CR_GPU_BUFFER_SSBO:
      dst_stage_mask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                        VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
      dst_access_mask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
      break;

    default:
      CR_FATAL(ctx->log, "Invalid GPU buffer type.");
      break;
  }

  VkBufferMemoryBarrier2 barrier = {
    .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
    .srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
    .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
    .dstStageMask  = dst_stage_mask,
    .dstAccessMask = dst_access_mask,
    .buffer        = buf->buf,
    .offset        = 0,
    .size          = VK_WHOLE_SIZE
  };

  VkDependencyInfo dep = {
    .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
    .bufferMemoryBarrierCount = 1,
    .pBufferMemoryBarriers    = &barrier
  };

  vkCmdPipelineBarrier2(frame->cmd_buf, &dep);

  return true;
}

bool 
cr_mem_upadate_lazy_destroys(struct cr_context_t* ctx) {
  for(uint32_t i = 0; i < _n_deferred_buffer_destroys; i++) {
    if(!cr_mem_destroy_gpu_buffer(ctx, &_deferred_buffer_destroys[i])) {
      CR_ERROR(ctx->log, "Failed to lazily destroy GPU buffer %p.\n", 
               _deferred_buffer_destroys[i].buf)
      return false;
    }
  }
  _n_deferred_buffer_destroys = 0;

  return true;
}

bool 
cr_mem_staging_ring_begin(struct cr_frame_t* frame) {
  if(!frame) _PARAM_CHECK_FAIL();

  _staging_ring.tail    = frame->staging_end;
  frame->staging_begin  = _staging_ring.head;
  frame->staging_end    = _staging_ring.head; 

  return true;
}

static inline size_t
_align_up(size_t v, size_t alignment) {
  return (v + alignment - 1) & ~(alignment - 1);
}

size_t 
cr_mem_staging_ring_alloc(struct cr_staging_ring_t* ring, size_t n, size_t align) {
  if(!ring) _PARAM_CHECK_FAIL();

  size_t size = _align_up(n, align);

  if (ring->head + size <= ring->capacity) {
    if (ring->head < ring->tail && ring->head + size > ring->tail) {
      return SIZE_MAX;
    }
    size_t off = ring->head;
    ring->head += size;
    return off;
  }

  size_t wrapped_head = 0;

  if (size > ring->tail) {
    return 0; 
  }

  ring->head = size;
  return wrapped_head;
}

bool 
cr_mem_staging_ring_end(struct cr_frame_t* frame) {
  if(!frame) _PARAM_CHECK_FAIL();

  frame->staging_end = _staging_ring.head;

  return true;
}

VmaAllocator 
cr_mem_get_allocator(){ 
  return _vma_allocator;
}
