#pragma once

#include <vulkan/vulkan_core.h>

#include <stdbool.h>

#include "../include/corender/corender.h"

enum cr_gpu_buffer_type_t {
  CR_GPU_BUFFER_NO_TYPE = 0,
  CR_GPU_BUFFER_INDEX,
  CR_GPU_BUFFER_VERTEX,
  CR_GPU_BUFFER_INDIRECT,
};

enum cr_gpu_buffer_memory_type_t {
  CR_GPU_BUFFER_MEM_DEVICE_LOCAL = 0,
  CR_GPU_BUFFER_MEM_STAGING,
};

struct cr_upload_context_t {
  VkCommandPool cmd_pool;
  VkCommandBuffer cmd_buf;
  VkFence fence;
};

struct cr_gpu_buffer_t {
  VkBuffer buf;
  size_t buf_size;
  void* mem_handle;
  
  enum cr_gpu_buffer_type_t type;
  enum cr_gpu_buffer_memory_type_t mem_type;

  void* _vma_allocation;
  VkBufferUsageFlags _usage;
  VkMemoryPropertyFlags _mem_props;
};

struct cr_staging_ring_t {
  struct cr_gpu_buffer_t buf;
  VkDeviceSize tail, head, capacity;
};

bool cr_mem_init(struct cr_context_t* ctx); 

bool cr_mem_create_gpu_buffer(
  struct cr_context_t* ctx, VkDeviceSize size, 
  enum cr_gpu_buffer_type_t type,
  enum cr_gpu_buffer_memory_type_t mem_type,
  struct cr_gpu_buffer_t* o_buf
); 

bool cr_mem_create_gpu_buffer_device_local(
  struct cr_context_t* ctx, 
  void* data, VkDeviceSize size, 
  enum cr_gpu_buffer_type_t type, 
  struct cr_gpu_buffer_t* o_buf);

bool cr_mem_destroy_gpu_buffer(struct cr_context_t* ctx, struct cr_gpu_buffer_t* buf);

bool cr_mem_resize_gpu_buffer(struct cr_context_t* ctx, struct cr_gpu_buffer_t* buf, VkDeviceSize new_size); 

size_t 
cr_mem_staging_ring_alloc(struct cr_staging_ring_t* ring, size_t n, size_t align);

bool 
cr_mem_upload_to_device_local_gpu_buffer(
  struct cr_context_t* ctx, 
  void* data, VkDeviceSize size, 
  enum cr_gpu_buffer_type_t type, 
  struct cr_gpu_buffer_t* o_buf);

bool 
cr_mem_upadate_lazy_destroys(struct cr_context_t* ctx);
