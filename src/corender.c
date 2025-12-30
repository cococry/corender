#include "../include/corender/corender.h"
#include "../include/corender/util.h"
#include <errno.h>
#include <linux/limits.h>
#include <stddef.h>
#include <string.h>
#include <vulkan/vulkan_core.h>

#define _SUBSYS_NAME "CORE"

#define _VK_CHECK(ctx, expr)                                  \
do {                                                          \
  VkResult _res = (expr);                                     \
  if (_res != VK_SUCCESS) {                                   \
    CR_ERROR(ctx->log, "Vulkan error: %s (%i) - %s failed.",  \
    _vk_result_to_string(_res), _res, #expr);                 \
    goto err;                                                 \
  }                                                           \
} while (0)

struct cr_swapchain_info_t {
  VkPresentModeKHR present_modes[16];
  uint32_t n_present_modes;
  VkSurfaceFormatKHR fmts[32];
  uint32_t n_fmts;
  VkSurfaceCapabilitiesKHR caps;
};

static bool     _create_log_context(struct cr_context_t* ctx, const struct cr_context_init_info_t* info);
static bool     _create_rendering_context(struct cr_context_t* ctx, const struct cr_context_init_info_t* info);
static VkResult _create_instance(struct cr_context_t* ctx, const struct cr_context_init_info_t* info);
static VkResult _create_logical_device(struct cr_context_t* ctx);
static bool     _create_swapchain(struct cr_context_t* ctx,  struct cr_swapchain_t* o_swapchain, uint32_t w, uint32_t h);
static bool     _create_frameloop(
  struct cr_context_t* ctx, struct cr_frameloop_t* o_frameloop, 
  uint32_t graphics_queue_family); 
static bool     _create_shader_module(struct 
  cr_context_t* ctx, const char* filepath, VkShaderModule* o_module); 
static bool     _create_pipeline(struct 
  cr_context_t* ctx); 
static bool     _create_gpu_buffer(
  struct 
  cr_context_t* ctx, 
  VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags mem_props,
  struct cr_gpu_buffer_t* o_buf
); 

static bool _handle_resize(struct cr_context_t* ctx);
static bool _render_flush(struct cr_context_t* ctx);

static bool _pick_physical_device(struct cr_context_t* ctx);
static bool _get_swapchain_info_from_physical_device(
  struct cr_context_t* ctx,
  VkPhysicalDevice dev, 
  VkSurfaceKHR surf,
  struct cr_swapchain_info_t* o_info 
);

static VkSurfaceFormatKHR _get_swapchain_surface_format(const struct cr_swapchain_info_t* swapchain);
static VkPresentModeKHR   _get_swapchain_present_mode(const struct cr_swapchain_info_t* swapchain);
static VkExtent2D         _get_swapchain_extent(
  const struct cr_swapchain_info_t* swapchain, uint32_t w, uint32_t h);

static const char* _vk_result_to_string(VkResult r);
  
struct _push_constant {
  vec2 scale, offset;
};

bool 
_create_log_context(struct cr_context_t* ctx, const struct cr_context_init_info_t* info) {
  if(!ctx || !info) return false;
  if(info->log_to_file) {
    ctx->log.stream = fopen(cr_util_log_get_filepath(), "a");
    if(!ctx->log.stream) return false;
      
    if(setvbuf(ctx->log.stream, NULL, _IONBF, 0) != 0) {
      CR_ERROR(ctx->log, "setvbuf() failed: %s", strerror(errno));
      return false;
    }
  } else {
    ctx->log.stream = stdout;
  }

  ctx->log.quiet = info->log_quiet;
  ctx->log.verbose = info->log_verbose;

  CR_TRACE(ctx->log, "Initialized log-state: (verbose: %s, quiet: %s, log-to-file: %s)", 
           ctx->log.verbose ? "true" : "false",
           ctx->log.quiet ? "true" : "false",
           info->log_to_file ? "true" : "false");;

  return true;
}

  bool 
_create_rendering_context(struct cr_context_t* ctx, const struct cr_context_init_info_t* info) {
  VkResult instance_res = _create_instance(ctx, info); 
  if(instance_res != VK_SUCCESS) {
    CR_ERROR(ctx->log, "Failed to create Vulkan instance: (error code: %i)", instance_res);
    return false;
  } 

  if(!info->surface_create) {
    CR_FATAL(ctx->log, "info->surface_create is NULL, you need to provide a surface creation function.") ;
    return false;
  }

  if(!info->surface_create(ctx->instance, &ctx->surf, info->surface_userdata)) {
    CR_ERROR(ctx->log, "Failed to create platform surface.");
    return false;
  }

  if(!_pick_physical_device(ctx)) {
    CR_ERROR(ctx->log, "Failed to pick Vulkan physical device.");
  }

  VkResult logical_dev_res = _create_logical_device(ctx); 
  if(logical_dev_res != VK_SUCCESS) {
    CR_ERROR(ctx->log, "Failed to create Vulkan logical device: (error code: %i)", logical_dev_res);
    return false;
  }

  if(ctx->surf.surf) {
    ctx->swapchain = (struct cr_swapchain_t){0};
    if(!_create_swapchain(ctx, &ctx->swapchain, ctx->surf.width, ctx->surf.height)) {
      CR_ERROR(ctx->log, "Failed to create Vulkan swap chain (width: %i, height: %i)", 
               ctx->surf.width, ctx->surf.height);
      return false;
    }

    if(!_create_frameloop(ctx, &ctx->frameloop, ctx->graphics_queue_family)) {
      CR_ERROR(ctx->log, "Failed to create Vulkan frame loop (width: %i, height: %i)", 
               ctx->surf.width, ctx->surf.height);
      return false;
    }

    if(!_create_pipeline(ctx)) {
      CR_ERROR(ctx->log, "Failed to create Vulkan graphics pipeline."); 
      return false;
    }
  }

  return true;
}

VkResult
_create_instance(struct cr_context_t* ctx, const struct cr_context_init_info_t* info) {

  VkApplicationInfo app_info = {
    .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
    .pApplicationName   = "corender",
    .applicationVersion = VK_MAKE_VERSION(0, 0, 1),
    .pEngineName        = "corender",
    .engineVersion      = VK_MAKE_VERSION(0, 0, 1),
    .apiVersion         = VK_API_VERSION_1_4
  };
  VkInstanceCreateInfo create_info = {
    .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    .pApplicationInfo = &app_info,
    .enabledExtensionCount = info->n_exts,
    .ppEnabledExtensionNames = info->exts,
    .enabledLayerCount = info->enable_validation ?  info->n_layers : 0,
    .ppEnabledLayerNames = info->enable_validation ? info->layers : NULL
  };

  VkResult res = vkCreateInstance(&create_info, NULL, &ctx->instance); 
  if(res == VK_SUCCESS) {
    CR_TRACE(ctx->log, "Initialized Vulkan instance: (version: 1.3, enabledExtensionCount: %i, enabledLayerCount: %i)",
             create_info.enabledExtensionCount, create_info.enabledLayerCount);
  }

  return res;

}

bool 
_handle_resize(struct cr_context_t* ctx) {
  vkDeviceWaitIdle(ctx->logical_dev);
  for(uint32_t i = 0; i < ctx->swapchain.n_imgs; i++) {
    vkDestroyFramebuffer(
      ctx->swapchain.logical_dev,
      ctx->frameloop.fbs[i], NULL); 
    CR_TRACE(ctx->log, "Destroyed framebuffer for swapchain image %i", i); 

    vkDestroyImageView(ctx->logical_dev, ctx->swapchain.img_views[i], NULL);
    CR_TRACE(ctx->log, "Destroyed image view for swapchain image %i", i); 
  }

  vkDestroySwapchainKHR(ctx->logical_dev, ctx->swapchain.swapchain_handle, NULL);

  if(!_create_swapchain(ctx, &ctx->swapchain, ctx->pending_resize.width, ctx->pending_resize.height)) goto err; 

  free(ctx->frameloop.fbs);
  ctx->frameloop.fbs = calloc(ctx->swapchain.n_imgs, sizeof(*ctx->frameloop.fbs));

  for(uint32_t i = 0; i < ctx->swapchain.n_imgs; i++) {
    VkImageView attachments[] = {
      ctx->swapchain.img_views[i]
    };

    VkFramebufferCreateInfo fb_info = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = ctx->frameloop.crnt_pass,
      .attachmentCount = 1,
      .pAttachments = attachments, 
      .width = ctx->swapchain.dimensions.width,
      .height = ctx->swapchain.dimensions.height,
      .layers = 1
    };

    _VK_CHECK(ctx, vkCreateFramebuffer(ctx->swapchain.logical_dev, &fb_info, NULL, 
                                       &ctx->frameloop.fbs[i]) != VK_SUCCESS);

    CR_TRACE(ctx->log, "Initialized Vulkan frameloop framebuffer for swapchain image view %i", 
             i); 
  }

  memset(&ctx->pending_resize, 0, sizeof(ctx->pending_resize)); 

  memset(
    ctx->frameloop.swapchain_image_fences, 0, 
    ctx->swapchain.n_imgs * sizeof(*ctx->frameloop.swapchain_image_fences));

  ctx->frameloop.frame_idx = 0;
 
  free(ctx->frameloop.swapchain_image_fences);
  ctx->frameloop.swapchain_image_fences = calloc(
    ctx->swapchain.n_imgs, sizeof(*ctx->frameloop.swapchain_image_fences));


  return true;

err:
  return false;
}

bool 
_render_flush(struct cr_context_t* ctx) {
  struct cr_frame_t* frame = &ctx->frameloop.frames[ctx->frameloop.frame_idx];
  VkDeviceSize vbo_offsets[] = { 0 };
  vkCmdBindVertexBuffers(frame->cmd_buf, 0, 1, &frame->batch_state.vbo.buf, vbo_offsets);

  vkCmdBindIndexBuffer(frame->cmd_buf, frame->batch_state.ibo.buf, 0, VK_INDEX_TYPE_UINT32);

  vkCmdDrawIndexed(frame->cmd_buf, frame->batch_state.n_indices, 1, 0, 0, 0); 

  vkCmdEndRenderPass(frame->cmd_buf);
  _VK_CHECK(ctx, vkEndCommandBuffer(frame->cmd_buf));

  VkDeviceSize offset = 0;

  VkPipelineStageFlags pipeline_flags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

  VkSubmitInfo submit_info = {
    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
    .waitSemaphoreCount = 1, 
    .pWaitSemaphores = &frame->image_available,
    .signalSemaphoreCount = 1,
    .pSignalSemaphores = &frame->render_finished_per_image[ctx->_swapchain_img_idx],
    .pWaitDstStageMask  = &pipeline_flags, 
    .commandBufferCount = 1,
    .pCommandBuffers = &frame->cmd_buf,
  };

  _VK_CHECK(ctx, vkQueueSubmit(ctx->graphics_queue, 1, &submit_info, frame->in_flight_fence));

  return true;

err: 
  return false;
}

bool _pick_physical_device(struct cr_context_t* ctx) {
  uint32_t count = 0;
  vkEnumeratePhysicalDevices(ctx->instance, &count, NULL);
  if (count == 0) return false;

  VkPhysicalDevice devices[16];
  vkEnumeratePhysicalDevices(ctx->instance, &count, devices);

  for (uint32_t i = 0; i < count; i++) {
    VkPhysicalDevice dev = devices[i];

    uint32_t qcount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &qcount, NULL);

    VkQueueFamilyProperties qprops[16];
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &qcount, qprops);

    int graphics = -1;
    int present  = -1;

    for (uint32_t q = 0; q < qcount; q++) {
      if (qprops[q].queueFlags & VK_QUEUE_GRAPHICS_BIT)
        graphics = q;

      if (ctx->surf.surf) {
        VkBool32 supported = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(dev, q, ctx->surf.surf, &supported);
        if (supported)
          present = q;
      } else {
        present = graphics;
      }
    }

    if (graphics >= 0 && present >= 0) {
      ctx->phys_dev = dev;
      ctx->graphics_queue_family = graphics;
      ctx->present_queue_family  = present;

      VkPhysicalDeviceProperties props;
      vkGetPhysicalDeviceProperties(dev, &props);
      CR_TRACE(
        ctx->log, 
        "Picked physical device: (name: %s, API version: %i, driver version: %i, present queue: %i, graphics queue: %i)",
        props.deviceName, 
        props.apiVersion,
        props.driverVersion,
        present,
        graphics); 

      return true;
    }
  }

  return false;
}

VkResult
_create_logical_device(struct cr_context_t* ctx) {
  const float priority = 1.0f;
  uint32_t queue_count = 0;
  VkDeviceQueueCreateInfo queues[2];

  queues[queue_count++] = (VkDeviceQueueCreateInfo) {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = ctx->graphics_queue_family,
      .queueCount = 1,
      .pQueuePriorities = &priority
    };

  if(ctx->graphics_queue_family != ctx->present_queue_family) {
    queues[queue_count++] = (VkDeviceQueueCreateInfo) {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = ctx->present_queue_family,
      .queueCount = 1,
      .pQueuePriorities = &priority
    };
  }

  const char* device_exts[] = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
  };

  VkDeviceCreateInfo device_info = {
    .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    .pQueueCreateInfos = queues,
    .queueCreateInfoCount = queue_count, 
    .enabledExtensionCount = ctx->surf.surf ? 1 : 0, 
    .ppEnabledExtensionNames = ctx->surf.surf ? device_exts : NULL
  };

  VkResult res = vkCreateDevice(ctx->phys_dev, &device_info, NULL, &ctx->logical_dev);
  if(res == VK_SUCCESS) {
    CR_TRACE(ctx->log, "Initialized Vulkan logical device (graphics queue index: %i, present queue index; %i)",
             ctx->graphics_queue_family, ctx->present_queue_family);
  }

  vkGetDeviceQueue(ctx->logical_dev, ctx->graphics_queue_family, 0, &ctx->graphics_queue);
  vkGetDeviceQueue(ctx->logical_dev, ctx->present_queue_family, 0, &ctx->present_queue);

  return res;
}


bool
_get_swapchain_info_from_physical_device(
  struct cr_context_t* ctx,
  VkPhysicalDevice dev, 
  VkSurfaceKHR surf,
struct cr_swapchain_info_t* o_info 
) {
  _VK_CHECK(ctx, vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev, surf, &o_info->caps));
  _VK_CHECK(ctx, vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surf, &o_info->n_fmts, NULL));
  _VK_CHECK(ctx, vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surf, &o_info->n_fmts, o_info->fmts));
  _VK_CHECK(ctx, vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surf, &o_info->n_present_modes, NULL));
  _VK_CHECK(ctx, vkGetPhysicalDeviceSurfacePresentModesKHR(
    dev, surf, &o_info->n_present_modes, o_info->present_modes));

  return true;
err:
  return false;
}

VkSurfaceFormatKHR 
_get_swapchain_surface_format(const struct cr_swapchain_info_t* swapchain) {
  for(uint32_t i = 0; i < swapchain->n_fmts; i++) {
    if(swapchain->fmts[i].format == VK_FORMAT_B8G8R8_SRGB && 
       swapchain->fmts[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) return swapchain->fmts[i];
  }
  return swapchain->fmts[0];
}

VkPresentModeKHR 
_get_swapchain_present_mode(const struct cr_swapchain_info_t* swapchain) {
  for(uint32_t i = 0; i < swapchain->n_fmts; i++) {
    if(swapchain->present_modes[i] == VK_PRESENT_MODE_MAILBOX_KHR) 
      return swapchain->present_modes[i]; 
  }
  return VK_PRESENT_MODE_FIFO_KHR;

}

VkExtent2D 
_get_swapchain_extent(const struct cr_swapchain_info_t* swapchain, uint32_t w, uint32_t h) {
  if(swapchain->caps.currentExtent.width != UINT32_MAX) return swapchain->caps.currentExtent;

  VkExtent2D extent = (VkExtent2D){
    .width = w,
    .height = h
  };

  extent.width = CR_MIN(swapchain->caps.maxImageExtent.width, extent.width); 
  extent.height = CR_MIN(swapchain->caps.maxImageExtent.height, extent.height); 
  extent.width = CR_MAX(swapchain->caps.minImageExtent.width, extent.width); 
  extent.height = CR_MAX(swapchain->caps.minImageExtent.height, extent.height); 

  return extent; 
}


const char* _vk_result_to_string(VkResult r)
{
    switch (r) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_NOT_READY: return "VK_NOT_READY";
        case VK_TIMEOUT: return "VK_TIMEOUT";
        case VK_EVENT_SET: return "VK_EVENT_SET";
        case VK_EVENT_RESET: return "VK_EVENT_RESET";
        case VK_INCOMPLETE: return "VK_INCOMPLETE";

        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";

        case VK_ERROR_OUT_OF_DATE_KHR: return "VK_ERROR_OUT_OF_DATE_KHR";
        case VK_SUBOPTIMAL_KHR: return "VK_SUBOPTIMAL_KHR";

        default: return "VK_ERROR_UNKNOWN";
    }
}


bool 
_create_swapchain(struct cr_context_t* ctx,  struct cr_swapchain_t* o_swapchain, uint32_t w, uint32_t h) {
  if(ctx->swapchain.imgs) free(ctx->swapchain.imgs);
  if(ctx->swapchain.img_views) free(ctx->swapchain.img_views);
  memset(&ctx->swapchain, 0, sizeof(ctx->swapchain));

  struct cr_swapchain_info_t info;
  if(!_get_swapchain_info_from_physical_device(ctx, ctx->phys_dev, ctx->surf.surf, &info)) {
    CR_ERROR(ctx->log, "Failed to get swapchain info from physical device.");
    goto err;
  }

  VkSurfaceFormatKHR fmt = _get_swapchain_surface_format(&info);
  VkPresentModeKHR present_mode = _get_swapchain_present_mode(&info);
  VkExtent2D extent = _get_swapchain_extent(&info, w, h);

  uint32_t n_imgs = info.caps.minImageCount + 1;
  if(info.caps.maxImageCount > 0 && n_imgs > info.caps.maxImageCount) {
    n_imgs = info.caps.maxImageCount;
  }

  VkSwapchainCreateInfoKHR create_info = {
    .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR, 
    .surface = ctx->surf.surf,
    .minImageCount = n_imgs,
    .imageFormat = fmt.format, 
    .imageColorSpace = fmt.colorSpace,
    .imageExtent = extent,
    .imageArrayLayers = 1,
    .imageUsage   = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
    .preTransform = info.caps.currentTransform,
    .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
    .presentMode = present_mode,
    .clipped = VK_TRUE
  };

  if(ctx->graphics_queue_family != ctx->present_queue_family) {
    create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    create_info.queueFamilyIndexCount = 2;
    uint32_t families[2] = {
      ctx->graphics_queue_family,
      ctx->present_queue_family
    };
    create_info.pQueueFamilyIndices = families;
  } else {
    create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  }
  
  _VK_CHECK(ctx, vkCreateSwapchainKHR(ctx->logical_dev, &create_info, NULL, &o_swapchain->swapchain_handle));

  vkGetSwapchainImagesKHR(ctx->logical_dev, o_swapchain->swapchain_handle, &o_swapchain->n_imgs, NULL);
  o_swapchain->imgs = calloc(o_swapchain->n_imgs, sizeof(VkImage));
  vkGetSwapchainImagesKHR(ctx->logical_dev, o_swapchain->swapchain_handle, &o_swapchain->n_imgs, o_swapchain->imgs);


  o_swapchain->img_views = calloc(o_swapchain->n_imgs, sizeof(VkImageView));

  o_swapchain->present_mode = present_mode;
  o_swapchain->fmt = fmt.format;
  o_swapchain->dimensions = extent;
  o_swapchain->logical_dev = ctx->logical_dev;

  for(uint32_t i = 0; i < o_swapchain->n_imgs; i++) {
    VkImageViewCreateInfo view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = o_swapchain->imgs[i],
      .format = o_swapchain->fmt,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .subresourceRange = {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .levelCount = 1, 
        .layerCount = 1
      }
    };

    _VK_CHECK(ctx, vkCreateImageView(ctx->logical_dev, &view_info, NULL, &o_swapchain->img_views[i]));
  }

  CR_TRACE(ctx->log, "Initialized Vulkan swapchain (width: %i, height: %i)", 
             o_swapchain->dimensions.width, o_swapchain->dimensions.height);

  return true;

err:
  return false;

}

bool
_create_frameloop(struct cr_context_t* ctx, struct cr_frameloop_t* o_frameloop, uint32_t graphics_queue_family) {
  VkCommandPoolCreateInfo pool_info = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    .queueFamilyIndex = graphics_queue_family, 
    .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
  };

  for(uint32_t i = 0; i < CR_FRAME_COUNT; i++) {
    struct cr_frame_t* frame = &o_frameloop->frames[i];
  
    _VK_CHECK(ctx, vkCreateCommandPool(
      ctx->swapchain.logical_dev, &pool_info, NULL, &frame->cmd_pool));

    VkCommandBufferAllocateInfo buf_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = frame->cmd_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1
    };

    _VK_CHECK(ctx, vkAllocateCommandBuffers(ctx->swapchain.logical_dev, &buf_info, &frame->cmd_buf));

    VkSemaphoreCreateInfo sem_info = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };

    _VK_CHECK(ctx, vkCreateSemaphore(
      ctx->swapchain.logical_dev, 
      &sem_info, NULL, &frame->image_available)); 

    frame->render_finished_per_image = calloc(
      ctx->swapchain.n_imgs, 
      sizeof(*frame->render_finished_per_image));

    for(uint32_t i = 0; i < ctx->swapchain.n_imgs; i++) {
      _VK_CHECK(ctx, vkCreateSemaphore(
        ctx->swapchain.logical_dev, &sem_info, NULL, &frame->render_finished_per_image[i]));
    }

    VkFenceCreateInfo fence_info = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      .flags = VK_FENCE_CREATE_SIGNALED_BIT
    };

    _VK_CHECK(ctx, vkCreateFence(
      ctx->swapchain.logical_dev, &fence_info, NULL, &frame->in_flight_fence));
  
    CR_TRACE(ctx->log, "Initialized Vulkan frameloop frame data for frame %i", 
             i);

    frame->batch_state = (struct cr_batch_state_t){0};
    frame->batch_state.vert_max = CR_MAX_BATCH * 4; 
    frame->batch_state.indicies_max = CR_MAX_BATCH * 6; 

    _create_gpu_buffer(
      ctx, frame->batch_state.vert_max * sizeof(struct cr_vertex_t), 
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, 
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &frame->batch_state.vbo);

    _create_gpu_buffer(
      ctx, frame->batch_state.indicies_max * sizeof(uint32_t), 
      VK_BUFFER_USAGE_INDEX_BUFFER_BIT, 
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &frame->batch_state.ibo);

    uint32_t* indices_mapped = (uint32_t*)frame->batch_state.ibo.mem_handle;

    for(uint32_t j = 0; j < CR_MAX_BATCH; j++) {

      uint32_t idx = j * 6;
      uint32_t vert = j * 4;
      indices_mapped[idx + 0] = vert + 0;
      indices_mapped[idx + 1] = vert + 1;
      indices_mapped[idx + 2] = vert + 2;

      indices_mapped[idx + 3] = vert + 2;
      indices_mapped[idx + 4] = vert + 3;
      indices_mapped[idx + 5] = vert + 0;
    }
  }

  o_frameloop->frame_idx = 0;

  VkAttachmentDescription clear_attachment = {
    .format = ctx->swapchain.fmt,
    .samples = VK_SAMPLE_COUNT_1_BIT, 

    .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
    .storeOp = VK_ATTACHMENT_STORE_OP_STORE,

    .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
    .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,

    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,

    .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
  };

  VkAttachmentReference clear_reference = {
    .attachment = 0,
    .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL 
  };

  VkSubpassDescription subpass_desc = {
    .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
    .colorAttachmentCount = 1,
    .pColorAttachments = &clear_reference,
  };

  VkSubpassDependency dep = {
    .srcSubpass = VK_SUBPASS_EXTERNAL,
    .dstSubpass = 0,

    .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
    .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,

    .srcAccessMask = 0,
    .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
  };

  VkRenderPassCreateInfo pass_info = {
    .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
    .attachmentCount = 1,
    .pAttachments = &clear_attachment,
    .subpassCount = 1,
    .pSubpasses = &subpass_desc,
    .dependencyCount = 1,
    .pDependencies = &dep,
  };

  _VK_CHECK(
    ctx, 
    vkCreateRenderPass(ctx->swapchain.logical_dev, &pass_info, NULL, &o_frameloop->crnt_pass));

  o_frameloop->fbs = calloc(ctx->swapchain.n_imgs, sizeof(*o_frameloop->fbs));


  for(uint32_t i = 0; i < ctx->swapchain.n_imgs; i++) {
    VkImageView attachments[] = {
      ctx->swapchain.img_views[i]
    };

    VkFramebufferCreateInfo fb_info = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = o_frameloop->crnt_pass,
      .attachmentCount = 1,
      .pAttachments = attachments, 
      .width = ctx->swapchain.dimensions.width,
      .height = ctx->swapchain.dimensions.height,
      .layers = 1
    };

    _VK_CHECK(ctx, vkCreateFramebuffer(ctx->swapchain.logical_dev, &fb_info, NULL, 
                                       &o_frameloop->fbs[i]) != VK_SUCCESS);
    
    CR_TRACE(ctx->log, "Initialized Vulkan frameloop framebuffer for swapchain image view %i", 
             i); 
  }
    
  CR_TRACE(ctx->log, "Initialized Vulkan frameloop."); 

  o_frameloop->swapchain_image_fences = calloc(
    ctx->swapchain.n_imgs, sizeof(*o_frameloop->swapchain_image_fences));

  return true;

err:
  if(o_frameloop->swapchain_image_fences) free(o_frameloop->swapchain_image_fences);
  for(uint32_t i = 0; i < CR_FRAME_COUNT; i++) {
    struct cr_frame_t* frame = &o_frameloop->frames[i];
    if(frame->render_finished_per_image) free(frame->render_finished_per_image);
  }

  return false;
}

bool 
_create_shader_module(struct 
  cr_context_t* ctx, const char* filepath, VkShaderModule* o_module) {
  size_t file_size;
  printf("filepath: %s\n", filepath);
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
_create_pipeline(struct cr_context_t* ctx) {

  VkShaderModule vert_mod, frag_mod;

  const char* state_dir = cr_util_get_state_folder();
  char shader_dir[PATH_MAX];
  snprintf(shader_dir, sizeof(shader_dir), "%s/%s/shaders", state_dir, _CR_BRAND_NAME);

  char vert_src[PATH_MAX];
  snprintf(vert_src, sizeof(vert_src), "%s/basic_vert.spv", shader_dir);
  char frag_src[PATH_MAX];
  snprintf(frag_src, sizeof(frag_src), "%s/basic_frag.spv", shader_dir);

  if(!_create_shader_module(ctx, vert_src, &vert_mod)) {
    CR_ERROR(ctx->log, "Failed to create vertex shader byte code for file '%s'", 
             vert_src);
    goto err;
  }
  if(!_create_shader_module(ctx, frag_src, &frag_mod)) {
    CR_ERROR(ctx->log, "Failed to create fragment shader byte code for file '%s'", 
             frag_src);
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


  VkVertexInputBindingDescription binding_desc = {
    .binding = 0,
    .stride = sizeof(struct cr_vertex_t),
    .inputRate = VK_VERTEX_INPUT_RATE_VERTEX
  };

  VkVertexInputAttributeDescription vert_attrs[2] = {
    (VkVertexInputAttributeDescription){
      .location = 0,
      .binding = 0,
      .format = VK_FORMAT_R32G32_SFLOAT,
      .offset = offsetof(struct cr_vertex_t, pos)
    },
    (VkVertexInputAttributeDescription){
      .location = 1,
      .binding = 0,
      .format = VK_FORMAT_R32G32B32A32_SFLOAT,
      .offset = offsetof(struct cr_vertex_t, color)
    },
  };

  VkPipelineVertexInputStateCreateInfo vertex_input_state = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    .vertexBindingDescriptionCount = 1,
    .pVertexBindingDescriptions = &binding_desc,
    .vertexAttributeDescriptionCount = 2,
    .pVertexAttributeDescriptions = vert_attrs
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

 
  VkPushConstantRange range = {
    .offset = 0,
    .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
    .size = sizeof(struct _push_constant) 
  };

  VkPipelineLayoutCreateInfo layout_info = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    .pPushConstantRanges = &range,
    .pushConstantRangeCount = 1
  };

  _VK_CHECK(ctx, vkCreatePipelineLayout(ctx->logical_dev, &layout_info, NULL, &ctx->pipeline_layout));

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

  _VK_CHECK(ctx, vkCreateGraphicsPipelines(ctx->logical_dev, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &ctx->pipeline));

  vkDestroyShaderModule(ctx->logical_dev, vert_mod, NULL);
  vkDestroyShaderModule(ctx->logical_dev, frag_mod, NULL);

  CR_TRACE(ctx->log, "Initialized Vulkan graphics pipeline for surface %p.", ctx->surf.surf);

  return true;

err:
  return false;
}

static uint32_t 
find_ideal_memory_type_index(
  struct cr_context_t* ctx,
  VkPhysicalDevice phys_dev, uint32_t preferred_type,
  VkMemoryPropertyFlags mem_props) {

  VkPhysicalDeviceMemoryProperties phys_mem_props;
  vkGetPhysicalDeviceMemoryProperties(phys_dev, &phys_mem_props);

  for(uint32_t i = 0; i < phys_mem_props.memoryTypeCount; i++) {
    if((preferred_type & (1 << i)) && 
       (phys_mem_props.memoryTypes[i].propertyFlags & mem_props) == mem_props) {
      return i;
    }
  }
  CR_FATAL(ctx->log, "Failed to find suitable memory type for buffer.");
  return 0;
}

bool 
_create_gpu_buffer(
  struct 
  cr_context_t* ctx, 
  VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags mem_props,
  struct cr_gpu_buffer_t* o_buf
) {

  VkBufferCreateInfo buffer_info = {
    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    .size = size,
    .usage = usage,
    .sharingMode = VK_SHARING_MODE_EXCLUSIVE
  };

  _VK_CHECK(ctx, vkCreateBuffer(ctx->logical_dev, &buffer_info, NULL, &o_buf->buf));

  VkMemoryRequirements buf_req;
  vkGetBufferMemoryRequirements(ctx->logical_dev, o_buf->buf, &buf_req);

  VkMemoryAllocateInfo alloc_info = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    .allocationSize = buf_req.size,
    .memoryTypeIndex = find_ideal_memory_type_index(
      ctx, ctx->phys_dev, buf_req.memoryTypeBits, mem_props
    ) 
  };

  _VK_CHECK(ctx, vkAllocateMemory(ctx->logical_dev, &alloc_info, NULL, &o_buf->vk_mem));

  vkBindBufferMemory(ctx->logical_dev, o_buf->buf, o_buf->vk_mem, 0);

  _VK_CHECK(ctx, vkMapMemory(ctx->logical_dev, o_buf->vk_mem, 0, size, 0, &o_buf->mem_handle));


  CR_TRACE(ctx->log, "Successfully allocated GPU buffer with size %li", size);
  return true;
err:
  return false;

}

bool 
cr_context_create(struct cr_context_t* ctx, const struct cr_context_init_info_t* info) {
  memset(ctx, 0, sizeof *ctx);
  if(!_create_log_context(ctx, info)) {
    CR_ERROR(ctx->log, "Failed to create logging context.");
    return false;
  }
  if(!_create_rendering_context(ctx, info)) {
    CR_ERROR(ctx->log, "Failed to create rendering context.");
    return false;
  } 

  return true;
}
bool 
cr_context_destroy(struct cr_context_t* ctx) {
  return true;
}

bool  
cr_begin(struct cr_context_t* ctx) {
  struct cr_frame_t* frame = &ctx->frameloop.frames[ctx->frameloop.frame_idx];

  vkWaitForFences(ctx->logical_dev, 1, &frame->in_flight_fence, VK_TRUE, UINT64_MAX);

  ctx->_swapchain_img_idx = 0;
  VkResult res = vkAcquireNextImageKHR(
    ctx->logical_dev,
    ctx->swapchain.swapchain_handle,
    UINT64_MAX,
    frame->image_available,
    VK_NULL_HANDLE, 
    &ctx->_swapchain_img_idx
  );
  if(ctx->frameloop.swapchain_image_fences[ctx->_swapchain_img_idx] != VK_NULL_HANDLE) {
    vkWaitForFences(ctx->logical_dev, 1, &ctx->frameloop.swapchain_image_fences[ctx->_swapchain_img_idx], VK_TRUE,
                    UINT64_MAX);
  }
  ctx->frameloop.swapchain_image_fences[ctx->_swapchain_img_idx] = frame->in_flight_fence;

  if(res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR) {
    if(!ctx->pending_resize.pending) {
      CR_ERROR(ctx->log, "A Vulkan swapchain resize is necessary but cr_resize_surface() was never invoked by the client. Call cr_resize_surface() in the resize handler of your surface provider."); 
      goto err;
    }
    if(!_handle_resize(ctx)) {
      CR_ERROR(ctx->log, "Failed to handle resize to size: %ix%i. ", ctx->pending_resize.width, ctx->pending_resize.height);
      goto err;
    }
    return true;
  }
  if(res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR) goto err;

  frame->batch_state.n_vertices = 0;
  frame->batch_state.n_indices = 0;

  _VK_CHECK(ctx, vkResetFences(ctx->logical_dev, 1, &frame->in_flight_fence));
  _VK_CHECK(ctx, vkResetCommandPool(ctx->logical_dev, frame->cmd_pool, 0));

  VkCommandBufferBeginInfo begin_info = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
  };

  _VK_CHECK(ctx, vkBeginCommandBuffer(frame->cmd_buf, &begin_info));

  VkClearValue clear = {
    .color = {
      { 0.1f, 0.1f, 0.1f, 1.0f}
    }
  };
  VkRenderPassBeginInfo renderpass_info = {
    .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
    .renderPass = ctx->frameloop.crnt_pass,
    .framebuffer = ctx->frameloop.fbs[ctx->_swapchain_img_idx],
    .renderArea = {
      .offset = {0, 0},
      .extent = ctx->swapchain.dimensions
    },
    .pClearValues = &clear,
    .clearValueCount = 1
  };

  vkCmdBeginRenderPass(frame->cmd_buf, &renderpass_info, VK_SUBPASS_CONTENTS_INLINE);

  vkCmdBindPipeline(frame->cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS, ctx->pipeline);

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

  struct _push_constant pc = {
    .scale = { 2.0f / ctx->swapchain.dimensions.width,  2.0f / ctx->swapchain.dimensions.height},
    .offset = { -1.0f, -1.0f}
  };

  vkCmdPushConstants(
    frame->cmd_buf, 
    ctx->pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc); 

  return true;

err: 
  return false;
}

void 
cr_draw_rect(struct cr_context_t* ctx, vec2 pos, vec2 size, vec4 color) {
  struct cr_frame_t* frame = &ctx->frameloop.frames[ctx->frameloop.frame_idx];
  struct cr_vertex_t* vertices =  frame->batch_state.vbo.mem_handle;

  vertices[frame->batch_state.n_vertices++] = (struct cr_vertex_t){
    .pos = { pos[0], pos[1] },
    .color = { color[0], color[1], color[2], color[3] }  
  };
  vertices[frame->batch_state.n_vertices++] = (struct cr_vertex_t){
    .pos = { pos[0] + size[0], pos[1] },
    .color = { color[0], color[1], color[2], color[3] }  
  };
  vertices[frame->batch_state.n_vertices++] = (struct cr_vertex_t){
    .pos = { pos[0] + size[0], pos[1] + size[1] },
    .color = { color[0], color[1], color[2], color[3] }  
  };
  vertices[frame->batch_state.n_vertices++] = (struct cr_vertex_t){
    .pos = { pos[0], pos[1] + size[1] },
    .color = { color[0], color[1], color[2], color[3] }  
  };

  frame->batch_state.n_indices += 6;
}


bool 
cr_end(struct cr_context_t* ctx) {
  struct cr_frame_t* frame = &ctx->frameloop.frames[ctx->frameloop.frame_idx];

  if(!_render_flush(ctx)) {
    CR_ERROR(ctx->log, "Failed to flush the renderer.");
    goto err;
  }
   VkPresentInfoKHR present_info = {
    .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
    .waitSemaphoreCount = 1,
    .pWaitSemaphores = &frame->render_finished_per_image[ctx->_swapchain_img_idx],
    .swapchainCount = 1,
    .pSwapchains = &ctx->swapchain.swapchain_handle,
    .pImageIndices = &ctx->_swapchain_img_idx
  };

  vkQueuePresentKHR(ctx->present_queue, &present_info);

  ctx->frameloop.frame_idx = (ctx->frameloop.frame_idx + 1) % CR_FRAME_COUNT;

  return true;

err:
  return false;
}

void 
cr_resize_surface(struct cr_context_t* ctx, uint32_t width, uint32_t height) {
  ctx->pending_resize.pending = true;
  ctx->pending_resize.width = width;
  ctx->pending_resize.height = height;

  ctx->surf.width = width;
}
