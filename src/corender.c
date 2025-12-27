#include "../include/corender/corender.h"
#include "../include/corender/util.h"
#include <errno.h>
#include <string.h>
#include <vulkan/vulkan_core.h>


#define _SUBSYS_NAME "CORE"

#define _VK_CHECK(ctx, expr)                              \
do {                                                      \
  VkResult _res = (expr);                                 \
  if (_res != VK_SUCCESS) {                               \
    CR_ERROR(ctx->log, "Vulkan error: %s (%i) - %s failed.", \
    _vk_result_to_string(_res), _res, #expr);              \
    return false;                                         \
  }                                                       \
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
  struct 
  cr_context_t* ctx, struct cr_frameloop_t* o_frameloop, uint32_t graphics_queue_family); 

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
    if(!_create_swapchain(ctx, &ctx->swapchain, ctx->surf.width, ctx->surf.height)) {
      CR_ERROR(ctx->log, "Failed to create Vulkan swap chain (width: %i, height: %i)", 
               ctx->surf.width, ctx->surf.height);
      return false;
    }

    ctx->frameloop.swapchain = ctx->swapchain;

    if(!_create_frameloop(ctx, &ctx->frameloop, ctx->graphics_queue_family)) {
      CR_ERROR(ctx->log, "Failed to create Vulkan frame loop (width: %i, height: %i)", 
               ctx->surf.width, ctx->surf.height);
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
  struct cr_swapchain_info_t info;
  if(!_get_swapchain_info_from_physical_device(ctx, ctx->phys_dev, ctx->surf.surf, &info)) {
    CR_ERROR(ctx->log, "Failed to get swapchain info from physical device.");
    return false;
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

    VkResult view_res = vkCreateImageView(ctx->logical_dev, &view_info, NULL, &o_swapchain->img_views[i]);
    if(view_res != VK_SUCCESS) {
      CR_ERROR(
        ctx->log, 
        "Failed to create Vulkan image view for swapchain image %i (error code: %i)", i, view_res);
      return false;
    }
  }

  CR_TRACE(ctx->log, "Initialized Vulkan swapchain (width: %i, height: %i)", 
             o_swapchain->dimensions.width, o_swapchain->dimensions.height); 


  return true;
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
      o_frameloop->swapchain.logical_dev, &pool_info, NULL, &frame->cmd_pool));

    VkCommandBufferAllocateInfo buf_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = frame->cmd_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1
    };

    _VK_CHECK(ctx, vkAllocateCommandBuffers(o_frameloop->swapchain.logical_dev, &buf_info, &frame->cmd_buf));

    VkSemaphoreCreateInfo sem_info = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };

    _VK_CHECK(ctx, vkCreateSemaphore(
      o_frameloop->swapchain.logical_dev, 
      &sem_info, NULL, &frame->image_available)); 

    frame->render_finished_per_image = calloc(
      o_frameloop->swapchain.n_imgs, 
      sizeof(*frame->render_finished_per_image));

    for(uint32_t i = 0; i < o_frameloop->swapchain.n_imgs; i++) {
      _VK_CHECK(ctx, vkCreateSemaphore(
        o_frameloop->swapchain.logical_dev, &sem_info, NULL, &frame->render_finished_per_image[i]));
    }

    VkFenceCreateInfo fence_info = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      .flags = VK_FENCE_CREATE_SIGNALED_BIT
    };

    _VK_CHECK(ctx, vkCreateFence(
      o_frameloop->swapchain.logical_dev, &fence_info, NULL, &frame->in_flight_fence));
  
    CR_TRACE(ctx->log, "Initialized Vulkan frameloop frame data for frame %i", 
             i); 
  }

  o_frameloop->frame_idx = 0;

  VkAttachmentDescription clear_attachment = {
    .format = o_frameloop->swapchain.fmt,
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

  _VK_CHECK(ctx, vkCreateRenderPass(o_frameloop->swapchain.logical_dev, &pass_info, NULL, &o_frameloop->crnt_pass));

  o_frameloop->fbs = calloc(o_frameloop->swapchain.n_imgs, sizeof(*o_frameloop->fbs));

  for(uint32_t i = 0; i < o_frameloop->swapchain.n_imgs; i++) {
    VkImageView attachments[] = {
      o_frameloop->swapchain.img_views[i]
    };

    VkFramebufferCreateInfo fb_info = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = o_frameloop->crnt_pass,
      .attachmentCount = 1,
      .pAttachments = attachments, 
      .width = o_frameloop->swapchain.dimensions.width,
      .height = o_frameloop->swapchain.dimensions.height,
      .layers = 1
    };

    _VK_CHECK(ctx, vkCreateFramebuffer(o_frameloop->swapchain.logical_dev, &fb_info, NULL, 
                                       &o_frameloop->fbs[i]) != VK_SUCCESS);
    
    CR_TRACE(ctx->log, "Initialized Vulkan frameloop framebuffer for swapchain image view %i", 
             i); 
  }
    
  CR_TRACE(ctx->log, "Initialized Vulkan frameloop."); 

  o_frameloop->swapchain_image_fences = calloc(
    o_frameloop->swapchain.n_imgs, sizeof(*o_frameloop->swapchain_image_fences));
  return true;

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
cr_draw_frame(struct cr_context_t* ctx) {

struct cr_frame_t* frame = &ctx->frameloop.frames[ctx->frameloop.frame_idx];
  _VK_CHECK(ctx, vkWaitForFences(ctx->logical_dev, 1, &frame->in_flight_fence, VK_TRUE, UINT64_MAX));

  uint32_t image_idx = 0;
  VkResult res = vkAcquireNextImageKHR(
    ctx->logical_dev,
    ctx->swapchain.swapchain_handle,
    UINT64_MAX,
    frame->image_available,
    VK_NULL_HANDLE, 
    &image_idx
  );
  if(ctx->frameloop.swapchain_image_fences[image_idx] != VK_NULL_HANDLE) {
    vkWaitForFences(ctx->logical_dev, 1, &ctx->frameloop.swapchain_image_fences[image_idx], VK_TRUE,
                    UINT64_MAX);
  }
  ctx->frameloop.swapchain_image_fences[image_idx] = frame->in_flight_fence;

  if(res == VK_ERROR_OUT_OF_DATE_KHR) {
    return true;
  }
  if(res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR) return false;

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
    .framebuffer = ctx->frameloop.fbs[image_idx],
    .renderArea = {
      .offset = {0, 0},
      .extent = ctx->swapchain.dimensions
    },
    .pClearValues = &clear,
    .clearValueCount = 1
  };

  vkCmdBeginRenderPass(frame->cmd_buf, &renderpass_info, VK_SUBPASS_CONTENTS_INLINE);

  vkCmdEndRenderPass(frame->cmd_buf);
  _VK_CHECK(ctx, vkEndCommandBuffer(frame->cmd_buf));

  VkPipelineStageFlags pipeline_flags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

  VkSubmitInfo submit_info = {
    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
    .waitSemaphoreCount = 1, 
    .pWaitSemaphores = &frame->image_available,
    .signalSemaphoreCount = 1,
    .pSignalSemaphores = &frame->render_finished_per_image[image_idx],
    .pWaitDstStageMask  = &pipeline_flags, 
    .commandBufferCount = 1,
    .pCommandBuffers = &frame->cmd_buf,
  };

  _VK_CHECK(ctx, vkQueueSubmit(ctx->graphics_queue, 1, &submit_info, frame->in_flight_fence));

  VkPresentInfoKHR present_info = {
    .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
    .waitSemaphoreCount = 1,
    .pWaitSemaphores = &frame->render_finished_per_image[image_idx],
    .swapchainCount = 1,
    .pSwapchains = &ctx->swapchain.swapchain_handle,
    .pImageIndices = &image_idx
  };

  _VK_CHECK(ctx, vkQueuePresentKHR(ctx->present_queue, &present_info));

  ctx->frameloop.frame_idx = (ctx->frameloop.frame_idx + 1) % CR_FRAME_COUNT;
  return true;

}
