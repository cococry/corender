#include "../include/corender/corender.h"
#include "../include/corender/util.h"
#include <errno.h>
#include <string.h>
#include <vulkan/vulkan_core.h>


#define _SUBSYS_NAME "CORE"

struct cr_swapchain_info_t {
  VkPresentModeKHR present_modes[16];
  uint32_t n_present_modes;
  VkSurfaceFormatKHR fmts[32];
  uint32_t n_fmts;
  VkSurfaceCapabilitiesKHR caps;
};


static bool _create_log_context(struct cr_context_t* ctx, const struct cr_context_init_info_t* info);
static bool _create_rendering_context(struct cr_context_t* ctx, const struct cr_context_init_info_t* info);
static bool _pick_physical_device(struct cr_context_t* ctx);

static VkResult _create_instance(struct cr_context_t* ctx, const struct cr_context_init_info_t* info);
static VkResult _create_logical_device(struct cr_context_t* ctx);
static VkResult _create_command_pool(struct cr_context_t* ctx);

static bool _get_swapchain_info_from_physical_device(
  struct cr_context_t* ctx,
  VkPhysicalDevice dev, 
  VkSurfaceKHR surf,
  struct cr_swapchain_info_t* o_info 
);

static VkSurfaceFormatKHR _get_swapchain_surface_format(const struct cr_swapchain_info_t* swapchain);
static VkPresentModeKHR _get_swapchain_present_mode(const struct cr_swapchain_info_t* swapchain);
static VkExtent2D _get_swapchain_extent(const struct cr_swapchain_info_t* swapchain, uint32_t w, uint32_t h);

bool _create_swapchain(struct cr_context_t* ctx,  struct cr_swapchain_t* o_swapchain, uint32_t w, uint32_t h);

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
  printf("seg.\n");
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

  VkResult cmd_pool_res = _create_command_pool(ctx); 
  if(cmd_pool_res != VK_SUCCESS) {
    CR_ERROR(ctx->log, "Failed to create Vulkan command pool with graphics queue index %i: (error code: %i)", 
             ctx->graphics_queue_family,
             logical_dev_res);
    return false;
  }

  if(ctx->surf.surf) {
    if(!_create_swapchain(ctx, &ctx->swapchain, ctx->surf.width, ctx->surf.height)) {
      CR_ERROR(ctx->log, "Failed to create Vulkan swap chain (width: %i, height: %i)", 
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

VkResult 
_create_command_pool(struct cr_context_t* ctx) {
  VkCommandPoolCreateInfo pool_info = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    .queueFamilyIndex = ctx->graphics_queue_family,
    .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
  };

  VkResult res = vkCreateCommandPool(ctx->logical_dev, &pool_info, NULL, &ctx->cmd_pool);

  if(res == VK_SUCCESS) { 
    CR_TRACE(ctx->log, "Initialized Vulkan command pool.");
  }

  return res;
}

bool
_get_swapchain_info_from_physical_device(
  struct cr_context_t* ctx,
  VkPhysicalDevice dev, 
  VkSurfaceKHR surf,
struct cr_swapchain_info_t* o_info 
) {
  bool success = true;
  {
    VkResult res = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev, surf, &o_info->caps);
    if(res != VK_SUCCESS) {
      CR_ERROR(ctx->log, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR() failed: (error code: %i)", res);
      success = false;
    }
  }

  {
    VkResult res = vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surf, &o_info->n_fmts, NULL);
    if(res != VK_SUCCESS) {
      CR_ERROR(ctx->log, "vkGetPhysicalDeviceSurfaceFormatsKHR() failed: (error code: %i)", res);
      success = false;
    }
  }
  {
    VkResult res = vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surf, &o_info->n_fmts, o_info->fmts);
    if(res != VK_SUCCESS) {
      CR_ERROR(ctx->log, "vkGetPhysicalDeviceSurfaceFormatsKHR() failed: (error code: %i)", res);
      success = false;
    }
  }

  {
    VkResult res = vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surf, &o_info->n_present_modes, NULL);
    if(res != VK_SUCCESS) {
      CR_ERROR(ctx->log, "vkGetPhysicalDeviceSurfacePresentModesKHR() failed: (error code: %i)", res);
      success = false;
    }
  }

  {
    VkResult res = vkGetPhysicalDeviceSurfacePresentModesKHR(
      dev, surf, &o_info->n_present_modes, o_info->present_modes);
    if(res != VK_SUCCESS) {
      CR_ERROR(ctx->log, "vkGetPhysicalDeviceSurfacePresentModesKHR() failed: (error code: %i)", res);
      success = false;
    }
  }

  return success;
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
  
  VkResult res = vkCreateSwapchainKHR(ctx->logical_dev, &create_info, NULL, &o_swapchain->swapchain_handle); 
  if(res != VK_SUCCESS) {
    CR_ERROR(ctx->log, "Failed to create Vulkan swap chain (error code: %i)", res);
    return false;
  }

  vkGetSwapchainImagesKHR(ctx->logical_dev, o_swapchain->swapchain_handle, &o_swapchain->n_imgs, NULL);
  o_swapchain->imgs = calloc(o_swapchain->n_imgs, sizeof(VkImage));
  vkGetSwapchainImagesKHR(ctx->logical_dev, o_swapchain->swapchain_handle, &o_swapchain->n_imgs, o_swapchain->imgs);


  o_swapchain->img_views = calloc(o_swapchain->n_imgs, sizeof(VkImageView));

  o_swapchain->present_mode = present_mode;
  o_swapchain->fmt = fmt.format;
  o_swapchain->dimensions = extent;

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
