#include <GLFW/glfw3.h>
#include <errno.h>
#include <linux/limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <vulkan/vulkan_core.h>

#include "../vendor/vma/vk_mem_alloc.h"
#include "../include/corender/corender.h"
#include "compute.h"
#include "raster.h"
#include "raster.h"
#include "util.h"
#include "mem.h"

#define _SUBSYS_NAME "CORE"

#define _MAX_BINDING_DESC 2
#define _MAX_VERT_ATTRS 5

struct _swapchain_info_t {
  VkPresentModeKHR present_modes[16];
  uint32_t n_present_modes;
  VkSurfaceFormatKHR fmts[32];
  uint32_t n_fmts;
  VkSurfaceCapabilitiesKHR caps;
};

struct _vertex_input_state_t {
  VkVertexInputBindingDescription binding_desc[_MAX_BINDING_DESC];
  VkVertexInputAttributeDescription vert_attrs[_MAX_VERT_ATTRS];
  VkPipelineVertexInputStateCreateInfo input_state;
};

struct _push_constant_t {
  vec2 scale, offset;
};

struct cr_raster_pipeline_t instanced_raster_pipeline = {0};
struct cr_raster_pipeline_t vertex_raster_pipeline    = {0};
struct cr_compute_pipeline_t compute_pipeline = {0};

static VmaAllocation* _depth_allocs = NULL;
static uint64_t _frame_start_time = 0.0f;

static bool                   _create_log_context(struct cr_context_t* ctx, const struct cr_context_init_info_t* info);
static bool                   _create_rendering_context(struct cr_context_t* ctx, const struct cr_context_init_info_t* info);
static bool                   _create_instance(struct cr_context_t* ctx, const struct cr_context_init_info_t* info);
static bool                   _create_logical_device(struct cr_context_t* ctx);
static bool                   _create_swapchain(struct cr_context_t* ctx,  struct cr_swapchain_t* o_swapchain, uint32_t w, uint32_t h);
static bool                   _create_upload_context(struct cr_context_t* ctx, struct cr_upload_context_t* o_upload, uint32_t graphics_queue_family);
static bool                   _create_frameloop(struct cr_context_t* ctx, struct cr_frameloop_t* o_frameloop, uint32_t graphics_queue_family); 
static bool                   _create_render_raster_pipelines(struct cr_context_t* ctx);
static bool                   _create_render_compute_pipeline(struct cr_context_t* ctx);
static bool                   _create_shader_module(struct cr_context_t* ctx, const char* filepath, VkShaderModule* o_module); 
static bool                   _create_raster_pipeline(struct cr_context_t* ctx, VkPipelineVertexInputStateCreateInfo vertex_input_state,
                                 const char* shader_subpath, VkPipeline* o_raster_pipeline); 
static bool                   _pick_physical_device(struct cr_context_t* ctx);

static bool                   _renderer_handle_resize(struct cr_context_t* ctx);

static bool                   _get_swapchain_info_from_physical_device(struct cr_context_t* ctx, VkPhysicalDevice dev, 
                                                                       VkSurfaceKHR surf, struct _swapchain_info_t* o_info);
static VkSurfaceFormatKHR     _get_swapchain_surface_format(const struct _swapchain_info_t* swapchain);
static VkPresentModeKHR       _get_swapchain_present_mode(const struct _swapchain_info_t* swapchain);
static VkExtent2D             _get_swapchain_extent(
  const struct _swapchain_info_t* swapchain, uint32_t w, uint32_t h);
static void                   _get_vertex_raster_pipeline_input_state(struct _vertex_input_state_t* o_input_state);
static void                   _get_instanced_raster_pipeline_input_state(struct _vertex_input_state_t* o_input_state);

static const char*            _vk_result_to_string(VkResult r);
static bool                   _pick_physical_device(struct cr_context_t* ctx);

static size_t                 _staging_ring_alloc(struct cr_staging_ring_t* ring, size_t n, size_t align);

bool 
_create_log_context(struct cr_context_t* ctx, const struct cr_context_init_info_t* info) {
  if(!ctx || !info) _PARAM_CHECK_FAIL();

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
  if(!ctx || !info) _PARAM_CHECK_FAIL();

  if(!_create_instance(ctx, info)) {
    CR_ERROR(ctx->log, "Failed to create Vulkan instance.");
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

  if(!_create_logical_device(ctx)) {
    CR_ERROR(ctx->log, "Failed to create Vulkan logical device.");
    return false;
  }

  if(!cr_mem_init(ctx)) {
    CR_ERROR(ctx->log, "Failed to create VMA context."); 
    return false;
  }

  if(ctx->surf.surf) {
    ctx->swapchain = (struct cr_swapchain_t){0};
    if(!_create_swapchain(ctx, &ctx->swapchain, ctx->surf.width, ctx->surf.height)) {
      CR_ERROR(ctx->log, "Failed to create Vulkan swap chain (width: %i, height: %i)", 
               ctx->surf.width, ctx->surf.height);
      return false;
    }
  }

  if(!_create_frameloop(ctx, &ctx->frameloop, ctx->graphics_queue_family)) {
    CR_ERROR(ctx->log, "Failed to create Vulkan frame loop (width: %i, height: %i)", 
             ctx->surf.width, ctx->surf.height);
    return false;
  }

  if(!_create_render_raster_pipelines(ctx)) {
    CR_ERROR(ctx->log, "Failed to create batch-rendering pipelines.");
    return false;
  }
  if(!_create_render_compute_pipeline(ctx)) {
    CR_ERROR(ctx->log, "Failed to create compute pipeline.");
    return false;
  }

  if(info->enable_gpu_profiler && !cr_util_global_gpu_profiler_init(ctx)) {
    CR_WARN(ctx->log, "GPU profiler cannot be enabled.");
  }

  cr_draw_set_clear_color(ctx, (vec4){0.1f, 0.1f, 0.1f, 1.0f});

  return true;

err:
  return false;
}


static inline const char* _vk_ver_to_string(uint32_t version, char* out, size_t size) {
  snprintf(out, size, "%u.%u.%u",
           VK_API_VERSION_MAJOR(version),
           VK_API_VERSION_MINOR(version),
           VK_API_VERSION_PATCH(version));
  return out;
}


VKAPI_ATTR VkBool32 VKAPI_CALL
debug_utils_message_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData
) {
    const char* msg = "(null validation message)";

    if (pCallbackData && pCallbackData->pMessage) {
        msg = pCallbackData->pMessage;
    }

    fprintf(stderr, "Validation: %s\n", msg);
    return VK_FALSE;
}

bool
_create_instance(struct cr_context_t* ctx, const struct cr_context_init_info_t* info) {
  if(!ctx || !info) _PARAM_CHECK_FAIL();

  uint32_t ver = VK_API_VERSION_1_2;
  uint32_t ver_ideal = VK_API_VERSION_1_4;
  if(vkEnumerateInstanceVersion(&ver_ideal) == VK_SUCCESS) {
    ver = VK_API_VERSION_1_4;
  }
  VkApplicationInfo app_info = {
    .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
    .pApplicationName   = "corender",
    .applicationVersion = VK_MAKE_VERSION(0, 0, 1),
    .pEngineName        = "corender",
    .engineVersion      = VK_MAKE_VERSION(0, 0, 1),
    .apiVersion         = ver 
  };


  VkValidationFeatureEnableEXT validation_feature_enables[] = {
      VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT
  };

  VkValidationFeaturesEXT validation_features = {0};
  validation_features.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
  validation_features.enabledValidationFeatureCount = 1;
  validation_features.pEnabledValidationFeatures    = validation_feature_enables;


  VkInstanceCreateInfo create_info = {
    .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    .pApplicationInfo = &app_info,
    .enabledExtensionCount = info->n_exts,
    .ppEnabledExtensionNames = info->exts,
    .pNext = &validation_features,
    .enabledLayerCount = info->enable_validation ?  info->n_layers : 0,
    .ppEnabledLayerNames = info->enable_validation ? info->layers : NULL,
  };

  _VK_CHECK(ctx, vkCreateInstance(&create_info, NULL, &ctx->instance)); 


  VkDebugUtilsMessengerCreateInfoEXT debug_utils_messenger_create_info = {0};

  debug_utils_messenger_create_info.sType =  VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  debug_utils_messenger_create_info.messageSeverity =
    VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
    VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
    VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
    VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  debug_utils_messenger_create_info.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
  debug_utils_messenger_create_info.pfnUserCallback = debug_utils_message_callback;
  debug_utils_messenger_create_info.pUserData = ctx; 

  VkDebugUtilsMessengerEXT debug_utils_messenger = VK_NULL_HANDLE;

  PFN_vkCreateDebugUtilsMessengerEXT func =
      (PFN_vkCreateDebugUtilsMessengerEXT)
      vkGetInstanceProcAddr(ctx->instance, "vkCreateDebugUtilsMessengerEXT");

  if(func == NULL) {
      CR_WARN(ctx->log, "Extension vkCreateDebugUtilsMessengerEXT not available.");
  } else {
      _VK_CHECK(ctx, func(ctx->instance, &debug_utils_messenger_create_info, NULL, &debug_utils_messenger));
  }

  if(ctx->log.verbose) {
    char ver_buf[32];
    _vk_ver_to_string(ver, ver_buf, sizeof(ver_buf));
    CR_TRACE(ctx->log, "Initialized Vulkan instance: (version: %s, enabledExtensionCount: %i, enabledLayerCount: %i)",
             ver_buf,
             create_info.enabledExtensionCount, create_info.enabledLayerCount);
  }

  return true; 

err:
  return false;

}

bool
_create_logical_device(struct cr_context_t* ctx) {
  if(!ctx) _PARAM_CHECK_FAIL();

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

  VkPhysicalDeviceFeatures enabled_features = {};
  enabled_features.multiDrawIndirect = ctx->_have_multi_draw_indirect;

  VkPhysicalDeviceSynchronization2Features sync2 = {
    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
    .synchronization2 = VK_TRUE
  };

  VkPhysicalDeviceSubgroupSizeControlFeaturesEXT subgroup_control = {
    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT,
    .pNext = NULL,
    .subgroupSizeControl = VK_TRUE,
    .computeFullSubgroups = VK_TRUE
  };


  VkPhysicalDeviceFeatures2 features2 = {
    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
    .pNext = &subgroup_control
  };


  vkGetPhysicalDeviceFeatures2(ctx->phys_dev, &features2);
  
  ctx->_have_subgroup_size_control = subgroup_control.subgroupSizeControl;

  if (!subgroup_control.subgroupSizeControl) {
    CR_WARN(ctx->log, "Subgroup size control is supported, will use native subgroup size: %i\n", ctx->_subgroup_size);
  } else {
    ctx->_subgroup_size = 32;
  ;}

  subgroup_control.pNext = &sync2;

  VkDeviceCreateInfo device_info = {
    .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    .pQueueCreateInfos = queues,
    .queueCreateInfoCount = queue_count, 
    .enabledExtensionCount = ctx->surf.surf ? 1 : 0, 
    .ppEnabledExtensionNames = ctx->surf.surf ? device_exts : NULL,
    .pEnabledFeatures = &enabled_features,
    .pNext = &subgroup_control,
  };


  _VK_CHECK(ctx, vkCreateDevice(ctx->phys_dev, &device_info, NULL, &ctx->logical_dev));

  CR_TRACE(ctx->log, "Initialized Vulkan logical device (graphics queue index: %i, present queue index; %i)",
           ctx->graphics_queue_family, ctx->present_queue_family);

  vkGetDeviceQueue(ctx->logical_dev, ctx->graphics_queue_family, 0, &ctx->graphics_queue);
  vkGetDeviceQueue(ctx->logical_dev, ctx->present_queue_family, 0, &ctx->present_queue);

  return true;

err:
  return false;
}


static VkFormat
_pick_depth_format(VkPhysicalDevice phys_dev) {
    VkFormat candidates[] = {
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_D32_SFLOAT_S8_UINT,
        VK_FORMAT_D24_UNORM_S8_UINT,
        VK_FORMAT_D16_UNORM,
    };

    for (uint32_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); i++) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(phys_dev, candidates[i], &props);

        if (props.optimalTilingFeatures &
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
            return candidates[i];
        }
    }

    return VK_FORMAT_UNDEFINED;
}

static VkImageAspectFlags
_depth_aspect_mask(VkFormat fmt) {
    switch (fmt) {
        case VK_FORMAT_D32_SFLOAT_S8_UINT:
        case VK_FORMAT_D24_UNORM_S8_UINT:
            return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;

        case VK_FORMAT_D32_SFLOAT:
        case VK_FORMAT_D16_UNORM:
            return VK_IMAGE_ASPECT_DEPTH_BIT;

        default:
            return VK_IMAGE_ASPECT_DEPTH_BIT;
    }
}

bool 
_create_swapchain(struct cr_context_t* ctx,  struct cr_swapchain_t* o_swapchain, uint32_t w, uint32_t h) {
  if(!ctx || !o_swapchain) _PARAM_CHECK_FAIL();
  if(o_swapchain->imgs) free(o_swapchain->imgs);
  if(o_swapchain->img_views) free(o_swapchain->img_views);
  if(o_swapchain->img_views_depth) free(o_swapchain->img_views_depth);
  if(o_swapchain->depth_images) free(o_swapchain->depth_images);
  if(o_swapchain->imgs_in_flight) free(o_swapchain->imgs_in_flight);


  memset(o_swapchain, 0, sizeof(*o_swapchain));
  

  struct _swapchain_info_t info;
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
  if(!_depth_allocs || n_imgs != ctx->swapchain.n_imgs) {
    if(_depth_allocs) free(_depth_allocs);
    _depth_allocs = calloc(n_imgs, sizeof(*_depth_allocs));
  }

  VkSwapchainCreateInfoKHR create_info = {
    .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR, 
    .surface = ctx->surf.surf,
    .minImageCount = n_imgs,
    .imageFormat = fmt.format, 
    .imageColorSpace = fmt.colorSpace,
    .imageExtent = extent,
    .imageArrayLayers = 1,
    .imageUsage   = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
    .preTransform = info.caps.currentTransform,
    .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
    .presentMode = present_mode,
    .clipped = VK_TRUE
  };

  uint32_t families[2] = {
    ctx->graphics_queue_family,
    ctx->present_queue_family
};

if (ctx->graphics_queue_family != ctx->present_queue_family) {
    create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    create_info.queueFamilyIndexCount = 2;
    create_info.pQueueFamilyIndices = families;
} else {
    create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    create_info.queueFamilyIndexCount = 0;
    create_info.pQueueFamilyIndices = NULL;
}

  _VK_CHECK(ctx, vkCreateSwapchainKHR(ctx->logical_dev, &create_info, NULL, &o_swapchain->swapchain_handle));

  _VK_CHECK(ctx, vkGetSwapchainImagesKHR(ctx->logical_dev, o_swapchain->swapchain_handle, &o_swapchain->n_imgs, NULL));
  o_swapchain->imgs = cr_util_alloc(ctx, o_swapchain->n_imgs, sizeof(VkImage));
  _VK_CHECK(ctx, vkGetSwapchainImagesKHR(ctx->logical_dev, o_swapchain->swapchain_handle, &o_swapchain->n_imgs, o_swapchain->imgs));


  o_swapchain->img_views = cr_util_alloc(ctx, o_swapchain->n_imgs, sizeof(VkImageView));
  o_swapchain->img_views_depth = cr_util_alloc(ctx, o_swapchain->n_imgs, sizeof(VkImageView));
  o_swapchain->depth_images = cr_util_alloc(ctx, o_swapchain->n_imgs, sizeof(VkImage));
  o_swapchain->imgs_in_flight = cr_util_alloc(ctx, o_swapchain->n_imgs, sizeof(VkFence));
  for (int i = 0; i < o_swapchain->n_imgs; i++)
    o_swapchain->imgs_in_flight[i] = VK_NULL_HANDLE;


  o_swapchain->present_mode = present_mode;
  o_swapchain->fmt = fmt.format;
  o_swapchain->dimensions = extent;
  o_swapchain->logical_dev = ctx->logical_dev;

  VkFormat depth_format = _pick_depth_format(ctx->phys_dev);
  o_swapchain->depth_format = depth_format;

  if (depth_format == VK_FORMAT_UNDEFINED) {
      CR_FATAL(ctx->log, "No supported depth format found.");
  }


for(uint32_t i = 0; i < o_swapchain->n_imgs; i++) {
    VkImageCreateInfo depth_image_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
    .format = depth_format,
    .extent = {
      .width  = o_swapchain->dimensions.width,
      .height = o_swapchain->dimensions.height,
      .depth  = 1
    },
    .mipLevels = 1,
    .arrayLayers = 1,
    .samples = VK_SAMPLE_COUNT_1_BIT,
    .tiling = VK_IMAGE_TILING_OPTIMAL,
    .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
  };
  VmaAllocationCreateInfo depth_alloc_info = {
    .usage = VMA_MEMORY_USAGE_GPU_ONLY,
    .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
  };
  _VK_CHECK(ctx, vmaCreateImage(
    cr_mem_get_allocator(),
    &depth_image_info,
    &depth_alloc_info,
    &o_swapchain->depth_images[i],
    &_depth_allocs[i],
    NULL
  ));

  VkImageViewCreateInfo depth_view_info = {
    .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
    .image = o_swapchain->depth_images[i],
    .viewType = VK_IMAGE_VIEW_TYPE_2D,
    .format = depth_format,
    .subresourceRange = {
        .aspectMask = _depth_aspect_mask(depth_format),
      .baseMipLevel = 0,
      .levelCount = 1,
      .baseArrayLayer = 0,
      .layerCount = 1
    }
  };


    VkImageViewCreateInfo view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = o_swapchain->imgs[i],
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = ctx->swapchain.fmt,
      .components = {
        VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY
      },
      .subresourceRange = {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1
      }
    };


    _VK_CHECK(ctx, vkCreateImageView(ctx->logical_dev, &view_info, NULL, &o_swapchain->img_views[i]));

    _VK_CHECK(ctx, vkCreateImageView(ctx->logical_dev, &depth_view_info, NULL, &o_swapchain->img_views_depth[i]));
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

    frame->render_finished_per_image = cr_util_alloc(ctx,
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
  }

  o_frameloop->frame_idx = 0;

  VkAttachmentDescription color_attachment = {
    .format = ctx->swapchain.fmt,
    .samples = VK_SAMPLE_COUNT_1_BIT, 

    .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
    .storeOp = VK_ATTACHMENT_STORE_OP_STORE,

    .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
    .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,

    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,

    .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
  };

  VkAttachmentReference color_reference = {
    .attachment = 0,
    .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL 
  };

  VkAttachmentDescription depth_attachment = {
    .format = ctx->swapchain.depth_format,
    .samples = VK_SAMPLE_COUNT_1_BIT,
    .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
    .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
    .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
    .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
  };

  VkAttachmentReference depth_reference = {
    .attachment = 1, 
    .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
  };


  VkSubpassDescription subpass = {
    .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
    .colorAttachmentCount = 1,
    .pColorAttachments = &color_reference,
    .pDepthStencilAttachment = &depth_reference
  };

  VkSubpassDependency dep = {
    .srcSubpass = VK_SUBPASS_EXTERNAL,
    .dstSubpass = 0,

    .srcStageMask =
    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
    VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,

    .dstStageMask =
    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
    VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,

    .srcAccessMask = 0,
    .dstAccessMask =
    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
  };

  VkAttachmentDescription attachments[] = {
    color_attachment,
    depth_attachment
  };

  VkRenderPassCreateInfo pass_info = {
    .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
    .attachmentCount = 2,
    .pAttachments = attachments,
    .subpassCount = 1,
    .pSubpasses = &subpass,
    .dependencyCount = 1,
    .pDependencies = &dep,
  };

  _VK_CHECK(
    ctx, 
    vkCreateRenderPass(ctx->swapchain.logical_dev, &pass_info, NULL, &o_frameloop->crnt_pass));

  o_frameloop->fbs = cr_util_alloc(ctx, ctx->swapchain.n_imgs, sizeof(*o_frameloop->fbs));


  for(uint32_t i = 0; i < ctx->swapchain.n_imgs; i++) {
    VkImageView attachments[] = {
      ctx->swapchain.img_views[i],
      ctx->swapchain.img_views_depth[i],
    };

    VkFramebufferCreateInfo fb_info = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = o_frameloop->crnt_pass,
      .attachmentCount = 2,
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


  return true;

err:

  return false;
}

bool _create_render_raster_pipelines(
  struct cr_context_t* ctx
) {
  VkPushConstantRange range = {
    .offset = 0,
    .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
    .size = sizeof(struct cr_raster_pipeline_push_constant_t) 
  };

  VkPipelineLayoutCreateInfo layout_info = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    .pPushConstantRanges = &range,
    .pushConstantRangeCount = 1
  };

  _VK_CHECK(ctx, vkCreatePipelineLayout(ctx->logical_dev, &layout_info, NULL, &ctx->pipeline_layout));

  {

    cr_raster_pipeline_add_vertex_input_attribute(
      &instanced_raster_pipeline,
      (VkVertexInputAttributeDescription){
        .location = 0,
        .binding  = 0,
        .format   = VK_FORMAT_R16G16_SFLOAT,
        .offset   = offsetof(struct cr_instance_t, px)
      });

    cr_raster_pipeline_add_vertex_input_attribute(
      &instanced_raster_pipeline,
      (VkVertexInputAttributeDescription){
        .location = 1,
        .binding  = 0,
        .format   = VK_FORMAT_R16G16_SFLOAT,
        .offset   = offsetof(struct cr_instance_t, sx)
      });

    cr_raster_pipeline_add_vertex_input_attribute(
      &instanced_raster_pipeline,
      (VkVertexInputAttributeDescription){
        .location = 2,
        .binding  = 0,
        .format   = VK_FORMAT_R8G8B8A8_UNORM,
        .offset   = offsetof(struct cr_instance_t, r)
      });


    cr_raster_pipeline_add_binding_desc(&instanced_raster_pipeline,
                                 (VkVertexInputBindingDescription){
                                 .binding   = 0,
                                 .stride    = sizeof(struct cr_instance_t), 
                                 .inputRate = VK_VERTEX_INPUT_RATE_INSTANCE
                                 });



    char* vertex_path, *fragment_path;
    cr_raster_pipeline_get_internal_shader_paths("instanced", &vertex_path, &fragment_path);

    struct cr_raster_pipeline_init_info_t info = {0};
    info.vertex_path = vertex_path; 
    info.fragment_path = fragment_path; 
    info.batch_element_size = sizeof(struct cr_instance_t);
    info.elements_per_batch = CR_MAX_BATCH; 
    info.vertices_per_instance = 6;
    info.use_device_local_buffer = true;

    cr_raster_pipeline_init(
      ctx, &instanced_raster_pipeline, &info 
    );

    cr_raster_pipeline_batching_allocate_buffer(ctx, &instanced_raster_pipeline, 
                                         CR_MAX_BATCH * CR_INITIAL_BATCH_CAP * sizeof(struct cr_instance_t),
                                         CR_GPU_BUFFER_VERTEX);
  }


  {
    cr_raster_pipeline_add_binding_desc(&vertex_raster_pipeline,
                                 (VkVertexInputBindingDescription){ 
                                 .binding = 0,
                                 .stride = sizeof(struct cr_vertex_t),
                                 .inputRate = VK_VERTEX_INPUT_RATE_VERTEX
                                 }
                                 );

    cr_raster_pipeline_add_vertex_input_attribute(
      &vertex_raster_pipeline,
      (VkVertexInputAttributeDescription) {
        .location = 1,
        .binding = 0,
        .format = VK_FORMAT_R32G32B32A32_SFLOAT,
        .offset = offsetof(struct cr_vertex_t, color)
      }
    );

    cr_raster_pipeline_add_vertex_input_attribute(
      &vertex_raster_pipeline,
      (VkVertexInputAttributeDescription) {
        .location = 0,
        .binding = 0,
        .format = VK_FORMAT_R32G32_SFLOAT,
        .offset = offsetof(struct cr_vertex_t, pos)
      });


    char* vertex_path, *fragment_path;
    cr_raster_pipeline_get_internal_shader_paths("default", &vertex_path, &fragment_path);

    struct cr_raster_pipeline_init_info_t info = {0};
    info.vertex_path = vertex_path; 
    info.fragment_path = fragment_path; 
    info.batch_element_size = sizeof(struct cr_instance_t);
    info.elements_per_batch = CR_MAX_BATCH; 

    cr_raster_pipeline_init(ctx, &vertex_raster_pipeline, &info);

    cr_raster_pipeline_batching_allocate_buffer(ctx, &vertex_raster_pipeline, 
                                         CR_MAX_BATCH * CR_INITIAL_BATCH_CAP * 3 * sizeof(struct cr_vertex_t),
                                         CR_GPU_BUFFER_VERTEX);

  }



  return true;
err:
  return false;

}

static 
bool 
_create_render_compute_pipeline(struct cr_context_t* ctx) {
  char** shader_stage_paths; 
  uint32_t n_shaders;

  if(!cr_compute_pipeline_get_internal_shader_paths(ctx, "compute", &shader_stage_paths, &n_shaders)) {
    return false;
  }

  struct cr_compute_pipeline_init_info_t info = {0};
  info.screen_w = ctx->swapchain.dimensions.width;
  info.screen_h = ctx->swapchain.dimensions.height;
  info.tile_size = 16;
  info.shader_paths = shader_stage_paths;
  info.n_shaders = n_shaders;

  info.fill_rule = CR_COMPUTE_FILL_RULE_EVEN_ODD;

  if(!cr_compute_pipeline_init(ctx, &info, &compute_pipeline)) {
    CR_ERROR(ctx->log, "Failed to set up Vulkan-Compute pipeline.");
    return false;
  }

  return true;
}

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
_create_raster_pipeline(
  struct cr_context_t* ctx, VkPipelineVertexInputStateCreateInfo vertex_input_state,
  const char* shader_subpath, VkPipeline* o_raster_pipeline) {
  VkShaderModule vert_mod, frag_mod;

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

  _VK_CHECK(ctx, vkCreateGraphicsPipelines(ctx->logical_dev, VK_NULL_HANDLE, 1, &pipeline_info, NULL, o_raster_pipeline));

  vkDestroyShaderModule(ctx->logical_dev, vert_mod, NULL);
  vkDestroyShaderModule(ctx->logical_dev, frag_mod, NULL);

  CR_TRACE(ctx->log, "Initialized %s graphics pipeline for surface %p.", shader_subpath, ctx->surf.surf);


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

      ctx->phys_dev_limits = props.limits;

      VkPhysicalDeviceFeatures features;
      vkGetPhysicalDeviceFeatures(dev, &features);

      ctx->_have_multi_draw_indirect = features.multiDrawIndirect;

      if (!features.multiDrawIndirect) {
        CR_FATAL(
          ctx->log, 
          "Physical device does not support the Multi-Draw-Indirect feature. The renderer will fallback to direct drawcall submissions.");
      }
      {
        VkPhysicalDeviceSubgroupProperties subgroup = {
          .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES
        };

        VkPhysicalDeviceProperties2 props2 = {
          .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
          .pNext = &subgroup
        };

        vkGetPhysicalDeviceProperties2(dev, &props2);

        ctx->_subgroup_size = subgroup.subgroupSize;

        bool have_subgroups =
          ((subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) &&
          (subgroup.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT)); 
        if(!have_subgroups) {
        CR_ERROR(
          ctx->log, 
          "Supgroup operations are not supported by physical device, compute pipeline will not function.");
        }
      }

      return true;
    }
  }

  return false;
}

bool 
_render_handle_resize(struct cr_context_t* ctx) {
  vkDeviceWaitIdle(ctx->logical_dev);

  for(uint32_t i = 0; i < ctx->swapchain.n_imgs; i++) {
    vmaDestroyImage(cr_mem_get_allocator(), ctx->swapchain.depth_images[i], _depth_allocs[i]);
    vkDestroyFramebuffer(
      ctx->swapchain.logical_dev,
      ctx->frameloop.fbs[i], NULL); 
    CR_TRACE(ctx->log, "Destroyed framebuffer for swapchain image %i", i); 

    vkDestroyImageView(ctx->logical_dev, ctx->swapchain.img_views[i], NULL);
    vkDestroyImageView(ctx->logical_dev, ctx->swapchain.img_views_depth[i], NULL);

    CR_TRACE(ctx->log, "Destroyed image view for swapchain image %i", i); 
  }


  free(ctx->frameloop.fbs);
  vkDestroySwapchainKHR(ctx->logical_dev, ctx->swapchain.swapchain_handle, NULL);

  if(!_create_swapchain(ctx, &ctx->swapchain, ctx->pending_resize.width, ctx->pending_resize.height)) goto err; 

  ctx->frameloop.fbs = cr_util_alloc(ctx, ctx->swapchain.n_imgs, sizeof(*ctx->frameloop.fbs));

  for(uint32_t i = 0; i < ctx->swapchain.n_imgs; i++) {
    VkImageView attachments[] = {
      ctx->swapchain.img_views[i],
      ctx->swapchain.img_views_depth[i]
    };

    VkFramebufferCreateInfo fb_info = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = ctx->frameloop.crnt_pass,
      .attachmentCount = 2,
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

  for (uint32_t i = 0; i < CR_FRAME_COUNT; i++) {
    vkResetCommandPool(
      ctx->logical_dev,
      ctx->frameloop.frames[i].cmd_pool,
      0);

    VkSemaphoreCreateInfo sem_info = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    struct cr_frame_t* frame = &ctx->frameloop.frames[i];

    if(frame->render_finished_per_image) {

      vkDestroySemaphore(ctx->logical_dev, frame->image_available, NULL);
      for(uint32_t i = 0; i < ctx->swapchain.n_imgs; i++) {
        vkDestroySemaphore(ctx->logical_dev, frame->render_finished_per_image[i], NULL);
      }
      free(frame->render_finished_per_image);
      frame->render_finished_per_image = NULL;
    }

    _VK_CHECK(ctx, vkCreateSemaphore(
      ctx->swapchain.logical_dev, 
      &sem_info, NULL, &frame->image_available)); 

    frame->render_finished_per_image = cr_util_alloc(ctx, 
                                                     ctx->swapchain.n_imgs, 
                                                     sizeof(*frame->render_finished_per_image));

    for(uint32_t i = 0; i < ctx->swapchain.n_imgs; i++) {
      _VK_CHECK(ctx, vkCreateSemaphore(
        ctx->swapchain.logical_dev, &sem_info, NULL, &frame->render_finished_per_image[i]));
    }
  }


  ctx->pending_resize.pending = false;

  ctx->frameloop.frame_idx = 0;
  ctx->_swapchain_img_idx = 0;

  cr_compute_pipeline_resize(ctx, &compute_pipeline, ctx->swapchain.dimensions.width, ctx->swapchain.dimensions.height);


  return true;

err:
  return false;
}

static inline size_t
align_up(size_t v, size_t alignment) {
  return (v + alignment - 1) & ~(alignment - 1);
}

bool
_get_swapchain_info_from_physical_device(
  struct cr_context_t* ctx,
  VkPhysicalDevice dev, 
  VkSurfaceKHR surf,
  struct _swapchain_info_t* o_info 
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
_get_swapchain_surface_format(const struct _swapchain_info_t* swapchain) {
  for(uint32_t i = 0; i < swapchain->n_fmts; i++) {
    if(swapchain->fmts[i].format == VK_FORMAT_B8G8R8_SRGB && 
      swapchain->fmts[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) return swapchain->fmts[i];
  }
  return swapchain->fmts[0];
}

VkPresentModeKHR 
_get_swapchain_present_mode(const struct _swapchain_info_t* swapchain) {
  for(uint32_t i = 0; i < swapchain->n_present_modes; i++) {
    if(swapchain->present_modes[i] == VK_PRESENT_MODE_MAILBOX_KHR) 
      return swapchain->present_modes[i]; 
  }
  return VK_PRESENT_MODE_FIFO_KHR;

}

VkExtent2D 
_get_swapchain_extent(const struct _swapchain_info_t* swapchain, uint32_t w, uint32_t h) {
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
cr_context_create(struct cr_context_t* ctx, const struct cr_context_init_info_t* info) {
  memset(ctx, 0, sizeof *ctx);
  ctx->_skip_render = false;

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

void 
cr_draw_set_clear_color(struct cr_context_t* ctx, vec4 color) {
  memcpy(ctx->_pass_info.clear_color, color, sizeof(vec4));
}

#if CR_ENABLE_GPU_STATS


#define FAIL_TOUCH_OVERFLOW         1u << 0
#define FAIL_TILE_SEGMENT_OVERFLOW  1u << 1
#define FAIL_ACTIVE_TILE_OVERFLOW   1u << 2
#define FAIL_RANK_OVERFLOW          1u << 3
#define FAIL_TILE_ID_OVERFLOW       1u << 4
#define FAIL_SCATTER_OOB            1u << 5
#define FAIL_BAD_TOUCH_TILE         1u << 6
#define FAIL_TILE_EVENT_OOB         1u << 7
#define FAIL_RANK_OOB               1u << 8
#define FAIL_FINE_TILE_SIZE_MISMATCH 1u << 9 
#define FAIL_FINE_OOB 1u << 10 

static void
cr_stats_hist_bin_range(int bin, uint32_t* lo, uint32_t* hi) {
    if (bin <= 0) {
        *lo = 0;
        *hi = 0;
        return;
    }

    if (bin == 1) {
        *lo = 1;
        *hi = 1;
        return;
    }

    if (bin >= 31) {
        *lo = 1u << 30;
        *hi = UINT32_MAX;
        return;
    }

    *lo = 1u << (bin - 1);
    *hi = (1u << bin) - 1u;
}


void
cr_read_stats_from_frame(
        struct cr_context_t* ctx,
        struct cr_gpu_stats_t* out_stats
        ) {
    memcpy(
            out_stats,
            compute_pipeline.dynamic[ctx->_swapchain_img_idx].gpu_stats_readback_buf.mem_handle,
            sizeof(*out_stats)
          );
}

static uint32_t
cr_stats_hist_count_ge(
    const uint32_t hist[CR_STATS_HIST_BINS],
    uint32_t threshold
) {
    uint32_t total = 0;

    for (int i = 0; i < CR_STATS_HIST_BINS; i++) {
        uint32_t lo = 0;
        uint32_t hi = 0;

        cr_stats_hist_bin_range(i, &lo, &hi);

        if (lo >= threshold) {
            total += hist[i];
        }
    }

    return total;
}

static int
cr_stats_percentile_bin_nonzero(
    const uint32_t hist[CR_STATS_HIST_BINS],
    uint32_t total_nonzero,
    float p
) {
    if (total_nonzero == 0) {
        return 0;
    }

    uint32_t target = (uint32_t)ceilf((float)total_nonzero * p);
    if (target == 0) {
        target = 1;
    }

    uint32_t acc = 0;

    /*
        Skip bin 0, because bin 0 is empty tiles.
        This gives percentiles over active/touched tiles.
    */
    for (int i = 1; i < CR_STATS_HIST_BINS; i++) {
        acc += hist[i];

        if (acc >= target) {
            return i;
        }
    }

    return CR_STATS_HIST_BINS - 1;
}
static int
cr_stats_percentile_bin(
    const uint32_t hist[CR_STATS_HIST_BINS],
    uint32_t total,
    float p
) {
    if (total == 0) {
        return 0;
    }

    uint32_t target = (uint32_t)ceilf((float)total * p);
    if (target == 0) {
        target = 1;
    }

    uint32_t acc = 0;

    for (int i = 0; i < CR_STATS_HIST_BINS; i++) {
        acc += hist[i];

        if (acc >= target) {
            return i;
        }
    }

    return CR_STATS_HIST_BINS - 1;
}

static void
cr_stats_print_bin_label(int bin) {
    uint32_t lo = 0;
    uint32_t hi = 0;

    cr_stats_hist_bin_range(bin, &lo, &hi);

    if (lo == hi) {
        printf("%u", lo);
    } else if (hi == UINT32_MAX) {
        printf("%u+", lo);
    } else {
        printf("%u-%u", lo, hi);
    }
}

static void
cr_stats_print_hist_compact(
    const char* name,
    const uint32_t hist[CR_STATS_HIST_BINS]
) {
    printf("  %-24s", name);

    int printed = 0;

    for (int i = 0; i < CR_STATS_HIST_BINS; i++) {
        if (hist[i] == 0) {
            continue;
        }

        if (printed == 0) {
            printf(" ");
        } else {
            printf(", ");
        }

        cr_stats_print_bin_label(i);
        printf(":%u", hist[i]);

        printed++;
    }

    if (printed == 0) {
        printf(" empty");
    }

    printf("\n");
}

static float
cr_stats_div_u32(uint32_t num, uint32_t den) {
    return den ? ((float)num / (float)den) : 0.0f;
}

static float
cr_stats_percent_u32(uint32_t num, uint32_t den) {
    return den ? (100.0f * (float)num / (float)den) : 0.0f;
}

void
cr_print_gpu_stats(const struct cr_gpu_stats_t* s) {
    if (!s) {
        return;
    }

    int tile_active_p50_bin = cr_stats_percentile_bin_nonzero(
        s->hist_tile_segments,
        s->active_tiles,
        0.50f
    );

    int tile_active_p90_bin = cr_stats_percentile_bin_nonzero(
        s->hist_tile_segments,
        s->active_tiles,
        0.90f
    );

    int tile_active_p95_bin = cr_stats_percentile_bin_nonzero(
        s->hist_tile_segments,
        s->active_tiles,
        0.95f
    );

    int tile_active_p99_bin = cr_stats_percentile_bin_nonzero(
        s->hist_tile_segments,
        s->active_tiles,
        0.99f
    );

    int fine_p50_bin = cr_stats_percentile_bin(
        s->hist_fine_segments,
        s->fine_tiles,
        0.50f
    );

    int fine_p90_bin = cr_stats_percentile_bin(
        s->hist_fine_segments,
        s->fine_tiles,
        0.90f
    );

    int fine_p95_bin = cr_stats_percentile_bin(
        s->hist_fine_segments,
        s->fine_tiles,
        0.95f
    );

    int fine_p99_bin = cr_stats_percentile_bin(
        s->hist_fine_segments,
        s->fine_tiles,
        0.99f
    );

    float avg_tile_segments = cr_stats_div_u32(
        s->total_tile_segments,
        s->n_tiles_seen
    );

    float avg_tile_segments_active = cr_stats_div_u32(
        s->total_tile_segments,
        s->active_tiles
    );

    float avg_fine_segments = cr_stats_div_u32(
        s->total_fine_segments,
        s->fine_tiles
    );

    float active_pct = cr_stats_percent_u32(
        s->active_tiles,
        s->n_tiles_seen
    );

    float fine_pct = cr_stats_percent_u32(
        s->fine_tiles,
        s->n_tiles_seen
    );

    float solid_pct = cr_stats_percent_u32(
        s->solid_tiles,
        s->n_tiles_seen
    );

    float empty_pct = cr_stats_percent_u32(
        s->empty_tiles,
        s->n_tiles_seen
    );

    float invalid_pct = cr_stats_percent_u32(
        s->invalid_edges,
        s->valid_edges + s->invalid_edges
    );

    float yedge_pct = cr_stats_percent_u32(
        s->y_edge_edges,
        s->valid_edges
    );

    float horiz_pct = cr_stats_percent_u32(
        s->horizontal_edges,
        s->valid_edges
    );

    /*
        Contention-derived stats.

        contention_tiles:
            tiles with 2+ touches.

        contended_tile_segments:
            all touches that landed on tiles with 2+ touches.

        excess_tile_segments:
            touches after the first touch per tile.

        tile_atomic_pair_pressure:
            sum n * (n - 1) / 2 over tiles.
            This is a concentration proxy for atomicAdd(tile_segments[tile]).
    */
    float contention_tile_pct = cr_stats_percent_u32(
        s->contention_tiles,
        s->active_tiles
    );

    float contended_touch_pct = cr_stats_percent_u32(
        s->contended_tile_segments,
        s->total_tile_segments
    );

    float excess_touch_pct = cr_stats_percent_u32(
        s->excess_tile_segments,
        s->total_tile_segments
    );

    float avg_extra_hits_per_active_tile = cr_stats_div_u32(
        s->excess_tile_segments,
        s->active_tiles
    );

    float avg_prior_hits_per_touch = cr_stats_div_u32(
        s->tile_atomic_pair_pressure,
        s->total_tile_segments
    );

    uint32_t hot_ge_8 = cr_stats_hist_count_ge(
        s->hist_tile_segments,
        8u
    );

    uint32_t hot_ge_16 = cr_stats_hist_count_ge(
        s->hist_tile_segments,
        16u
    );

    uint32_t hot_ge_32 = cr_stats_hist_count_ge(
        s->hist_tile_segments,
        32u
    );

    uint32_t hot_ge_64 = cr_stats_hist_count_ge(
        s->hist_tile_segments,
        64u
    );

    uint32_t hot_ge_128 = cr_stats_hist_count_ge(
        s->hist_tile_segments,
        128u
    );

    float hot_ge_8_pct = cr_stats_percent_u32(
        hot_ge_8,
        s->active_tiles
    );

    float hot_ge_16_pct = cr_stats_percent_u32(
        hot_ge_16,
        s->active_tiles
    );

    float hot_ge_32_pct = cr_stats_percent_u32(
        hot_ge_32,
        s->active_tiles
    );

    float hot_ge_64_pct = cr_stats_percent_u32(
        hot_ge_64,
        s->active_tiles
    );

    float hot_ge_128_pct = cr_stats_percent_u32(
        hot_ge_128,
        s->active_tiles
    );

    printf("\n");
    printf("[cr-gpu-stats]\n");

    printf(
        "  tiles                  seen=%u active=%u (%.1f%%) empty=%u (%.1f%%) solid=%u (%.1f%%) fine=%u (%.1f%%)\n",
        s->n_tiles_seen,
        s->active_tiles,
        active_pct,
        s->empty_tiles,
        empty_pct,
        s->solid_tiles,
        solid_pct,
        s->fine_tiles,
        fine_pct
    );

    printf(
        "  tile segments          total=%u avg/tile=%.2f avg/active=%.2f max/tile=%u active_p50~",
        s->total_tile_segments,
        avg_tile_segments,
        avg_tile_segments_active,
        s->max_tile_segments
    );
    cr_stats_print_bin_label(tile_active_p50_bin);

    printf(" p90~");
    cr_stats_print_bin_label(tile_active_p90_bin);

    printf(" p95~");
    cr_stats_print_bin_label(tile_active_p95_bin);

    printf(" p99~");
    cr_stats_print_bin_label(tile_active_p99_bin);

    printf("\n");

    printf(
        "  tile contention        tiles=%u (%.1f%% active) contended_touches=%u (%.1f%% touches) excess_touches=%u (%.1f%% touches)\n",
        s->contention_tiles,
        contention_tile_pct,
        s->contended_tile_segments,
        contended_touch_pct,
        s->excess_tile_segments,
        excess_touch_pct
    );

    printf(
        "  tile atomic pressure   pair_pressure=%u avg_prior_hits/touch=%.2f avg_extra_hits/active_tile=%.2f\n",
        s->tile_atomic_pair_pressure,
        avg_prior_hits_per_touch,
        avg_extra_hits_per_active_tile
    );

    printf(
        "  hot tile thresholds    >=8=%u (%.1f%% active) >=16=%u (%.1f%%) >=32=%u (%.1f%%) >=64=%u (%.1f%%) >=128=%u (%.1f%%)\n",
        hot_ge_8,
        hot_ge_8_pct,
        hot_ge_16,
        hot_ge_16_pct,
        hot_ge_32,
        hot_ge_32_pct,
        hot_ge_64,
        hot_ge_64_pct,
        hot_ge_128,
        hot_ge_128_pct
    );

    printf(
        "  fine segments          total=%u avg/fine=%.2f max/fine=%u p50~",
        s->total_fine_segments,
        avg_fine_segments,
        s->max_fine_segments
    );
    cr_stats_print_bin_label(fine_p50_bin);

    printf(" p90~");
    cr_stats_print_bin_label(fine_p90_bin);

    printf(" p95~");
    cr_stats_print_bin_label(fine_p95_bin);

    printf(" p99~");
    cr_stats_print_bin_label(fine_p99_bin);

    printf("\n");

    printf(
        "  fine pressure          estimated_edge_evals=%u\n",
        s->estimated_fine_edge_evals
    );

    printf(
        "  scatter edges          valid=%u invalid=%u (%.2f%%) y_edge=%u (%.1f%%) horizontal=%u (%.1f%%)\n",
        s->valid_edges,
        s->invalid_edges,
        invalid_pct,
        s->y_edge_edges,
        yedge_pct,
        s->horizontal_edges,
        horiz_pct
    );

    printf(
        "  scatter rejects        zero_len=%u tile_mismatch=%u\n",
        s->zero_length_edges,
        s->tile_mismatch_edges
    );

    printf(
        "  winding                max_abs=%u\n",
        s->max_abs_winding
    );

    cr_stats_print_hist_compact(
        "tile_seg_hist",
        s->hist_tile_segments
    );

    cr_stats_print_hist_compact(
        "fine_seg_hist",
        s->hist_fine_segments
    );

    if (s->tile_mismatch_edges != 0) {
        printf(
            "  WARNING                tile_mismatch_edges is nonzero. Walk/scatter disagree.\n"
        );
    }

    if (s->invalid_edges > s->valid_edges / 10u && s->valid_edges > 0u) {
        printf(
            "  WARNING                invalid_edges > 10%% of valid_edges. Scatter may be rejecting too much.\n"
        );
    }

    if (s->max_tile_segments > 128u || hot_ge_64 != 0u) {
        printf(
            "  NOTE                   tile segment contention looks high. tile_segments[] atomics may be a real bottleneck.\n"
        );
    }

    if (avg_prior_hits_per_touch > 8.0f) {
        printf(
            "  NOTE                   avg_prior_hits/touch is high. Touches are concentrated onto hot tile addresses.\n"
        );
    }

    if (s->max_fine_segments > 512u) {
        printf(
            "  NOTE                   max_fine_segments is high. Check for pathological hot tiles.\n"
        );
    }

        if (s->failed != 0u) {
        printf(
            "  ERROR                  gpu failed flags set: 0x%08x\n",
            s->failed
        );

        if ((s->failed & FAIL_TOUCH_OVERFLOW) != 0u) {
            printf(
                "  ERROR                  FAIL_TOUCH_OVERFLOW. bump.n_touches exceeded touch record capacity.\n"
            );
        }

          if ((s->failed & FAIL_TILE_SEGMENT_OVERFLOW) != 0u) {
            printf(
                "  ERROR                  FAIL_TILE_SEGMENT_OVERFLOW. tile segment storage exceeded max_tile_storage.\n"
            );
        }

        if ((s->failed & FAIL_ACTIVE_TILE_OVERFLOW) != 0u) {
            printf(
                "  ERROR                  FAIL_ACTIVE_TILE_OVERFLOW. active_tiles[] capacity was exceeded.\n"
            );
        }

        if ((s->failed & FAIL_RANK_OVERFLOW) != 0u) {
            printf(
                "  ERROR                  FAIL_RANK_OVERFLOW. per-tile touch rank exceeded encodable range.\n"
            );
        }

        if ((s->failed & FAIL_TILE_ID_OVERFLOW) != 0u) {
            printf(
                "  ERROR                  FAIL_TILE_ID_OVERFLOW. computed tile id exceeded encodable or valid tile range.\n"
            );
        }

        if ((s->failed & FAIL_SCATTER_OOB) != 0u) {
            printf(
                "  ERROR                  FAIL_SCATTER_OOB. scatter pass attempted out-of-bounds tile edge write.\n"
            );
        }

        if ((s->failed & FAIL_BAD_TOUCH_TILE) != 0u) {
            printf(
                "  ERROR                  FAIL_BAD_TOUCH_TILE. touch record referenced an invalid tile.\n"
            );
        }

        if ((s->failed & FAIL_TILE_EVENT_OOB) != 0u) {
            printf(
                "  ERROR                  FAIL_TILE_EVENT_OOB. tile event access exceeded tile_events[] bounds.\n"
            );
        }

        if ((s->failed & FAIL_RANK_OOB) != 0u) {
            printf(
                "  ERROR                  FAIL_RANK_OOB. scatter rank resolved outside the expected tile segment range.\n"
            );
        }

        if ((s->failed & FAIL_FINE_TILE_SIZE_MISMATCH) != 0u) {
            printf(
                "  ERROR                  FAIL_FINE_TILE_SIZE_MISMATCH. fine shader tile size assumptions do not match push constants.\n"
            );
        }

        if ((s->failed & FAIL_FINE_OOB) != 0u) {
            printf(
                "  ERROR                  FAIL_FINE_OOB. fine pass read an invalid active tile or wrote outside expected tile bounds.\n"
            );
        }

        uint32_t known_failed_bits =
            FAIL_TOUCH_OVERFLOW |
            FAIL_TILE_SEGMENT_OVERFLOW |
            FAIL_ACTIVE_TILE_OVERFLOW |
            FAIL_RANK_OVERFLOW |
            FAIL_TILE_ID_OVERFLOW |
            FAIL_SCATTER_OOB |
            FAIL_BAD_TOUCH_TILE |
            FAIL_TILE_EVENT_OOB |
            FAIL_RANK_OOB |
            FAIL_FINE_TILE_SIZE_MISMATCH |
            FAIL_FINE_OOB;

        uint32_t unknown_failed_bits = s->failed & ~known_failed_bits;

        if (unknown_failed_bits != 0u) {
            printf(
                "  ERROR                  unknown failed bits set: 0x%08x\n",
                unknown_failed_bits
            );
        }
    }


    printf("\n");
}

#endif

bool  
cr_draw_begin(struct cr_context_t* ctx) {
  if(ctx->pending_resize.pending) {
    if(!_render_handle_resize(ctx)) {
      CR_ERROR(ctx->log, "Failed to handle resize to size: %ix%i. ", ctx->pending_resize.width, ctx->pending_resize.height);
      goto err;
    }
  }
  struct cr_frame_t* frame = &ctx->frameloop.frames[ctx->frameloop.frame_idx];

  VkResult wait_res = vkWaitForFences(
          ctx->logical_dev,
          1,
          &frame->in_flight_fence,
          VK_TRUE,
          UINT64_MAX
          );

  if (wait_res != VK_SUCCESS) {
      CR_ERROR(ctx->log, "vkWaitForFences failed");
      goto err;
  }

#if CR_ENABLE_GPU_STATS 
  if(ctx->frameloop.frame_counter % 600 == 0) {
      struct cr_gpu_stats_t stats;
      cr_read_stats_from_frame(ctx, &stats);
      cr_print_gpu_stats(&stats);
  }
#endif

  /* GPU has finished all work previously submitted for this frame slot.
     Safe to read timestamp query results for this frame_idx. */
  
  if(cr_util_global_gpu_profiler_get()) { 
      cr_gpu_profiler_collect_frame(
          cr_util_global_gpu_profiler_get(),
          ctx->frameloop.frame_idx
          );
  }


  cr_mem_staging_ring_begin(frame);

  ctx->_skip_render = false;

  //if(!cr_mem_upadate_lazy_destroys(ctx)) goto err; 

  ctx->_swapchain_img_idx = 0;

  VkResult res = vkAcquireNextImageKHR(
    ctx->logical_dev,
    ctx->swapchain.swapchain_handle,
    UINT64_MAX,
    frame->image_available,
    VK_NULL_HANDLE, 
    &ctx->_swapchain_img_idx
  );

  /* Wait if this image is already being used */
  if (ctx->swapchain.imgs_in_flight[ctx->_swapchain_img_idx] != VK_NULL_HANDLE) {
    vkWaitForFences(
      ctx->logical_dev,
      1,
      &ctx->swapchain.imgs_in_flight[ctx->_swapchain_img_idx],
      VK_TRUE,
      UINT64_MAX
    );
  }

/* Mark the image as now being used by this frame */
ctx->swapchain.imgs_in_flight[ctx->_swapchain_img_idx] = frame->in_flight_fence;


  if (res == VK_ERROR_OUT_OF_DATE_KHR) {
    ctx->pending_resize.pending = true;
    ctx->_skip_render = true;
    return true;
  }

  if(res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR) goto err;

  _VK_CHECK(ctx, vkResetFences(ctx->logical_dev, 1, &frame->in_flight_fence));
  _VK_CHECK(ctx, vkResetCommandPool(ctx->logical_dev, frame->cmd_pool, 0));

  cr_raster_pipeline_batching_begin(ctx, &instanced_raster_pipeline, ctx->frameloop.frame_idx);
  cr_raster_pipeline_batching_begin(ctx, &vertex_raster_pipeline, ctx->frameloop.frame_idx);

  float current_time = glfwGetTime();

  return true;

err: 
  return false;
}

void 
cr_draw_rect(struct cr_context_t* ctx, vec2 pos, vec2 size,  uint8_t r, uint8_t g, uint8_t b, uint8_t a) {

  if (pos[0] + size[0] < 0 || pos[1] + size[1] < 0 ||
    pos[0] > ctx->swapchain.dimensions.width || pos[1] > ctx->swapchain.dimensions.height) return;
  struct cr_instance_t instance = (struct cr_instance_t){
    .px = (_Float16)pos[0], 
    .py = (_Float16)pos[1], 
    .sx = (_Float16)size[0], 
    .sy = (_Float16)size[1], 
    .r =  r, 
    .g =  g, 
    .b =  b, 
    .a =  a, 
  };

  if(!cr_raster_pipeline_batching_write_to_batch(ctx, &instanced_raster_pipeline, &instance, ctx->frameloop.frame_idx)) {

  }
}

void 
cr_draw_vertex(struct cr_context_t* ctx, vec2 pos, vec4 color) { 
  struct cr_vertex_t vertex = (struct cr_vertex_t){
    .pos = {pos[0], pos[1]},
    .color = {color[0], color[1], color[2], color[3]},
  };
  if(!cr_raster_pipeline_batching_write_to_batch(ctx, &vertex_raster_pipeline, &vertex, ctx->frameloop.frame_idx)) {

  }
}

bool 
cr_draw_begin_path(struct cr_context_t* ctx) {
  cr_compute_pipeline_begin_path(ctx, &compute_pipeline, ctx->_swapchain_img_idx);

  return true;
}
bool 
cr_draw_end_path(struct cr_context_t* ctx) {
  cr_compute_pipeline_end_path(ctx, &compute_pipeline, ctx->_swapchain_img_idx);

  return true;
}

bool 
cr_draw_segment(struct cr_context_t* ctx, struct cr_segment_t segment) {
  cr_compute_pipeline_insert_segment(ctx, &compute_pipeline, segment, ctx->_swapchain_img_idx);
  return true;
}

static void _render_set_dynamic_state(struct cr_context_t* ctx) {
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

  struct _push_constant_t pc = {
    .scale = { 2.0f / ctx->swapchain.dimensions.width,  2.0f / ctx->swapchain.dimensions.height},
    .offset = { -1.0f, -1.0f},
  };

  vkCmdPushConstants(
    frame->cmd_buf, 
    ctx->pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc); 
}


bool 
cr_draw_end(struct cr_context_t* ctx) {
  if(!ctx) _PARAM_CHECK_FAIL();

  if(ctx->_skip_render) return true;

  struct cr_frame_t* frame = &ctx->frameloop.frames[ctx->frameloop.frame_idx];

  VkCommandBufferBeginInfo begin_info = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
  };

  _VK_CHECK(ctx, vkBeginCommandBuffer(frame->cmd_buf, &begin_info));

  if(cr_util_global_gpu_profiler_get()) {
      cr_gpu_profiler_begin_frame(
              cr_util_global_gpu_profiler_get(),
              ctx->frameloop.frame_idx,
              frame->cmd_buf
              );
  }

  //cr_raster_pipeline_batching_commit(ctx, &instanced_raster_pipeline, ctx->frameloop.frame_idx);
  cr_compute_pipeline_dispatch(ctx, &compute_pipeline, ctx->frameloop.frame_idx, ctx->_swapchain_img_idx);
  

  //cr_raster_pipeline_batching_commit(ctx, &instanced_raster_pipeline, ctx->frameloop.frame_idx);

  _VK_CHECK(ctx, vkEndCommandBuffer(frame->cmd_buf));

  VkPipelineStageFlags wait_stage =
    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

VkSubmitInfo submit_info = {
    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
    .waitSemaphoreCount = 1,
    .pWaitSemaphores = &frame->image_available, // from vkAcquireNextImageKHR
    .pWaitDstStageMask = &wait_stage,
    .commandBufferCount = 1,
    .pCommandBuffers = &frame->cmd_buf,
    .signalSemaphoreCount = 1,
    .pSignalSemaphores = &frame->render_finished_per_image[ctx->_swapchain_img_idx]
};

_VK_CHECK(ctx,
    vkQueueSubmit(
        ctx->graphics_queue,
        1,
        &submit_info,
        frame->in_flight_fence
    )
);


  VkPresentInfoKHR present_info = {
    .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
    .waitSemaphoreCount = 1,
    .pWaitSemaphores = &frame->render_finished_per_image[ctx->_swapchain_img_idx],
    .swapchainCount = 1,
    .pSwapchains = &ctx->swapchain.swapchain_handle,
    .pImageIndices = &ctx->_swapchain_img_idx
  };

  vkQueuePresentKHR(ctx->present_queue, &present_info);

  
  ctx->frameloop.frame_counter++;

  cr_mem_staging_ring_end(frame);

  ctx->frameloop.frame_idx = (ctx->frameloop.frame_idx + 1) % CR_FRAME_COUNT;


  return true;

err:
  return false;
}

void 
cr_surface_resize(struct cr_context_t* ctx, uint32_t width, uint32_t height) {
  ctx->pending_resize.pending = true;
  ctx->pending_resize.width = width;
  ctx->pending_resize.height = height;

  ctx->surf.width = width;
}

