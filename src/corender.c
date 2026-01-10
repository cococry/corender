#include <GLFW/glfw3.h>
#include <errno.h>
#include <linux/limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan_core.h>

#include "../vendor/vma/vk_mem_alloc.h"
#include "../include/corender/corender.h"
#include "pipeline.h"
#include "util.h"
#include "mem.h"

#define _SUBSYS_NAME "CORE"

#define _MAX_BINDING_DESC 2
#define _MAX_VERT_ATTRS 5
#define _MAX_STAGING_RING_MEM 1024 * 1024 * 256

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

struct cr_pipeline_t instanced_pipeline = {0};
struct cr_pipeline_t vertex_pipeline    = {0};

static bool     _create_log_context(struct cr_context_t* ctx, const struct cr_context_init_info_t* info);
static bool     _create_rendering_context(struct cr_context_t* ctx, const struct cr_context_init_info_t* info);
static bool     _create_instance(struct cr_context_t* ctx, const struct cr_context_init_info_t* info);
static bool     _create_logical_device(struct cr_context_t* ctx);
static bool     _create_swapchain(struct cr_context_t* ctx,  struct cr_swapchain_t* o_swapchain, uint32_t w, uint32_t h);
static bool     _create_upload_context(struct cr_context_t* ctx, struct cr_upload_context_t* o_upload, uint32_t graphics_queue_family);
static bool     _create_frameloop(struct cr_context_t* ctx, struct cr_frameloop_t* o_frameloop, uint32_t graphics_queue_family); 
static bool     _create_render_pipelines(struct cr_context_t* ctx);
static bool     _create_shader_module(struct cr_context_t* ctx, const char* filepath, VkShaderModule* o_module); 
static bool     _create_pipeline(struct cr_context_t* ctx, VkPipelineVertexInputStateCreateInfo vertex_input_state,
                                 const char* shader_subpath, VkPipeline* o_pipeline); 
static bool     _pick_physical_device(struct cr_context_t* ctx);

static bool                   _renderer_handle_resize(struct cr_context_t* ctx);

static bool                   _get_swapchain_info_from_physical_device(struct cr_context_t* ctx, VkPhysicalDevice dev, 
                                                                   VkSurfaceKHR surf, struct _swapchain_info_t* o_info);
static VkSurfaceFormatKHR     _get_swapchain_surface_format(const struct _swapchain_info_t* swapchain);
static VkPresentModeKHR       _get_swapchain_present_mode(const struct _swapchain_info_t* swapchain);
static VkExtent2D             _get_swapchain_extent(
  const struct _swapchain_info_t* swapchain, uint32_t w, uint32_t h);
static void                   _get_vertex_pipeline_input_state(struct _vertex_input_state_t* o_input_state);
static void                   _get_instanced_pipeline_input_state(struct _vertex_input_state_t* o_input_state);

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

  if(ctx->surf.surf) {
    ctx->swapchain = (struct cr_swapchain_t){0};
    if(!_create_swapchain(ctx, &ctx->swapchain, ctx->surf.width, ctx->surf.height)) {
      CR_ERROR(ctx->log, "Failed to create Vulkan swap chain (width: %i, height: %i)", 
               ctx->surf.width, ctx->surf.height);
      return false;
    }
  }

  if(!cr_mem_init(ctx)) {
    CR_ERROR(ctx->log, "Failed to create VMA context."); 
    return false;
  }

  if(!_create_frameloop(ctx, &ctx->frameloop, ctx->graphics_queue_family)) {
    CR_ERROR(ctx->log, "Failed to create Vulkan frame loop (width: %i, height: %i)", 
             ctx->surf.width, ctx->surf.height);
    return false;
  }

  if(!_create_render_pipelines(ctx)) {
    CR_ERROR(ctx->log, "Failed to create batch-rendering pipelines.");
    return false;
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


  VkInstanceCreateInfo create_info = {
    .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    .pApplicationInfo = &app_info,
    .enabledExtensionCount = info->n_exts,
    .ppEnabledExtensionNames = info->exts,
    .enabledLayerCount = info->enable_validation ?  info->n_layers : 0,
    .ppEnabledLayerNames = info->enable_validation ? info->layers : NULL
  };

  _VK_CHECK(ctx, vkCreateInstance(&create_info, NULL, &ctx->instance)); 

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

  VkPhysicalDeviceFeatures enabledFeatures = {};
  enabledFeatures.multiDrawIndirect = ctx->_have_multi_draw_indirect;

  VkDeviceCreateInfo device_info = {
    .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    .pQueueCreateInfos = queues,
    .queueCreateInfoCount = queue_count, 
    .enabledExtensionCount = ctx->surf.surf ? 1 : 0, 
    .ppEnabledExtensionNames = ctx->surf.surf ? device_exts : NULL,
    .pEnabledFeatures = &enabledFeatures,
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

bool 
_create_swapchain(struct cr_context_t* ctx,  struct cr_swapchain_t* o_swapchain, uint32_t w, uint32_t h) {
  if(!ctx || !o_swapchain) _PARAM_CHECK_FAIL();
  if(ctx->swapchain.imgs) free(ctx->swapchain.imgs);
  if(ctx->swapchain.img_views) free(ctx->swapchain.img_views);

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

  uint32_t families[2] = { ctx->graphics_queue_family, ctx->present_queue_family };

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

  o_frameloop->fbs = cr_util_alloc(ctx, ctx->swapchain.n_imgs, sizeof(*o_frameloop->fbs));


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


  return true;

err:

  return false;
}

bool _create_render_pipelines(
  struct cr_context_t* ctx
) {
  VkPushConstantRange range = {
    .offset = 0,
    .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
    .size = sizeof(struct cr_pipeline_push_constant_t) 
  };

  VkPipelineLayoutCreateInfo layout_info = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    .pPushConstantRanges = &range,
    .pushConstantRangeCount = 1
  };

  _VK_CHECK(ctx, vkCreatePipelineLayout(ctx->logical_dev, &layout_info, NULL, &ctx->pipeline_layout));

  {

    cr_pipeline_add_vertex_input_attribute(&instanced_pipeline,  (VkVertexInputAttributeDescription) {
      .location = 0,
      .binding = 0,
      .format = VK_FORMAT_R32G32_SFLOAT,
      .offset = offsetof(struct cr_instance_vertex_t, pos)
    });

    cr_pipeline_add_vertex_input_attribute(&instanced_pipeline, (VkVertexInputAttributeDescription){
      .location = 1,
      .binding = 1,
      .format = VK_FORMAT_R32G32_SFLOAT,
      .offset = offsetof(struct cr_instance_t, pos)
    });
    cr_pipeline_add_vertex_input_attribute(&instanced_pipeline, (VkVertexInputAttributeDescription){
      .location = 2,
      .binding = 1,
      .format = VK_FORMAT_R32G32_SFLOAT,
      .offset = offsetof(struct cr_instance_t, size)
    });
    cr_pipeline_add_vertex_input_attribute(&instanced_pipeline, (VkVertexInputAttributeDescription){
      .location = 3,
      .binding = 1,
      .format = VK_FORMAT_R32G32B32A32_SFLOAT,
      .offset = offsetof(struct cr_instance_t, color)
    });

    cr_pipeline_add_binding_desc(&instanced_pipeline,
                                 (VkVertexInputBindingDescription){ 
                                 .binding = 0,
                                 .stride = sizeof(struct cr_instance_vertex_t),
                                 .inputRate = VK_VERTEX_INPUT_RATE_VERTEX
                                 });
    cr_pipeline_add_binding_desc(&instanced_pipeline, (VkVertexInputBindingDescription){ 
      .binding = 1,
      .stride = sizeof(struct cr_instance_t),
      .inputRate = VK_VERTEX_INPUT_RATE_INSTANCE
    });


    char* vertex_path, *fragment_path;
    cr_pipeline_get_internal_shader_paths("instanced", &vertex_path, &fragment_path);

    struct cr_pipeline_init_info_t info = {0};
    info.vertex_path = vertex_path; 
    info.fragment_path = fragment_path; 
    info.batch_element_size = sizeof(struct cr_instance_t);
    info.elements_per_batch = CR_MAX_BATCH; 
    info.indices_per_instance = 6;

    cr_pipeline_init(
      ctx, &instanced_pipeline, &info 
    );

    struct cr_instance_vertex_t vertices[4]; 

    vertices[0] = (struct cr_instance_vertex_t){ .pos = { 0.0f, 0.0f } };
    vertices[1] = (struct cr_instance_vertex_t){ .pos = { 1.0f, 0.0f, } };
    vertices[2] = (struct cr_instance_vertex_t){ .pos = { 1.0f, 1.0f } };
    vertices[3] = (struct cr_instance_vertex_t){ .pos = { 0.0f, 1.0f} };

    uint32_t indices[6]; 

    indices[0] = 0;
    indices[1] = 1;
    indices[2] = 2;

    indices[3] = 2;
    indices[4] = 3;
    indices[5] = 0;

    cr_pipeline_add_static_buffer(ctx, &instanced_pipeline, vertices,
                                  sizeof(vertices), CR_GPU_BUFFER_VERTEX);
    
    cr_pipeline_add_static_buffer(ctx, &instanced_pipeline, indices,
                                  sizeof(indices), CR_GPU_BUFFER_INDEX);
    
    cr_pipeline_batching_allocate_buffer(ctx, &instanced_pipeline, 
                                   CR_MAX_BATCH * CR_INITIAL_BATCH_CAP * sizeof(struct cr_instance_t),
                                   CR_GPU_BUFFER_VERTEX);
  }


  {
    cr_pipeline_add_binding_desc(&vertex_pipeline,
                                 (VkVertexInputBindingDescription){ 
                                 .binding = 0,
                                 .stride = sizeof(struct cr_vertex_t),
                                 .inputRate = VK_VERTEX_INPUT_RATE_VERTEX
                                 }
    );

    cr_pipeline_add_vertex_input_attribute(
      &vertex_pipeline,
      (VkVertexInputAttributeDescription) {
        .location = 1,
        .binding = 0,
        .format = VK_FORMAT_R32G32B32A32_SFLOAT,
        .offset = offsetof(struct cr_vertex_t, color)
      }
    );

    cr_pipeline_add_vertex_input_attribute(
      &vertex_pipeline,
      (VkVertexInputAttributeDescription) {
        .location = 0,
        .binding = 0,
        .format = VK_FORMAT_R32G32_SFLOAT,
        .offset = offsetof(struct cr_vertex_t, pos)
      });
    

    char* vertex_path, *fragment_path;
    cr_pipeline_get_internal_shader_paths("default", &vertex_path, &fragment_path);

    struct cr_pipeline_init_info_t info = {0};
    info.vertex_path = vertex_path; 
    info.fragment_path = fragment_path; 
    info.batch_element_size = sizeof(struct cr_instance_t);
    info.elements_per_batch = CR_MAX_BATCH; 

    cr_pipeline_init(ctx, &vertex_pipeline, &info);

    cr_pipeline_batching_allocate_buffer(ctx, &vertex_pipeline, 
                                    CR_MAX_BATCH * CR_INITIAL_BATCH_CAP * 3 * sizeof(struct cr_vertex_t),
                                   CR_GPU_BUFFER_VERTEX);
  
  }


 
  return true;
err:
  return false;

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
_create_pipeline(
  struct cr_context_t* ctx, VkPipelineVertexInputStateCreateInfo vertex_input_state,
  const char* shader_subpath, VkPipeline* o_pipeline) {
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

  _VK_CHECK(ctx, vkCreateGraphicsPipelines(ctx->logical_dev, VK_NULL_HANDLE, 1, &pipeline_info, NULL, o_pipeline));

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
        CR_WARN(
          ctx->log, 
          "Physical device does not support the Multi-Draw-Indirect feature. The renderer will fallback to direct drawcall submissions.");
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
    vkDestroyFramebuffer(
      ctx->swapchain.logical_dev,
      ctx->frameloop.fbs[i], NULL); 
    CR_TRACE(ctx->log, "Destroyed framebuffer for swapchain image %i", i); 

    vkDestroyImageView(ctx->logical_dev, ctx->swapchain.img_views[i], NULL);
    CR_TRACE(ctx->log, "Destroyed image view for swapchain image %i", i); 
  }

  free(ctx->frameloop.fbs);

  vkDestroySwapchainKHR(ctx->logical_dev, ctx->swapchain.swapchain_handle, NULL);

  if(!_create_swapchain(ctx, &ctx->swapchain, ctx->pending_resize.width, ctx->pending_resize.height)) goto err; 

  ctx->frameloop.fbs = cr_util_alloc(ctx, ctx->swapchain.n_imgs, sizeof(*ctx->frameloop.fbs));

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

static double last_time = 0.0f, delta_time = 0.0f, last_print_time = 0.0f;
static uint32_t frame_count = 0;
bool  
cr_draw_begin(struct cr_context_t* ctx) {
  if(ctx->pending_resize.pending) {
    if(!_render_handle_resize(ctx)) {
      CR_ERROR(ctx->log, "Failed to handle resize to size: %ix%i. ", ctx->pending_resize.width, ctx->pending_resize.height);
      goto err;
    }
  }
  struct cr_frame_t* frame = &ctx->frameloop.frames[ctx->frameloop.frame_idx];
  vkWaitForFences(ctx->logical_dev, 1, &frame->in_flight_fence, VK_TRUE, UINT64_MAX);

  cr_mem_staging_ring_begin(frame);

  ctx->_skip_render = false;
 
  if(!cr_mem_upadate_lazy_destroys(ctx)) goto err; 

  frame->_n_pending_buffer_destroys = 0;

  ctx->_swapchain_img_idx = 0;
  VkResult res = vkAcquireNextImageKHR(
    ctx->logical_dev,
    ctx->swapchain.swapchain_handle,
    UINT64_MAX,
    frame->image_available,
    VK_NULL_HANDLE, 
    &ctx->_swapchain_img_idx
  );


  if (res == VK_ERROR_OUT_OF_DATE_KHR) {
    ctx->pending_resize.pending = true;
    ctx->_skip_render = true;
    return true;
  }

  if(res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR) goto err;

  _VK_CHECK(ctx, vkResetFences(ctx->logical_dev, 1, &frame->in_flight_fence));
  _VK_CHECK(ctx, vkResetCommandPool(ctx->logical_dev, frame->cmd_pool, 0));

  cr_pipeline_batching_begin(ctx, &instanced_pipeline, ctx->frameloop.frame_idx);
  cr_pipeline_batching_begin(ctx, &vertex_pipeline, ctx->frameloop.frame_idx);

  float current_time = glfwGetTime();

  frame_count++;
  delta_time = current_time - last_time;

  last_time = current_time;

  if(current_time - last_print_time >= 5.0f) {
    double elapsed = current_time - last_print_time;
    double fps = frame_count / elapsed;

    printf("FPS: %.2f\n", fps);

    frame_count = 0;
    last_print_time = current_time;
  }

  return true;

err: 
  return false;
}

void 
cr_draw_rect(struct cr_context_t* ctx, vec2 pos, vec2 size, vec4 color) {
    struct cr_instance_t instance = (struct cr_instance_t){
      .pos   = { pos[0], pos[1] },
      .size  = { size[0], size[1] },
      .color = { color[0], color[1], color[2], color[3] }
    };

  if(!cr_pipeline_batching_write_to_batch(ctx, &instanced_pipeline, &instance, ctx->frameloop.frame_idx)) {

  }
}

void 
cr_draw_vertex(struct cr_context_t* ctx, vec2 pos, vec4 color) { 
  struct cr_vertex_t vertex = (struct cr_vertex_t){
    .pos = {pos[0], pos[1]},
    .color = {color[0], color[1], color[2], color[3]},
  };
  if(!cr_pipeline_batching_write_to_batch(ctx, &vertex_pipeline, &vertex, ctx->frameloop.frame_idx)) {

  }
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
  if(ctx->_skip_render) {
    return true;
  }
  struct cr_frame_t* frame = &ctx->frameloop.frames[ctx->frameloop.frame_idx];
  
  VkCommandBufferBeginInfo begin_info = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
  };

  _VK_CHECK(ctx, vkBeginCommandBuffer(frame->cmd_buf, &begin_info));

  cr_pipeline_batching_upload(ctx, &instanced_pipeline, ctx->frameloop.frame_idx);
  cr_pipeline_batching_upload(ctx, &vertex_pipeline, ctx->frameloop.frame_idx);

  VkClearValue clear = {
    .color = {
      ctx->_pass_info.clear_color[0],
      ctx->_pass_info.clear_color[1],
      ctx->_pass_info.clear_color[2],
      ctx->_pass_info.clear_color[3]
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

  cr_pipeline_batching_commit(ctx, &instanced_pipeline, ctx->frameloop.frame_idx);
  cr_pipeline_batching_commit(ctx, &vertex_pipeline, ctx->frameloop.frame_idx);

  vkCmdEndRenderPass(frame->cmd_buf);
  _VK_CHECK(ctx, vkEndCommandBuffer(frame->cmd_buf));


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


   VkPresentInfoKHR present_info = {
    .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
    .waitSemaphoreCount = 1,
    .pWaitSemaphores = &frame->render_finished_per_image[ctx->_swapchain_img_idx],
    .swapchainCount = 1,
    .pSwapchains = &ctx->swapchain.swapchain_handle,
    .pImageIndices = &ctx->_swapchain_img_idx
  };

  vkQueuePresentKHR(ctx->present_queue, &present_info);

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

