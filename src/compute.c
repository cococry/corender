#include "compute.h"
#include "mem.h"
#include "util.h"
#include <errno.h>
#include <linux/limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include <sys/types.h>
#include <float.h>
#include <vulkan/vulkan_core.h>

#define _SUBSYS_NAME "COMPUTE"

#define DIV_UP(x, y) (((x) + (y) - 1) / (y))

enum {
    CR_BINDING_SEGMENTS      = 0,
    CR_BINDING_PATHS         = 1,
    CR_BINDING_STORAGE_IMAGE = 2,
    CR_BINDING_INDIRECT      = 3,
    CR_FIRST_USER_BINDING    = 4
};

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
static struct cr_gpu_buffer_t*  _buffer_by_name(struct cr_context_t* ctx, struct cr_compute_pipeline_t* pipeline, uint32_t swap_idx, const char* buffer_name);

static void _vk_dispatch_compute(
        struct cr_frame_t* frame, VkPipeline pipeline, 
        uint32_t swap_idx, const struct cr_compute_pipeline_push_constant_t* pc,
        uint32_t gc_x, uint32_t gc_y, uint32_t gc_z, const struct cr_gpu_buffer_t barrier_buffers[], uint32_t n_barrier_buffers);

static void _vk_compute_present(struct cr_context_t* ctx, struct cr_frame_t* frame, struct cr_compute_pipeline_t* pipeline, uint32_t swapchain_image_idx, 
        uint32_t screen_w, uint32_t screen_h);

static bool _vk_update_descriptors_for_frame(
        struct cr_context_t* ctx,
        struct cr_compute_pipeline_t* pipeline,
        uint32_t swap_idx);

static bool _compute_binding_size(
        struct cr_context_t* ctx,
        struct cr_compute_pipeline_t* pipeline,
        const char* name,
        uint32_t w,
        uint32_t h,
        size_t fallback_size,
        size_t* o_size);

static bool _ensure_segment_capacity(
        struct cr_context_t* ctx,
        struct cr_compute_pipeline_t* pipeline,
        uint32_t swap_idx,
        size_t capacity);

static bool _ensure_path_capacity(
        struct cr_context_t* ctx,
        struct cr_compute_pipeline_t* pipeline,
        uint32_t swap_idx,
        size_t capacity);

static const uint avg_touch_capacity_per_tile = 100;

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
        .requiredSubgroupSize = ctx->_subgroup_size 
    };
    VkPipelineShaderStageCreateInfo stage_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = comp_mod,
        .pName = "main",
        .pNext  = ctx->_have_subgroup_size_control ? &subgroupInfo : NULL
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
        if(pipeline->kernels[i].hash == name_hash) {
            return pipeline->kernels[i].kernel_pipeline;
        }
    }
    CR_ERROR(ctx->log, "Kernel name '%s' did not match any pipelines.", kernel_name);
    return VK_NULL_HANDLE;
}

struct cr_gpu_buffer_t* 
_buffer_by_name(struct cr_context_t* ctx, struct cr_compute_pipeline_t* pipeline, uint32_t swap_idx, const char* buffer_name) {
    uint32_t name_hash = cr_util_djb2_hash((char*)buffer_name);
    struct cr_compute_pipeline_dynamic_state_t* dyn = &pipeline->dynamic[swap_idx];

    for(uint32_t i = 0; i < dyn->n_buffers; i++) {
        if(dyn->buffers[i].hash == name_hash) return &dyn->buffers[i].buf;
    }
    CR_ERROR(ctx->log, "Buffer name '%s' did not match any bindings.", buffer_name);
    return NULL; 
}

void 
_vk_dispatch_compute(
        struct cr_frame_t* frame, VkPipeline kernel_pipeline, 
        uint32_t swap_idx, const struct cr_compute_pipeline_push_constant_t* pc,
        uint32_t gc_x, uint32_t gc_y, uint32_t gc_z, const struct cr_gpu_buffer_t barrier_buffers[], uint32_t n_barrier_buffers) {

    vkCmdBindPipeline(
            frame->cmd_buf,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            kernel_pipeline
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
                .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT, 
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

static void
_barrier_compute_write_to_indirect_read(
        VkCommandBuffer cmd,
        VkBuffer indirect_buf,
        VkDeviceSize indirect_offset
        ) {
    VkBufferMemoryBarrier2 barrier = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,

        .srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,

        .dstStageMask  = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
        .dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,

        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,

        .buffer = indirect_buf,
        .offset = indirect_offset,
        .size   = sizeof(VkDispatchIndirectCommand),
    };

    VkDependencyInfo dep = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &barrier,
    };

    vkCmdPipelineBarrier2(cmd, &dep);
}

static void
_barrier_indirect_read_to_compute_write(
        VkCommandBuffer cmd,
        VkBuffer indirect_buf,
        VkDeviceSize indirect_offset
        ) {
    VkBufferMemoryBarrier2 b = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,

        .srcStageMask  = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
        .srcAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,

        .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,

        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,

        .buffer = indirect_buf,
        .offset = indirect_offset,
        .size   = sizeof(VkDispatchIndirectCommand),
    };

    VkDependencyInfo dep = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &b,
    };

    vkCmdPipelineBarrier2(cmd, &dep);
}

static void
_vk_dispatch_compute_indirect(
        struct cr_frame_t* frame,
        VkPipeline kernel_pipeline,
        uint32_t swap_idx,
        const struct cr_compute_pipeline_push_constant_t* pc,
        const struct cr_gpu_buffer_t* indirect_buf,
        VkDeviceSize indirect_offset,
        const struct cr_gpu_buffer_t barrier_buffers[],
        uint32_t n_barrier_buffers
        ) {
    vkCmdBindPipeline(frame->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, kernel_pipeline);

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

    _barrier_compute_write_to_indirect_read(
            frame->cmd_buf,
            indirect_buf->buf,
            indirect_offset
            );


    vkCmdDispatchIndirect(
            frame->cmd_buf,
            indirect_buf->buf,
            indirect_offset
            );

    if (n_barrier_buffers) {
        VkBufferMemoryBarrier2 shader_barriers[n_barrier_buffers];

        for (uint32_t i = 0; i < n_barrier_buffers; i++) {
            shader_barriers[i] = (VkBufferMemoryBarrier2){
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                .srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,

                .buffer        = barrier_buffers[i].buf,
                .offset        = 0,
                .size          = VK_WHOLE_SIZE,
            };
        }

        VkDependencyInfo dep_shader = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .bufferMemoryBarrierCount = n_barrier_buffers,
            .pBufferMemoryBarriers = shader_barriers,
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
_compute_binding_size(
        struct cr_context_t* ctx,
        struct cr_compute_pipeline_t* pipeline,
        const char* name,
        uint32_t w,
        uint32_t h,
        size_t fallback_size,
        size_t* o_size
        ) {
    uint32_t tile_size = pipeline->info.tile_size;
    uint32_t tiles_x = (w + tile_size - 1) / tile_size;
    uint32_t tiles_y = (h + tile_size - 1) / tile_size;
    size_t n_tiles = (size_t)tiles_x * (size_t)tiles_y;

    if(!strcmp(name, "bump")) {
        *o_size = sizeof(struct cr_compute_bump_t);
        return true;
    }

    if(!strcmp(name, "tile_touch_records")) {
        *o_size = sizeof(struct cr_compute_tile_touch_record_t) * n_tiles * avg_touch_capacity_per_tile;
        return true;
    }

    if(!strcmp(name, "tile_events")) {
        *o_size = sizeof(int32_t) * n_tiles;
        return true;
    }

    if(!strcmp(name, "subgroup_tmp")) {
        *o_size = sizeof(int32_t) * n_tiles;
        return true;
    }

    if(!strcmp(name, "tile_n_segments")) {
        *o_size = sizeof(uint32_t) * n_tiles;
        return true;
    }

    if(!strcmp(name, "tile_infos")) {
        *o_size = sizeof(struct cr_compute_tile_info_t) * n_tiles;
        return true;
    }

    if(!strcmp(name, "active_tiles")) {
        *o_size = sizeof(uint32_t) * n_tiles;
        return true;
    }

    if(!strcmp(name, "tile_edges")) {
        *o_size = sizeof(struct cr_compute_tile_edge_t) * n_tiles * avg_touch_capacity_per_tile;
        return true;
    }

    *o_size = fallback_size;
    return true;
}

bool
_vk_update_descriptors_for_frame(
        struct cr_context_t* ctx,
        struct cr_compute_pipeline_t* pipeline,
        uint32_t swap_idx
        ) {
    uint32_t n_set_bindings =
        CR_FIRST_USER_BINDING + pipeline->n_buffers;

    struct cr_compute_pipeline_dynamic_state_t* dyn =
        &pipeline->dynamic[swap_idx];

    VkWriteDescriptorSet* writes =
        calloc(n_set_bindings, sizeof(*writes));

    VkDescriptorBufferInfo* buffer_infos =
        calloc(n_set_bindings, sizeof(*buffer_infos));

    VkDescriptorImageInfo* image_infos =
        calloc(n_set_bindings, sizeof(*image_infos));

    if (!writes || !buffer_infos || !image_infos) {
        free(writes);
        free(buffer_infos);
        free(image_infos);
        goto err;
    }

    buffer_infos[CR_BINDING_SEGMENTS] = (VkDescriptorBufferInfo) {
        .buffer = dyn->segment_buf.buf,
        .offset = 0,
        .range = VK_WHOLE_SIZE
    };

    writes[CR_BINDING_SEGMENTS] = (VkWriteDescriptorSet) {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = _global_sets[swap_idx],
        .dstBinding = CR_BINDING_SEGMENTS,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buffer_infos[CR_BINDING_SEGMENTS]
    };

    buffer_infos[CR_BINDING_PATHS] = (VkDescriptorBufferInfo) {
        .buffer = dyn->path_buf.buf,
        .offset = 0,
        .range = VK_WHOLE_SIZE
    };

    writes[CR_BINDING_PATHS] = (VkWriteDescriptorSet) {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = _global_sets[swap_idx],
        .dstBinding = CR_BINDING_PATHS,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buffer_infos[CR_BINDING_PATHS]
    };

    image_infos[CR_BINDING_STORAGE_IMAGE] = (VkDescriptorImageInfo) {
        .sampler = VK_NULL_HANDLE,
        .imageView = pipeline->storage_img.view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL
    };

    writes[CR_BINDING_STORAGE_IMAGE] = (VkWriteDescriptorSet) {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = _global_sets[swap_idx],
        .dstBinding = CR_BINDING_STORAGE_IMAGE,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &image_infos[CR_BINDING_STORAGE_IMAGE]
    };

    buffer_infos[CR_BINDING_INDIRECT] = (VkDescriptorBufferInfo) {
        .buffer = dyn->indirect_buf.buf,
        .offset = 0,
        .range = VK_WHOLE_SIZE
    };

    writes[CR_BINDING_INDIRECT] = (VkWriteDescriptorSet) {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = _global_sets[swap_idx],
        .dstBinding = CR_BINDING_INDIRECT,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buffer_infos[CR_BINDING_INDIRECT]
    };

    for (uint32_t j = 0; j < pipeline->n_buffers; j++) {
        uint32_t binding = CR_FIRST_USER_BINDING + j;

        buffer_infos[binding] = (VkDescriptorBufferInfo) {
            .buffer = dyn->buffers[j].buf.buf,
            .offset = 0,
            .range = VK_WHOLE_SIZE
        };

        writes[binding] = (VkWriteDescriptorSet) {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = _global_sets[swap_idx],
            .dstBinding = binding,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &buffer_infos[binding]
        };
    }

    vkUpdateDescriptorSets(
            ctx->logical_dev,
            n_set_bindings,
            writes,
            0,
            NULL
            );

    free(image_infos);
    free(buffer_infos);
    free(writes);

    return true;

err:
    return false;
}

bool
_vk_pipeline_layout_init(
        struct cr_context_t* ctx,
        struct cr_compute_pipeline_t* pipeline,
        struct cr_compute_pipeline_layout_binding_t* bindings,
        uint32_t n_bindings
        ) {
    const uint32_t n_set_bindings =
        CR_FIRST_USER_BINDING + n_bindings;

    const uint32_t n_storage_buffers =
        3u + n_bindings;

    VkResult res;

    VkDescriptorSetLayoutBinding* set_bindings =
        calloc(n_set_bindings, sizeof(*set_bindings));

    if (!set_bindings)
        goto err;

    for (uint32_t i = 0; i < n_set_bindings; i++) {
        set_bindings[i] = (VkDescriptorSetLayoutBinding) {
            .binding = i,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
        };

        if (i == CR_BINDING_STORAGE_IMAGE) {
            set_bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        } else {
            set_bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        }
    }

    VkDescriptorSetLayoutCreateInfo desc_layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = n_set_bindings,
        .pBindings = set_bindings
    };

    res = vkCreateDescriptorSetLayout(
            ctx->logical_dev,
            &desc_layout_info,
            NULL,
            &_set_layout
            );

    free(set_bindings);
    set_bindings = NULL;

    if (res != VK_SUCCESS)
        goto err;

    VkDescriptorPoolSize pool_sizes[] = {
        {
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = n_storage_buffers * ctx->swapchain.n_imgs
        },
        {
            .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = ctx->swapchain.n_imgs
        }
    };

    VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = ctx->swapchain.n_imgs,
        .poolSizeCount = sizeof(pool_sizes) / sizeof(pool_sizes[0]),
        .pPoolSizes = pool_sizes
    };

    res = vkCreateDescriptorPool(
            ctx->logical_dev,
            &pool_info,
            NULL,
            &pipeline->descriptor_pool
            );

    if (res != VK_SUCCESS)
        goto err;

    VkDescriptorSetLayout* layouts =
        calloc(ctx->swapchain.n_imgs, sizeof(*layouts));

    if (!layouts)
        goto err;

    for (uint32_t i = 0; i < ctx->swapchain.n_imgs; i++)
        layouts[i] = _set_layout;

    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = pipeline->descriptor_pool,
        .descriptorSetCount = ctx->swapchain.n_imgs,
        .pSetLayouts = layouts
    };

    res = vkAllocateDescriptorSets(
            ctx->logical_dev,
            &alloc_info,
            _global_sets
            );

    free(layouts);
    layouts = NULL;

    if (res != VK_SUCCESS)
        goto err;

    pipeline->n_buffers = n_bindings;

    pipeline->dynamic =
        cr_util_alloc(
                ctx,
                ctx->swapchain.n_imgs,
                sizeof(*pipeline->dynamic)
                );

    if (!pipeline->dynamic)
        goto err;

    for (uint32_t i = 0; i < ctx->swapchain.n_imgs; i++) {
        struct cr_compute_pipeline_dynamic_state_t* dyn =
            &pipeline->dynamic[i];

        memset(dyn, 0, sizeof(*dyn));

        dyn->segment_capacity = 50000;
        dyn->segment_data =
            cr_util_alloc(ctx, dyn->segment_capacity, sizeof(*dyn->segment_data));

        if (!dyn->segment_data)
            goto err;

        if(!cr_mem_create_gpu_buffer(
                    ctx,
                    sizeof(*dyn->segment_data) * dyn->segment_capacity,
                    CR_GPU_BUFFER_SSBO,
                    CR_GPU_BUFFER_MEM_DEVICE_LOCAL,
                    &dyn->segment_buf
                    )) {
            CR_ERROR(ctx->log, "Failed to create segment buffer.");
            return false;
        }

#if CR_ENABLE_GPU_STATS
        if (!cr_mem_create_gpu_buffer(
                    ctx,
                    sizeof(cr_gpu_stats_t),
                    CR_GPU_BUFFER_READBACK,
                    CR_GPU_BUFFER_MEM_READBACK,
                    &dyn->gpu_stats_readback_buf
                    )) {
            CR_ERROR(ctx->log, "Failed to create stats readback buffer.");
            return false;
        }
#endif

        dyn->path_capacity = 5000;
        dyn->path_data =
            cr_util_alloc(ctx, dyn->path_capacity, sizeof(*dyn->path_data));

        if (!dyn->path_data)
            goto err;

        dyn->n_paths = 0;

        if(!cr_mem_create_gpu_buffer(
                    ctx,
                    sizeof(*dyn->path_data) * dyn->path_capacity,
                    CR_GPU_BUFFER_SSBO,
                    CR_GPU_BUFFER_MEM_DEVICE_LOCAL,
                    &dyn->path_buf
                    )) {
            CR_ERROR(ctx->log, "Failed to create paths buffer.");
            return false;
        }

        dyn->indirect_data =
            cr_util_alloc(ctx, 1, sizeof(*dyn->indirect_data));

        if (!dyn->indirect_data)
            goto err;

        if(!cr_mem_create_gpu_buffer(
                    ctx,
                    sizeof(*dyn->indirect_data),
                    CR_GPU_BUFFER_INDIRECT,
                    CR_GPU_BUFFER_MEM_DEVICE_LOCAL,
                    &dyn->indirect_buf
                    )) {
            CR_ERROR(ctx->log, "Failed to create indirect dispatch buffer.");
            return false;
        }

        dyn->buffers =
            cr_util_alloc(ctx, n_bindings, sizeof(*dyn->buffers));

        if (!dyn->buffers && n_bindings > 0)
            goto err;

        dyn->n_buffers = n_bindings;

        for (uint32_t j = 0; j < n_bindings; j++) {
            struct cr_compute_pipeline_layout_buffer_t* buf =
                &dyn->buffers[j];

            CR_TRACE(
                    ctx->log,
                    "Created buffer '%s' with size %lu, binding at location: %u in layout.",
                    bindings[j].name,
                    bindings[j].buffer_size,
                    j + CR_FIRST_USER_BINDING
                    );

            buf->name = bindings[j].name;
            buf->hash = cr_util_djb2_hash((char*)bindings[j].name);
            buf->buffer_size = bindings[j].buffer_size;
        }
    }

    if (!cr_compute_pipeline_resize(
                ctx,
                pipeline,
                ctx->swapchain.dimensions.width,
                ctx->swapchain.dimensions.height
                ))
        goto err;

    VkPushConstantRange pc_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(struct cr_compute_pipeline_push_constant_t)
    };

    VkPipelineLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pc_range
    };

    res = vkCreatePipelineLayout(
            ctx->logical_dev,
            &layout_info,
            NULL,
            &_compute_pipeline_layout
            );

    if (res != VK_SUCCESS)
        goto err;

    return true;

err:
    if (set_bindings)
        free(set_bindings);

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

    if(!_global_sets || !pipeline->kernels)
        return false;


    uint32_t screen_w = info->screen_w; 
    uint32_t screen_h = info->screen_h; 
    uint32_t tile_size = info->tile_size;

    uint32_t tiles_x = (screen_w + tile_size - 1) / tile_size;
    uint32_t tiles_y = (screen_h + tile_size - 1) / tile_size;

    struct cr_compute_pipeline_layout_binding_t bindings[] = {
        (struct cr_compute_pipeline_layout_binding_t) {
            .buffer_size = sizeof(struct cr_compute_bump_t), 
            .name = "bump"  
        },
        (struct cr_compute_pipeline_layout_binding_t) {
            .buffer_size = sizeof(struct cr_compute_tile_touch_record_t) * tiles_x * tiles_y * avg_touch_capacity_per_tile, 
            .name = "tile_touch_records"  
        },
        (struct cr_compute_pipeline_layout_binding_t) {
            .buffer_size = sizeof(int32_t) * tiles_x * tiles_y, 
            .name = "tile_events"  
        },
        (struct cr_compute_pipeline_layout_binding_t) {
            .buffer_size = sizeof(int32_t) * tiles_x * tiles_y, 
            .name = "subgroup_tmp"  
        },
        (struct cr_compute_pipeline_layout_binding_t) {
            .buffer_size = sizeof(uint32_t) * tiles_x * tiles_y, 
            .name = "tile_n_segments"  
        },
        (struct cr_compute_pipeline_layout_binding_t){
            .buffer_size = sizeof(struct cr_compute_tile_info_t) * tiles_x * tiles_y, 
            .name = "tile_infos"  
        },
        (struct cr_compute_pipeline_layout_binding_t) {
            .buffer_size = sizeof(uint32_t) * tiles_x * tiles_y, 
            .name = "active_tiles"  
        },
        (struct cr_compute_pipeline_layout_binding_t) {
            .buffer_size = sizeof(struct cr_compute_tile_edge_t) * tiles_x * tiles_y * avg_touch_capacity_per_tile, 
            .name = "tile_edges"  
        },
#if CR_ENABLE_GPU_STATS
        (struct cr_compute_pipeline_layout_binding_t) {
            .buffer_size = sizeof(struct cr_gpu_stats_t), 
            .name = "stats"  
        },
#endif
    };


    if(!_vk_pipeline_layout_init(ctx, pipeline, bindings, sizeof(bindings) / sizeof(bindings[0]))) {
        CR_ERROR(ctx->log, "Failed to create Vulkan-Compute pipeline layout."); 
        return false;
    }


    for (uint32_t i = 0; i < info->n_shaders; i++) {
        const char* path = info->shader_paths[i];
        if(strstr(path, "prefix") != NULL) {
            char sg_buf[32];
            sprintf(sg_buf,"sg%i", ctx->_subgroup_size); 
            if(strstr(path, sg_buf) == NULL) continue;
        }
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
cr_compute_pipeline_resize(
        struct cr_context_t* ctx,
        struct cr_compute_pipeline_t* pipeline,
        uint32_t w,
        uint32_t h
        ) {
    vkDeviceWaitIdle(ctx->logical_dev);

    if (pipeline->storage_img.image != VK_NULL_HANDLE) {
        cr_mem_destroy_storage_image(ctx, &pipeline->storage_img);
    }

    cr_mem_create_storage_image(ctx, w, h, &pipeline->storage_img);

    for (uint32_t i = 0; i < ctx->swapchain.n_imgs; ++i) {
        struct cr_compute_pipeline_dynamic_state_t* dyn =
            &pipeline->dynamic[i];

        for (uint32_t j = 0; j < dyn->n_buffers; j++) {
            struct cr_compute_pipeline_layout_buffer_t* buf =
                &dyn->buffers[j];

            size_t buffer_size = 0;

            if (!_compute_binding_size(
                        ctx,
                        pipeline,
                        buf->name,
                        w,
                        h,
                        buf->buffer_size,
                        &buffer_size))
                goto err;

            if (buf->buf.buf != VK_NULL_HANDLE && buf->buffer_size != buffer_size) {
                cr_mem_destroy_gpu_buffer(ctx, &buf->buf);
            }

            if (buf->buf.buf == VK_NULL_HANDLE) {
                cr_mem_create_gpu_buffer(
                        ctx,
                        buffer_size,
                        CR_GPU_BUFFER_SSBO,
                        CR_GPU_BUFFER_MEM_DEVICE_LOCAL,
                        &buf->buf
                        );
            }

            buf->buffer_size = buffer_size;
        }

        if (!_vk_update_descriptors_for_frame(ctx, pipeline, i))
            goto err;
    }

    return true;

err:
    return false;
}

struct _vk_access_masks {
    VkAccessFlags2 src_access; 
    VkAccessFlags2 dst_access;
};

static 
bool 
_dispatch_prefix_sum_pass(struct cr_context_t* ctx, struct cr_compute_pipeline_t* pipeline, struct cr_frame_t* frame,
        uint32_t swapchain_image_idx,
        const struct cr_compute_pipeline_push_constant_t* pc,
        uint32_t n_elements_x, uint32_t n_rows,
        const char* pipeline_stage,
        const char* tmp_name,
        const char* output_name) {

    char step1[PATH_MAX];
    sprintf(step1, "%s_prefix2d_step1_sg%i", pipeline_stage, ctx->_subgroup_size);
    char step2[PATH_MAX];
    sprintf(step2, "%s_prefix2d_step2_sg%i", pipeline_stage, ctx->_subgroup_size);
    char step3[PATH_MAX];
    sprintf(step3, "%s_prefix2d_step3_sg%i", pipeline_stage, ctx->_subgroup_size);
    const uint32_t WG = 256;

    uint32_t blocks_per_row = DIV_UP(n_elements_x, WG);

    _vk_dispatch_compute(
            frame,
            _kernel_by_name(ctx, pipeline, step1),
            swapchain_image_idx,
            pc,
            blocks_per_row,
            n_rows,
            1,
            (struct cr_gpu_buffer_t[]){
            *_buffer_by_name(ctx, pipeline, swapchain_image_idx, output_name),
            *_buffer_by_name(ctx, pipeline, swapchain_image_idx, tmp_name),
            },
            2
            );

    if (blocks_per_row > 1) {
        if (blocks_per_row > WG) {
            CR_ERROR(ctx->log, "Prefix pass row too wide: blocks_per_row=%u > %u", blocks_per_row, WG);
            return false;
        }

        _vk_dispatch_compute(
                frame,
                _kernel_by_name(ctx, pipeline, step2),
                swapchain_image_idx,
                pc,
                1,
                n_rows,
                1,
                (struct cr_gpu_buffer_t[]){
                *_buffer_by_name(ctx, pipeline, swapchain_image_idx, tmp_name),
                },
                1
                );

        _vk_dispatch_compute(
                frame,
                _kernel_by_name(ctx, pipeline, step3),
                swapchain_image_idx,
                pc,
                blocks_per_row,
                n_rows,
                1,
                (struct cr_gpu_buffer_t[]){
                *_buffer_by_name(ctx, pipeline, swapchain_image_idx, output_name),
                *_buffer_by_name(ctx, pipeline, swapchain_image_idx, tmp_name),
                },
                2
                );
    }
    return true;
} 


#if CR_ENABLE_GPU_STATS
bool
_copy_stats_to_readback(
        struct cr_compute_pipeline_t* pipeline,
        uint32_t swapchain_image_idx,
        struct cr_context_t* ctx,
        struct cr_frame_t* frame
        ) {
    VkDeviceSize stats_size = sizeof(struct cr_gpu_stats_t);

    VkBufferMemoryBarrier2 to_transfer = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = _buffer_by_name(ctx, pipeline, swapchain_image_idx, "stats")->buf,
        .offset = 0,
        .size = stats_size
    };

    VkDependencyInfo dep0 = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &to_transfer
    };

    vkCmdPipelineBarrier2(frame->cmd_buf, &dep0);

    VkBufferCopy copy = {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = stats_size
    };

    vkCmdCopyBuffer(
            frame->cmd_buf,
            _buffer_by_name(ctx, pipeline, swapchain_image_idx, "stats")->buf,
            pipeline->dynamic[swapchain_image_idx].gpu_stats_readback_buf.buf,
            1,
            &copy
            );

    VkBufferMemoryBarrier2 to_host = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
        .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = pipeline->dynamic[swapchain_image_idx].gpu_stats_readback_buf.buf,
        .offset = 0,
        .size = stats_size
    };

    VkDependencyInfo dep1 = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &to_host
    };

    vkCmdPipelineBarrier2(frame->cmd_buf, &dep1);

    return true;
}
#endif 

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

    if(pipeline->dynamic[swapchain_image_idx].n_paths > 0) {
        cr_mem_transfer_to_device_local_gpu_buffer(ctx, frame, pipeline->dynamic[swapchain_image_idx].path_data,
                pipeline->dynamic[swapchain_image_idx].n_paths * 
                sizeof(struct cr_compute_draw_path_t),
                &pipeline->dynamic[swapchain_image_idx].path_buf);
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
        .n_tiles = tiles_x * tiles_y,
        .n_segments = n_segments,
        .screen_w = screen_w,
        .screen_h = screen_h,
        .max_tile_storage = tiles_x * tiles_y * avg_touch_capacity_per_tile,
        .n_paths =  pipeline->dynamic[swapchain_image_idx].n_paths, 
    };

    for(uint32_t i = 0; i < pipeline->dynamic[swapchain_image_idx].n_buffers; i++) {
        cr_mem_clear_gpu_buffer(ctx, frame, &pipeline->dynamic[swapchain_image_idx].buffers[i].buf);
    }

    cr_mem_clear_storage_image_color(ctx, frame, &pipeline->storage_img,
            (float[4]){0.0f, 0.0f, 0.0f, 1.0f});

    VkImageMemoryBarrier2 to_compute = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = 
            VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
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

    struct cr_gpu_profiler_t* profiler = cr_util_global_gpu_profiler_get();

    {
        uint32_t scope_id = 0;

        if (profiler != NULL) {
            scope_id = cr_gpu_profiler_begin(
                    profiler,
                    swapchain_image_idx,
                    frame->cmd_buf,
                    "walk_segments"
                    );
        }

        _vk_dispatch_compute(
                frame,
                _kernel_by_name(ctx, pipeline, "walk_segments"),
                swapchain_image_idx,
                &pc,
                DIV_UP(pc.n_segments, 256),
                1,
                1,
                (struct cr_gpu_buffer_t[]){
                *_buffer_by_name(ctx, pipeline, swapchain_image_idx, "bump"),
                *_buffer_by_name(ctx, pipeline, swapchain_image_idx, "tile_n_segments"),
                *_buffer_by_name(ctx, pipeline, swapchain_image_idx, "tile_touch_records"),
                *_buffer_by_name(ctx, pipeline, swapchain_image_idx, "tile_events"),
                },
                4
                );

        if (profiler != NULL) {
            cr_gpu_profiler_end(
                    profiler,
                    swapchain_image_idx,
                    frame->cmd_buf,
                    scope_id
                    );
        }

        if (profiler != NULL) {
            scope_id = cr_gpu_profiler_begin(
                    profiler,
                    swapchain_image_idx,
                    frame->cmd_buf,
                    "prefix_sum_tile_events"
                    );
        }

        _dispatch_prefix_sum_pass(
                ctx,
                pipeline,
                frame,
                swapchain_image_idx,
                &pc,
                pc.n_tiles_x,
                pc.n_tiles_y,
                "tile_events",
                "subgroup_tmp",
                "tile_events"
                );

        if (profiler != NULL) {
            cr_gpu_profiler_end(
                    profiler,
                    swapchain_image_idx,
                    frame->cmd_buf,
                    scope_id
                    );
        }

        if (profiler != NULL) {
            scope_id = cr_gpu_profiler_begin(
                    profiler,
                    swapchain_image_idx,
                    frame->cmd_buf,
                    "allocate_tiles"
                    );
        }

        _vk_dispatch_compute(
                frame,
                _kernel_by_name(ctx, pipeline, "allocate_tiles"),
                swapchain_image_idx,
                &pc,
                DIV_UP(pc.n_tiles_x * pc.n_tiles_y, 256),
                1,
                1,
                (struct cr_gpu_buffer_t[]){
                *_buffer_by_name(ctx, pipeline, swapchain_image_idx, "bump"),
                *_buffer_by_name(ctx, pipeline, swapchain_image_idx, "tile_n_segments"),
                *_buffer_by_name(ctx, pipeline, swapchain_image_idx, "active_tiles"),
                *_buffer_by_name(ctx, pipeline, swapchain_image_idx, "tile_infos")
#if CR_ENABLE_GPU_STATS
                ,*_buffer_by_name(ctx, pipeline, swapchain_image_idx, "stats")
#endif
                },
                5
                );

        if (profiler != NULL) {
            cr_gpu_profiler_end(
                    profiler,
                    swapchain_image_idx,
                    frame->cmd_buf,
                    scope_id
                    );
        }

        if (profiler != NULL) {
            scope_id = cr_gpu_profiler_begin(
                    profiler,
                    swapchain_image_idx,
                    frame->cmd_buf,
                    "build_scatter_indirect"
                    );
        }

        _vk_dispatch_compute(
                frame,
                _kernel_by_name(ctx, pipeline, "build_scatter_indirect"),
                swapchain_image_idx,
                &pc,
                1,
                1,
                1,
                NULL,
                0
                );

        if (profiler != NULL) {
            cr_gpu_profiler_end(
                    profiler,
                    swapchain_image_idx,
                    frame->cmd_buf,
                    scope_id
                    );
        }

        if (profiler != NULL) {
            scope_id = cr_gpu_profiler_begin(
                    profiler,
                    swapchain_image_idx,
                    frame->cmd_buf,
                    "scatter_touches"
                    );
        }

        _vk_dispatch_compute_indirect(
                frame,
                _kernel_by_name(ctx, pipeline, "scatter_touches"),
                swapchain_image_idx,
                &pc,
                &pipeline->dynamic[swapchain_image_idx].indirect_buf,
                0,
                (struct cr_gpu_buffer_t[]){
                *_buffer_by_name(ctx, pipeline, swapchain_image_idx, "bump"),
                *_buffer_by_name(ctx, pipeline, swapchain_image_idx, "tile_edges")
#if CR_ENABLE_GPU_STATS
                ,*_buffer_by_name(ctx, pipeline, swapchain_image_idx, "stats")
#endif
                },
                3
                );

        if (profiler != NULL) {
            cr_gpu_profiler_end(
                    profiler,
                    swapchain_image_idx,
                    frame->cmd_buf,
                    scope_id
                    );
        }

        /* Same indirect buffer is about to be written by build_active_tiles_indirect */
        _barrier_indirect_read_to_compute_write(
                frame->cmd_buf,
                pipeline->dynamic[swapchain_image_idx].indirect_buf.buf,
                0
                );

        if (profiler != NULL) {
            scope_id = cr_gpu_profiler_begin(
                    profiler,
                    swapchain_image_idx,
                    frame->cmd_buf,
                    "build_active_tiles_indirect"
                    );
        }

        _vk_dispatch_compute(
                frame,
                _kernel_by_name(ctx, pipeline, "build_active_tiles_indirect"),
                swapchain_image_idx,
                &pc,
                1,
                1,
                1,
                NULL,
                0
                );

        if (profiler != NULL) {
            cr_gpu_profiler_end(
                    profiler,
                    swapchain_image_idx,
                    frame->cmd_buf,
                    scope_id
                    );
        }

        if (profiler != NULL) {
            scope_id = cr_gpu_profiler_begin(
                    profiler,
                    swapchain_image_idx,
                    frame->cmd_buf,
                    "fine_eval"
                    );
        }

        _vk_dispatch_compute_indirect(
                frame,
                _kernel_by_name(ctx, pipeline, "fine_eval"),
                swapchain_image_idx,
                &pc,
                &pipeline->dynamic[swapchain_image_idx].indirect_buf,
                0,
                NULL,
                0
                );

        if (profiler != NULL) {
            cr_gpu_profiler_end(
                    profiler,
                    swapchain_image_idx,
                    frame->cmd_buf,
                    scope_id
                    );
        }
    }

#if CR_ENABLE_GPU_STATS
    _copy_stats_to_readback(pipeline, swapchain_image_idx, ctx, frame);
#endif

    _vk_compute_present(ctx, frame, pipeline, swapchain_image_idx, screen_w, screen_h);

    pipeline->dynamic[swapchain_image_idx].n_segments = 0;
    pipeline->dynamic[swapchain_image_idx].n_paths = 0;
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
        if(strcmp(namelist[j]->d_name, ".") == 0) {final_n--; free(namelist[j]); continue;}
        if(strcmp(namelist[j]->d_name, "..") == 0) {final_n--; free(namelist[j]); continue;}
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
_ensure_segment_capacity(
        struct cr_context_t* ctx,
        struct cr_compute_pipeline_t* pipeline,
        uint32_t swap_idx,
        size_t capacity
        ) {
    struct cr_compute_pipeline_dynamic_state_t* dyn =
        &pipeline->dynamic[swap_idx];

    if (capacity <= dyn->segment_capacity)
        return true;

    size_t new_capacity = dyn->segment_capacity ? dyn->segment_capacity : 1;

    while (new_capacity < capacity)
        new_capacity *= 2;

    struct cr_segment_t* segment_data =
        realloc(dyn->segment_data, sizeof(*dyn->segment_data) * new_capacity);

    if (!segment_data)
        return false;

    dyn->segment_data = segment_data;
    dyn->segment_capacity = new_capacity;

    if (dyn->segment_buf.buf != VK_NULL_HANDLE)
        cr_mem_destroy_gpu_buffer(ctx, &dyn->segment_buf);

    cr_mem_create_gpu_buffer(
            ctx,
            sizeof(*dyn->segment_data) * dyn->segment_capacity,
            CR_GPU_BUFFER_SSBO,
            CR_GPU_BUFFER_MEM_DEVICE_LOCAL,
            &dyn->segment_buf
            );

    return _vk_update_descriptors_for_frame(ctx, pipeline, swap_idx);
}

bool
_ensure_path_capacity(
        struct cr_context_t* ctx,
        struct cr_compute_pipeline_t* pipeline,
        uint32_t swap_idx,
        size_t capacity
        ) {
    struct cr_compute_pipeline_dynamic_state_t* dyn =
        &pipeline->dynamic[swap_idx];

    if (capacity <= dyn->path_capacity)
        return true;

    size_t new_capacity = dyn->path_capacity ? dyn->path_capacity : 1;

    while (new_capacity < capacity)
        new_capacity *= 2;

    struct cr_compute_draw_path_t* path_data =
        realloc(dyn->path_data, sizeof(*dyn->path_data) * new_capacity);

    if (!path_data)
        return false;

    dyn->path_data = path_data;
    dyn->path_capacity = new_capacity;

    if (dyn->path_buf.buf != VK_NULL_HANDLE)
        cr_mem_destroy_gpu_buffer(ctx, &dyn->path_buf);

    cr_mem_create_gpu_buffer(
            ctx,
            sizeof(*dyn->path_data) * dyn->path_capacity,
            CR_GPU_BUFFER_SSBO,
            CR_GPU_BUFFER_MEM_DEVICE_LOCAL,
            &dyn->path_buf
            );

    return _vk_update_descriptors_for_frame(ctx, pipeline, swap_idx);
}

bool 
cr_compute_pipeline_insert_segment(struct cr_context_t* ctx, struct cr_compute_pipeline_t* pipeline,
        struct cr_segment_t segment, uint32_t swapchain_idx) {
    if(!_ensure_segment_capacity(ctx, pipeline, swapchain_idx, pipeline->dynamic[swapchain_idx].n_segments + 1))
        return false;

    pipeline->dynamic[swapchain_idx].segment_data[pipeline->dynamic[swapchain_idx].n_segments++] = segment;
    pipeline->dynamic[swapchain_idx].n_segments_in_path++;

    return true;
}

bool cr_compute_pipeline_insert_path(struct cr_context_t* ctx, struct cr_compute_pipeline_t* pipeline,
        struct cr_compute_draw_path_t path, uint32_t swapchain_idx) {

    if(!_ensure_path_capacity(ctx, pipeline, swapchain_idx, pipeline->dynamic[swapchain_idx].n_paths + 1))
        return false;

    pipeline->dynamic[swapchain_idx].path_data[pipeline->dynamic[swapchain_idx].n_paths++] = path;

    return true;
}


bool 
cr_compute_pipeline_begin_path(struct cr_context_t* ctx, struct cr_compute_pipeline_t* pipeline, uint32_t swapchain_idx) {
    pipeline->dynamic[swapchain_idx].n_segments_in_path = 0;

    return true;
}

bool cr_compute_pipeline_end_path(struct cr_context_t* ctx, struct cr_compute_pipeline_t* pipeline, uint32_t swapchain_idx) {
    uint32_t count = pipeline->dynamic[swapchain_idx].n_segments_in_path;
    uint32_t offset = pipeline->dynamic[swapchain_idx].n_segments - pipeline->dynamic[swapchain_idx].n_segments_in_path;

    if(!_ensure_path_capacity(ctx, pipeline, swapchain_idx, pipeline->dynamic[swapchain_idx].n_paths + 1))
        return false;

    vec2 path_mn = {FLT_MAX, FLT_MAX}, path_mx = {FLT_MIN, FLT_MIN}; 
    for(size_t i = 0; i < count; i++) {
        const struct cr_segment_t* seg = &pipeline->dynamic[swapchain_idx].segment_data[i + offset];

        path_mn[0] = CR_MIN(path_mn[0], (CR_MIN(seg->p0[0], seg->p1[0])));
        path_mn[1] = CR_MIN(path_mn[1], (CR_MIN(seg->p0[1], seg->p1[1])));

        path_mx[0] = CR_MAX(path_mx[0], (CR_MAX(seg->p0[0], seg->p1[0])));
        path_mx[1] = CR_MAX(path_mx[1], (CR_MAX(seg->p0[1], seg->p1[1])));
    }


    pipeline->dynamic[swapchain_idx].path_data[pipeline->dynamic[swapchain_idx].n_paths] = (struct cr_compute_draw_path_t){
        .id = pipeline->dynamic[swapchain_idx].n_paths,
        .segment_count = count,
        .segment_offset = offset,
        .min = {path_mn[0], path_mn[1]},
        .max = {path_mx[0], path_mx[1]},
    };

    pipeline->dynamic[swapchain_idx].n_paths++;



    return true;
}
