#ifndef CR_GPU_PROFILER_H
#define CR_GPU_PROFILER_H

#include <stdint.h>
#include <stdbool.h>
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef CR_GPU_PROFILER_MAX_NAME
#define CR_GPU_PROFILER_MAX_NAME 64
#endif

struct cr_gpu_profiler_scope_t {
    char name[CR_GPU_PROFILER_MAX_NAME];
    uint32_t start_query;
    uint32_t end_query;
};

struct cr_gpu_profiler_frame_t {
    VkQueryPool query_pool;

    uint32_t query_count;
    uint32_t scope_count;

    struct cr_gpu_profiler_scope_t* scopes;
    uint64_t* results;

    bool frame_recorded;
};

struct cr_gpu_profiler_stat_t {
    char name[CR_GPU_PROFILER_MAX_NAME];

    uint64_t samples;
    double total_ms;
    double min_ms;
    double max_ms;
};

struct cr_gpu_profiler_t {
    VkDevice device;

    uint32_t frames_in_flight;
    uint32_t max_scopes;
    uint32_t max_queries;

    double timestamp_period_ns;
    uint32_t timestamp_valid_bits;

    uint32_t print_interval_frames;
    uint32_t collected_since_print;

    struct cr_gpu_profiler_frame_t* frames;

    struct cr_gpu_profiler_stat_t* stats;
    uint32_t stat_count;
    uint32_t max_stats;

    bool enabled;
};

bool cr_gpu_profiler_init(
    struct cr_gpu_profiler_t* profiler,
    VkPhysicalDevice physical_device,
    VkDevice device,
    uint32_t queue_family_index,
    uint32_t frames_in_flight,
    uint32_t max_scopes_per_frame,
    uint32_t print_interval_frames
);

void cr_gpu_profiler_destroy(struct cr_gpu_profiler_t* profiler);

void cr_gpu_profiler_begin_frame(
    struct cr_gpu_profiler_t* profiler,
    uint32_t frame_idx,
    VkCommandBuffer cmd
);

uint32_t cr_gpu_profiler_begin(
    struct cr_gpu_profiler_t* profiler,
    uint32_t frame_idx,
    VkCommandBuffer cmd,
    const char* name
);

void cr_gpu_profiler_end(
    struct cr_gpu_profiler_t* profiler,
    uint32_t frame_idx,
    VkCommandBuffer cmd,
    uint32_t scope_id
);

void cr_gpu_profiler_end_frame(
    struct cr_gpu_profiler_t* profiler,
    uint32_t frame_idx
);

void cr_gpu_profiler_collect_frame(
    struct cr_gpu_profiler_t* profiler,
    uint32_t frame_idx
);

void cr_gpu_profiler_print_now(struct cr_gpu_profiler_t* profiler);

#ifdef __cplusplus
}
#endif

#endif
