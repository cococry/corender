#include "profiler.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

static void
cr_profiler_copy_name(char dst[CR_GPU_PROFILER_MAX_NAME], const char* src) {
    if (!src) {
        src = "<unnamed>";
    }

    snprintf(dst, CR_GPU_PROFILER_MAX_NAME, "%s", src);
}

static struct cr_gpu_profiler_stat_t*
cr_profiler_get_stat(struct cr_gpu_profiler_t* profiler, const char* name) {
    for (uint32_t i = 0; i < profiler->stat_count; i++) {
        if (strncmp(profiler->stats[i].name, name, CR_GPU_PROFILER_MAX_NAME) == 0) {
            return &profiler->stats[i];
        }
    }

    if (profiler->stat_count >= profiler->max_stats) {
        return NULL;
    }

    struct cr_gpu_profiler_stat_t* stat = &profiler->stats[profiler->stat_count++];
    memset(stat, 0, sizeof(*stat));

    cr_profiler_copy_name(stat->name, name);
    stat->min_ms = DBL_MAX;
    stat->max_ms = 0.0;

    return stat;
}

static uint64_t
cr_profiler_mask_timestamp(uint64_t value, uint32_t valid_bits) {
    if (valid_bits == 0 || valid_bits >= 64) {
        return value;
    }

    uint64_t mask = (1ull << valid_bits) - 1ull;
    return value & mask;
}

static uint64_t
cr_profiler_timestamp_delta(uint64_t start, uint64_t end, uint32_t valid_bits) {
    if (valid_bits == 0 || valid_bits >= 64) {
        return end - start;
    }

    uint64_t mask = (1ull << valid_bits) - 1ull;

    start &= mask;
    end &= mask;

    if (end >= start) {
        return end - start;
    }

    return (end + (mask + 1ull)) - start;
}

bool
cr_gpu_profiler_init(
    struct cr_gpu_profiler_t* profiler,
    VkPhysicalDevice physical_device,
    VkDevice device,
    uint32_t queue_family_index,
    uint32_t frames_in_flight,
    uint32_t max_scopes_per_frame,
    uint32_t print_interval_frames
) {
    if (!profiler || !physical_device || !device || frames_in_flight == 0 || max_scopes_per_frame == 0) {
        return false;
    }

    memset(profiler, 0, sizeof(*profiler));

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical_device, &props);

    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &qf_count, NULL);

    if (queue_family_index >= qf_count) {
        fprintf(stderr, "[gpu-profiler] Invalid queue family index %u\n", queue_family_index);
        return false;
    }

    VkQueueFamilyProperties* qprops =
        (VkQueueFamilyProperties*)calloc(qf_count, sizeof(VkQueueFamilyProperties));

    if (!qprops) {
        return false;
    }

    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &qf_count, qprops);

    uint32_t valid_bits = qprops[queue_family_index].timestampValidBits;
    free(qprops);

    if (valid_bits == 0) {
        fprintf(stderr, "[gpu-profiler] Queue family does not support timestamp queries\n");
        return false;
    }

    profiler->device = device;
    profiler->frames_in_flight = frames_in_flight;
    profiler->max_scopes = max_scopes_per_frame;
    profiler->max_queries = max_scopes_per_frame * 2;
    profiler->timestamp_period_ns = props.limits.timestampPeriod;
    profiler->timestamp_valid_bits = valid_bits;
    profiler->print_interval_frames = print_interval_frames ? print_interval_frames : 60;
    profiler->max_stats = max_scopes_per_frame * 4;
    profiler->enabled = true;

    profiler->frames =
        (struct cr_gpu_profiler_frame_t*)calloc(frames_in_flight, sizeof(struct cr_gpu_profiler_frame_t));

    profiler->stats =
        (struct cr_gpu_profiler_stat_t*)calloc(profiler->max_stats, sizeof(struct cr_gpu_profiler_stat_t));

    if (!profiler->frames || !profiler->stats) {
        cr_gpu_profiler_destroy(profiler);
        return false;
    }

    for (uint32_t i = 0; i < frames_in_flight; i++) {
        struct cr_gpu_profiler_frame_t* frame = &profiler->frames[i];

        frame->scopes =
            (struct cr_gpu_profiler_scope_t*)calloc(max_scopes_per_frame, sizeof(struct cr_gpu_profiler_scope_t));

        frame->results =
            (uint64_t*)calloc(profiler->max_queries, sizeof(uint64_t));

        if (!frame->scopes || !frame->results) {
            cr_gpu_profiler_destroy(profiler);
            return false;
        }

        VkQueryPoolCreateInfo qpi = {
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = profiler->max_queries,
            .pipelineStatistics = 0,
        };

        VkResult res = vkCreateQueryPool(device, &qpi, NULL, &frame->query_pool);
        if (res != VK_SUCCESS) {
            fprintf(stderr, "[gpu-profiler] vkCreateQueryPool failed: %d\n", res);
            cr_gpu_profiler_destroy(profiler);
            return false;
        }
    }

    fprintf(stdout,
            "[gpu-profiler] enabled: frames=%u max_scopes=%u timestamp_period=%.3f ns valid_bits=%u\n",
            frames_in_flight,
            max_scopes_per_frame,
            profiler->timestamp_period_ns,
            profiler->timestamp_valid_bits);

    return true;
}

void
cr_gpu_profiler_destroy(struct cr_gpu_profiler_t* profiler) {
    if (!profiler) {
        return;
    }

    if (profiler->frames) {
        for (uint32_t i = 0; i < profiler->frames_in_flight; i++) {
            struct cr_gpu_profiler_frame_t* frame = &profiler->frames[i];

            if (frame->query_pool) {
                vkDestroyQueryPool(profiler->device, frame->query_pool, NULL);
            }

            free(frame->scopes);
            free(frame->results);
        }
    }

    free(profiler->frames);
    free(profiler->stats);

    memset(profiler, 0, sizeof(*profiler));
}

void
cr_gpu_profiler_begin_frame(
    struct cr_gpu_profiler_t* profiler,
    uint32_t frame_idx,
    VkCommandBuffer cmd
) {
    if (!profiler || !profiler->enabled || frame_idx >= profiler->frames_in_flight) {
        return;
    }

    struct cr_gpu_profiler_frame_t* frame = &profiler->frames[frame_idx];

    frame->query_count = 0;
    frame->scope_count = 0;
    frame->frame_recorded = true;

    vkCmdResetQueryPool(cmd, frame->query_pool, 0, profiler->max_queries);
}

uint32_t
cr_gpu_profiler_begin(
    struct cr_gpu_profiler_t* profiler,
    uint32_t frame_idx,
    VkCommandBuffer cmd,
    const char* name
) {
    if (!profiler || !profiler->enabled || frame_idx >= profiler->frames_in_flight) {
        return UINT32_MAX;
    }

    struct cr_gpu_profiler_frame_t* frame = &profiler->frames[frame_idx];

    if (frame->scope_count >= profiler->max_scopes ||
        frame->query_count + 2 > profiler->max_queries) {
        return UINT32_MAX;
    }

    uint32_t scope_id = frame->scope_count++;
    uint32_t query_id = frame->query_count++;

    struct cr_gpu_profiler_scope_t* scope = &frame->scopes[scope_id];

    cr_profiler_copy_name(scope->name, name);
    scope->start_query = query_id;
    scope->end_query = UINT32_MAX;

    vkCmdWriteTimestamp(
        cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        frame->query_pool,
        query_id
    );

    return scope_id;
}

void
cr_gpu_profiler_end(
    struct cr_gpu_profiler_t* profiler,
    uint32_t frame_idx,
    VkCommandBuffer cmd,
    uint32_t scope_id
) {
    if (!profiler || !profiler->enabled || frame_idx >= profiler->frames_in_flight) {
        return;
    }

    if (scope_id == UINT32_MAX) {
        return;
    }

    struct cr_gpu_profiler_frame_t* frame = &profiler->frames[frame_idx];

    if (scope_id >= frame->scope_count || frame->query_count >= profiler->max_queries) {
        return;
    }

    uint32_t query_id = frame->query_count++;
    frame->scopes[scope_id].end_query = query_id;

    vkCmdWriteTimestamp(
        cmd,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        frame->query_pool,
        query_id
    );
}

void
cr_gpu_profiler_end_frame(
    struct cr_gpu_profiler_t* profiler,
    uint32_t frame_idx
) {
    (void)profiler;
    (void)frame_idx;
}

void
cr_gpu_profiler_collect_frame(
    struct cr_gpu_profiler_t* profiler,
    uint32_t frame_idx
) {
    if (!profiler || !profiler->enabled || frame_idx >= profiler->frames_in_flight) {
        return;
    }

    struct cr_gpu_profiler_frame_t* frame = &profiler->frames[frame_idx];

    if (!frame->frame_recorded || frame->query_count == 0) {
        return;
    }

    VkResult res = vkGetQueryPoolResults(
        profiler->device,
        frame->query_pool,
        0,
        frame->query_count,
        sizeof(uint64_t) * frame->query_count,
        frame->results,
        sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT
    );

    if (res == VK_NOT_READY) {
        /*
            Do not wait. Waiting here would turn the profiler into the bottleneck,
            because apparently measuring time is a great way to destroy it.
        */
        return;
    }

    if (res != VK_SUCCESS) {
        fprintf(stderr, "[gpu-profiler] vkGetQueryPoolResults failed: %d\n", res);
        return;
    }

    for (uint32_t i = 0; i < frame->scope_count; i++) {
        struct cr_gpu_profiler_scope_t* scope = &frame->scopes[i];

        if (scope->start_query == UINT32_MAX || scope->end_query == UINT32_MAX) {
            continue;
        }

        uint64_t start = cr_profiler_mask_timestamp(
            frame->results[scope->start_query],
            profiler->timestamp_valid_bits
        );

        uint64_t end = cr_profiler_mask_timestamp(
            frame->results[scope->end_query],
            profiler->timestamp_valid_bits
        );

        uint64_t ticks = cr_profiler_timestamp_delta(
            start,
            end,
            profiler->timestamp_valid_bits
        );

        double ns = (double)ticks * profiler->timestamp_period_ns;
        double ms = ns / 1000000.0;

        struct cr_gpu_profiler_stat_t* stat = cr_profiler_get_stat(profiler, scope->name);
        if (!stat) {
            continue;
        }

        stat->samples++;
        stat->total_ms += ms;

        if (ms < stat->min_ms) {
            stat->min_ms = ms;
        }

        if (ms > stat->max_ms) {
            stat->max_ms = ms;
        }
    }

    frame->frame_recorded = false;

    profiler->collected_since_print++;

    if (profiler->collected_since_print >= profiler->print_interval_frames) {
        cr_gpu_profiler_print_now(profiler);
    }
}

static int
cr_profiler_stat_compare(const void* a, const void* b) {
    const struct cr_gpu_profiler_stat_t* sa = *(const struct cr_gpu_profiler_stat_t* const*)a;
    const struct cr_gpu_profiler_stat_t* sb = *(const struct cr_gpu_profiler_stat_t* const*)b;

    double aa = sa->samples ? sa->total_ms / (double)sa->samples : 0.0;
    double bb = sb->samples ? sb->total_ms / (double)sb->samples : 0.0;

    if (aa < bb) return 1;
    if (aa > bb) return -1;
    return 0;
}

void
cr_gpu_profiler_print_now(struct cr_gpu_profiler_t* profiler) {
    if (!profiler || !profiler->enabled || profiler->stat_count == 0) {
        return;
    }

    struct cr_gpu_profiler_stat_t** sorted =
        (struct cr_gpu_profiler_stat_t**)calloc(profiler->stat_count, sizeof(struct cr_gpu_profiler_stat_t*));

    if (!sorted) {
        return;
    }

    for (uint32_t i = 0; i < profiler->stat_count; i++) {
        sorted[i] = &profiler->stats[i];
    }

    qsort(sorted, profiler->stat_count, sizeof(struct cr_gpu_profiler_stat_t*), cr_profiler_stat_compare);

    fprintf(stdout, "\n[gpu-profiler] averaged over ~%u collected frame(s)\n",
            profiler->collected_since_print);

    fprintf(stdout, "  %-36s %10s %10s %10s %10s\n",
            "scope", "avg ms", "min ms", "max ms", "samples");

    fprintf(stdout, "  %-36s %10s %10s %10s %10s\n",
            "------------------------------------",
            "----------",
            "----------",
            "----------",
            "----------");

    double total_avg_ms = 0.0;

    for (uint32_t i = 0; i < profiler->stat_count; i++) {
        struct cr_gpu_profiler_stat_t* stat = sorted[i];

        if (stat->samples == 0) {
            continue;
        }

        double avg_ms = stat->total_ms / (double)stat->samples;
        total_avg_ms += avg_ms;

        fprintf(stdout, "  %-36s %10.4f %10.4f %10.4f %10llu\n",
                stat->name,
                avg_ms,
                stat->min_ms,
                stat->max_ms,
                (unsigned long long)stat->samples);
    }

    fprintf(stdout, "  %-36s %10.4f\n", "sum(avg)", total_avg_ms);
    fprintf(stdout, "\n");

    free(sorted);

    for (uint32_t i = 0; i < profiler->stat_count; i++) {
        profiler->stats[i].samples = 0;
        profiler->stats[i].total_ms = 0.0;
        profiler->stats[i].min_ms = DBL_MAX;
        profiler->stats[i].max_ms = 0.0;
    }

    profiler->stat_count = 0;
    profiler->collected_since_print = 0;
}
