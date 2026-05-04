#version 450
#extension GL_KHR_shader_subgroup_basic      : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 256) in;

#define WG_SIZE 256
#define MAX_BINS 160
#define MAX_SUBGROUPS 8

struct SegmentMacroRange {
    uint min_xy;
    uint max_xy;
};

layout(set = 0, binding = 13, std430) readonly buffer SegmentMacroRanges {
    SegmentMacroRange segment_macro_ranges[];
};

layout(set = 0, binding = 2, std430) readonly buffer MacrotileOffsets {
    uint macrotile_offsets[];
};

layout(set = 0, binding = 3, std430) buffer MacrotileSegments {
    uint macrotile_segments[];
};

layout(set = 0, binding = 12, std430) readonly buffer MacroBlockOffsets {
    uint macro_block_offsets[];
};

layout(push_constant) uniform push_constant {
    uint screen_w, screen_h;
    uint n_tiles_x, n_tiles_y;
    uint n_macrotiles_x, n_macrotiles_y;
    uint n_seg_blocks;
    uint n_bins;

    uint tile_size, macrotile_size;

    uint n_segments;
    uint n_paths;

    uint fill_rule;
} pc;

shared uint sg_counts[MAX_SUBGROUPS][MAX_BINS];
shared uint sg_bases[MAX_SUBGROUPS][MAX_BINS];
shared uint sg_cursors[MAX_SUBGROUPS][MAX_BINS];

void main()
{
    uint tid   = gl_LocalInvocationID.x;
    uint block = gl_WorkGroupID.x;

    if (block >= pc.n_seg_blocks)
        return;

    if (pc.n_bins > MAX_BINS)
        return;

    uint n_subgroups =
        (WG_SIZE + gl_SubgroupSize - 1u) / gl_SubgroupSize;

    if (n_subgroups > MAX_SUBGROUPS)
        return;

    uint total_slots = n_subgroups * pc.n_bins;

    // Clear subgroup-local counts and cursors.
    for (uint i = tid; i < total_slots; i += WG_SIZE) {
        uint sg  = i / pc.n_bins;
        uint bin = i - sg * pc.n_bins;

        sg_counts[sg][bin]  = 0u;
        sg_cursors[sg][bin] = 0u;
    }

    barrier();

    uint seg_id = block * WG_SIZE + tid;

    // First pass inside scatter: count per subgroup/bin.
    if (seg_id < pc.n_segments) {
        SegmentMacroRange r = segment_macro_ranges[seg_id];

        uint x0 = r.min_xy & 0xffffu;
        uint y0 = r.min_xy >> 16u;
        uint x1 = r.max_xy & 0xffffu;
        uint y1 = r.max_xy >> 16u;

        uint sg = gl_SubgroupID;

        for (uint y = y0; y <= y1; ++y) {
            uint row = y * pc.n_macrotiles_x;

            for (uint x = x0; x <= x1; ++x) {
                uint bin = row + x;
                atomicAdd(sg_counts[sg][bin], 1u);
            }
        }
    }

    barrier();

    // Compute per-subgroup base for each bin.
    // sg_bases[sg][bin] = number of refs for this bin from earlier subgroups.
    for (uint bin = tid; bin < pc.n_bins; bin += WG_SIZE) {
        uint running = 0u;

        for (uint sg = 0u; sg < n_subgroups; ++sg) {
            sg_bases[sg][bin] = running;
            running += sg_counts[sg][bin];
        }
    }

    barrier();

    // Second pass: scatter each touched bin.
    if (seg_id < pc.n_segments) {
        SegmentMacroRange r = segment_macro_ranges[seg_id];

        uint x0 = r.min_xy & 0xffffu;
        uint y0 = r.min_xy >> 16u;
        uint x1 = r.max_xy & 0xffffu;
        uint y1 = r.max_xy >> 16u;

        uint sg = gl_SubgroupID;

        for (uint y = y0; y <= y1; ++y) {
            uint row = y * pc.n_macrotiles_x;

            for (uint x = x0; x <= x1; ++x) {
                uint bin = row + x;

                uint local_in_sg =
                    atomicAdd(sg_cursors[sg][bin], 1u);

                uint local_in_block =
                    sg_bases[sg][bin] + local_in_sg;

                uint block_base =
                    macro_block_offsets[bin * pc.n_seg_blocks + block];

                uint bin_base =
                    macrotile_offsets[bin];

                uint out_idx =
                    bin_base + block_base + local_in_block;

                macrotile_segments[out_idx] = seg_id;
            }
        }
    }
}
