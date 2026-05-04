#version 450
#extension GL_KHR_shader_subgroup_basic      : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 256) in;

#define WG_SIZE 256

#define MAX_BINS 160 

#define MAX_SUBGROUPS 8 

struct Segment {
    vec2 p0;
    vec2 p1;
};

layout(set = 0, binding = 0, std430) readonly buffer Segments {
    Segment segments[];
};

layout(set = 0, binding = 11, std430) buffer MacroBlockCounts {
    uint macro_block_counts[];
};


struct SegmentMacroRange {
    uint min_xy;
    uint max_xy;
};

layout(set = 0, binding = 13, std430) buffer SegmentMacroRanges {
    SegmentMacroRange segment_macro_ranges[];
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

shared uint subgroup_counts[MAX_SUBGROUPS][MAX_BINS];

void main()
{
    uint tid   = gl_LocalInvocationID.x;
    uint block = gl_WorkGroupID.x;

    if (block >= pc.n_seg_blocks)
        return;

    uint n_subgroups =
        (WG_SIZE + gl_SubgroupSize - 1u) / gl_SubgroupSize;

    uint total_slots = n_subgroups * pc.n_bins;

    for (uint i = tid; i < total_slots; i += WG_SIZE) {
        uint sg  = i / pc.n_bins;
        uint bin = i - sg * pc.n_bins;
        subgroup_counts[sg][bin] = 0u;
    }

    barrier();

    uint seg_id = block * WG_SIZE + tid;

    if (seg_id < pc.n_segments) {
        Segment s = segments[seg_id];


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
                atomicAdd(subgroup_counts[sg][bin], 1u);
            }
        }
    }

    barrier();

    for (uint bin = tid; bin < pc.n_bins; bin += WG_SIZE) {
        uint count = 0u;

        for (uint sg = 0u; sg < n_subgroups; ++sg) {
            count += subgroup_counts[sg][bin];
        }

        macro_block_counts[bin * pc.n_seg_blocks + block] = count;
    }
}
