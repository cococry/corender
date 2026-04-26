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

        vec2 mn = min(s.p0, s.p1);
        vec2 mx = max(s.p0, s.p1);

        float ms = float(pc.macrotile_size);

        ivec2 t0 = ivec2(floor(mn / ms));
        ivec2 t1 = ivec2(floor(mx / ms));

        t0 = clamp(t0, ivec2(0),
                   ivec2(int(pc.n_macrotiles_x - 1u),
                         int(pc.n_macrotiles_y - 1u)));

        t1 = clamp(t1, ivec2(0),
                   ivec2(int(pc.n_macrotiles_x - 1u),
                         int(pc.n_macrotiles_y - 1u)));

        uint sg = gl_SubgroupID;

        for (int y = t0.y; y <= t1.y; ++y) {
            uint row = uint(y) * pc.n_macrotiles_x;

            for (int x = t0.x; x <= t1.x; ++x) {
                uint bin = row + uint(x);

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
