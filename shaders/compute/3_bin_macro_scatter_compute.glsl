#version 450
#extension GL_KHR_shader_subgroup_basic      : enable
#extension GL_KHR_shader_subgroup_ballot     : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 256) in;

#define WG_SIZE   256
#define SG_SIZE   32
#define N_WARPS   (WG_SIZE / SG_SIZE)

struct Segment {
    vec2 p0;
    vec2 p1;
};

layout(set = 0, binding = 0, std430) readonly buffer Segments {
    Segment segments[];
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

shared uint warp_mask[N_WARPS];

void main()
{
    uint tid  = gl_LocalInvocationID.x;
    uint lane = gl_SubgroupInvocationID;
    uint warp = tid / SG_SIZE;

    uint n_bins = pc.n_macrotiles_x * pc.n_macrotiles_y;
    uint block  = gl_WorkGroupID.x;
    uint bin    = gl_WorkGroupID.y;
    uint n_seg_blocks = (pc.n_segments + WG_SIZE - 1u) / WG_SIZE;

    if (bin >= n_bins || block >= n_seg_blocks)
        return;

    if (lane == 0u)
        warp_mask[warp] = 0u;

    barrier();

    uint seg_id = block * WG_SIZE + tid;

    bool isactive = false;

    if (seg_id < pc.n_segments) {
        Segment s = segments[seg_id];

        vec2 mn = min(s.p0, s.p1);
        vec2 mx = max(s.p0, s.p1);

        float ms = float(pc.macrotile_size);

        ivec2 t0 = ivec2(floor(mn / ms));
        ivec2 t1 = ivec2(floor(mx / ms));

        t0 = clamp(t0, ivec2(0),
                   ivec2(int(pc.n_macrotiles_x - 1u), int(pc.n_macrotiles_y - 1u)));
        t1 = clamp(t1, ivec2(0),
                   ivec2(int(pc.n_macrotiles_x - 1u), int(pc.n_macrotiles_y - 1u)));

        uint bin_x = bin % pc.n_macrotiles_x;
        uint bin_y = bin / pc.n_macrotiles_x;

        isactive = (int(bin_x) >= t0.x && int(bin_x) <= t1.x &&
                  int(bin_y) >= t0.y && int(bin_y) <= t1.y);

        if (isactive) {
            atomicOr(warp_mask[warp], 1u << lane);
        }
    }

    barrier();

    if (!isactive)
        return;

    uint mask = warp_mask[warp];
    uint bit  = 1u << lane;

    uint rank = bitCount(mask & (bit - 1u));

    uint prefix = 0u;
    for (uint w = 0u; w < warp; ++w)
        prefix += bitCount(warp_mask[w]);

    uint n_seg_blocks_idx = (pc.n_segments + WG_SIZE - 1u) / WG_SIZE;
    uint block_offset = macro_block_offsets[bin * n_seg_blocks_idx + block];
    uint bin_base     = macrotile_offsets[bin];

    uint out_idx = bin_base + block_offset + prefix + rank;
    macrotile_segments[out_idx] = seg_id;
}
