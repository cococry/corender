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

layout(set = 0, binding = 11, std430) buffer MacroBlockCounts {
    uint macro_block_counts[];
};

layout(push_constant) uniform push_constant {
  uint screen_w,  screen_h;
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

        if (int(bin_x) >= t0.x && int(bin_x) <= t1.x &&
            int(bin_y) >= t0.y && int(bin_y) <= t1.y) {
            atomicOr(warp_mask[warp], 1u << lane);
        }
    }

    barrier();

    if (tid == 0u) {
        uint count = 0u;
        for (uint w = 0u; w < N_WARPS; ++w)
            count += bitCount(warp_mask[w]);

        macro_block_counts[bin * n_seg_blocks + block] = count;
    }
}
