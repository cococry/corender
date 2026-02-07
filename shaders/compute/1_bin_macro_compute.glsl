#version 450
#extension GL_KHR_shader_subgroup_basic      : enable
#extension GL_KHR_shader_subgroup_ballot     : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 256) in;

#define WG_SIZE   256
#define SG_SIZE   32
#define N_WARPS   (WG_SIZE/SG_SIZE)

struct Segment {
    vec2 p0;
    vec2 p1;
};

layout(set=0,binding=0,std430) readonly buffer Segments {
    Segment segments[];
};

layout(set = 0, binding = 1, std430) buffer MacrotileNSegments {
  uint macrotile_n_segments[];
};
layout(set = 0, binding = 2, std430) buffer MacrotileOffsets {
  uint macrotile_offsets[];
};

layout(set = 0, binding = 3, std430) buffer MacrotileSegments {
  uint macrotile_segments[];
};

layout(set = 0,binding = 4, std430) buffer BumpAlloc {
    uint bump_cursor;
};

layout(push_constant) uniform push_constant {
  uint screen_w, screen_h;
  uint n_tiles_x, n_tiles_y;
  uint n_macrotiles_x, n_macrotiles_y;
  uint tile_size, macrotile_size;

  uint n_segments;
  uint n_paths;
  uint fill_rule;
} pc;

shared uint warp_mask[N_WARPS][256];
shared uint bin_base[256];

void main()
{
    uint tid  = gl_LocalInvocationID.x;
    uint lane = gl_SubgroupInvocationID;
    uint warp = tid / SG_SIZE;

    uint n_bins = pc.n_macrotiles_x * pc.n_macrotiles_y;

    for(uint i = lane; i < n_bins; i += SG_SIZE)
        warp_mask[warp][i] = 0;

    if(tid < n_bins)
        bin_base[tid] = 0;

    barrier();

    uint seg_id = gl_WorkGroupID.x * WG_SIZE + tid;

    ivec2 t0 = ivec2(0);
    ivec2 t1 = ivec2(-1);

    if(seg_id < pc.n_segments)
    {
        Segment s = segments[seg_id];

        vec2 mn = min(s.p0, s.p1);
        vec2 mx = max(s.p0, s.p1);

        float ts = float(pc.macrotile_size);

        t0 = ivec2(floor(mn / ts));
        t1 = ivec2(floor(mx / ts));

        t0 = clamp(t0, ivec2(0),
                   ivec2(int(pc.n_macrotiles_x-1), int(pc.n_macrotiles_y-1)));

        t1 = clamp(t1, ivec2(0),
                   ivec2(int(pc.n_macrotiles_x-1), int(pc.n_macrotiles_y-1)));

        for(int ty=t0.y; ty<=t1.y; ty++)
        for(int tx=t0.x; tx<=t1.x; tx++)
        {
            uint bin = uint(ty)*pc.n_macrotiles_x + uint(tx);
            atomicOr(warp_mask[warp][bin], 1u<<lane);
        }
    }

    barrier();

    for(uint bin = tid; bin < n_bins; bin += WG_SIZE)
    {
      if(bin < pc.n_macrotiles_x * pc.n_macrotiles_y) {
        uint count = 0;
        for(uint w=0; w<N_WARPS; w++)
            count += bitCount(warp_mask[w][bin]);

        uint base = 0;
        if(count != 0)
            base = atomicAdd(bump_cursor, count);

        macrotile_n_segments[bin] = count;
        macrotile_offsets[bin] = base;

        bin_base[bin] = base;
      }
    }

    barrier();

    if(seg_id >= pc.n_segments)
        return;

    for(int ty=t0.y; ty<=t1.y; ty++)
    for(int tx=t0.x; tx<=t1.x; tx++)
    {
        uint bin = uint(ty)*pc.n_macrotiles_x + uint(tx);

        uint mask = warp_mask[warp][bin];
        uint bit  = 1u<<lane;

        if((mask & bit) != 0)
        {
            uint rank = bitCount(mask & (bit-1u));

            uint prefix = 0;
            for(uint w=0; w<warp; w++)
                prefix += bitCount(warp_mask[w][bin]);

            uint out_idx = bin_base[bin] + prefix + rank;

            macrotile_segments[out_idx] = seg_id;
        }
    }
}

