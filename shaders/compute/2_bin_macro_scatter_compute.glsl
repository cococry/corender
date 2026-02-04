
#version 450

#extension GL_KHR_shader_subgroup_basic  : enable
#extension GL_KHR_shader_subgroup_ballot : enable


layout(local_size_x = 256) in;

#define WG_SIZE   256
#define SG_SIZE   32
#define N_WARPS   (WG_SIZE / SG_SIZE)
#define MAX_MACRO 256

struct Segment {
  vec2 p0;
  vec2 p1;
};

layout(set=0,binding=0,std430) readonly buffer Segments {
  Segment segments[];
};


layout(set = 0, binding = 1, std430) buffer macrotileNSegments {
  uint macrotile_count[];
};
layout(set = 0, binding = 2, std430) buffer macrotileOffsets {
  uint macrotile_offsets[];
};

layout(set = 0, binding = 3, std430) buffer macrotileSegments {
  uint macrotile_segments[];
};

layout(set = 0, binding = 4, std430) buffer macrotileCursor {
  uint macrotile_cursor[];
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

shared uint warp_mask[N_WARPS][MAX_MACRO];
shared uint bin_base[MAX_MACRO];

void main() {
  uint tid   = gl_LocalInvocationID.x;
  uint lane  = gl_SubgroupInvocationID;
  uint warp  = tid / SG_SIZE;

  uint n_macro = pc.n_macrotiles_x * pc.n_macrotiles_y;

  for(uint i = lane; i < MAX_MACRO; i += SG_SIZE)
    warp_mask[warp][i] = 0;

  for(uint i = tid; i < MAX_MACRO; i += WG_SIZE)
    bin_base[i] = 0;
  barrier();

  uint seg_id = gl_WorkGroupID.x * WG_SIZE + tid;

  ivec2 start = ivec2(0);
  ivec2 end   = ivec2(-1);

  if(seg_id < pc.n_segments)
  {
    Segment seg = segments[seg_id];

    vec2 seg_min = min(seg.p0, seg.p1);
    vec2 seg_max = max(seg.p0, seg.p1);

    float ms = float(pc.macrotile_size);

    start = ivec2(floor(seg_min / ms));
    end   = ivec2(floor(seg_max / ms));

    start = clamp(start, ivec2(0),
        ivec2(int(pc.n_macrotiles_x - 1),
          int(pc.n_macrotiles_y - 1)));

    end   = clamp(end, ivec2(0),
        ivec2(int(pc.n_macrotiles_x - 1),
          int(pc.n_macrotiles_y - 1)));

    for(int ty = start.y; ty <= end.y; ty++)
    {
      for(int tx = start.x; tx <= end.x; tx++)
      {
        uint bin = uint(ty) * pc.n_macrotiles_x + uint(tx);
        atomicOr(warp_mask[warp][bin], 1u << lane);
      }
    }
  }

  barrier();

  if (tid < n_macro)
{
    uint bin = tid;

    uint count = 0;
    for(uint w = 0; w < N_WARPS; w++)
        count += bitCount(warp_mask[w][bin]);

    uint base = 0;
    if(count != 0)
        base = macrotile_offsets[bin] +
               atomicAdd(macrotile_cursor[bin], count);

    bin_base[bin] = base;
}
barrier();


  barrier();

  for(int ty = start.y; ty <= end.y; ty++)
  {
    for(int tx = start.x; tx <= end.x; tx++)
    {
      uint bin = uint(ty) * pc.n_macrotiles_x + uint(tx);

      uint mask = warp_mask[warp][bin];
      uint bit  = 1u << lane;

      if((mask & bit) != 0)
      {
        uint rank = bitCount(mask & (bit - 1u));

        uint prefix = 0;
        for(uint w = 0; w < warp; w++)
          prefix += bitCount(warp_mask[w][bin]);

        uint out_idx = bin_base[bin] + prefix + rank;

        macrotile_segments[out_idx] = seg_id;
      }
    }
  }
}

