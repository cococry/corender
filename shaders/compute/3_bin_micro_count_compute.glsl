
#version 450

#extension GL_KHR_shader_subgroup_basic  : enable
#extension GL_KHR_shader_subgroup_ballot : enable

layout(local_size_x = 64) in;

#define WG_SIZE   256
#define SG_SIZE   32
#define N_WARPS   (WG_SIZE / SG_SIZE)
#define N_MICRO   64 

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


layout(set = 0, binding = 6, std430) buffer TileNSegments {
    uint tile_n_segments[];
};

layout(set = 0, binding = 13, std430) buffer TileCountsMacro {
    uint tile_counts_micro[];
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

shared uint warp_mask[N_WARPS][N_MICRO];

uint macroIndex(vec2 p)
{
  float ms = float(pc.macrotile_size);

  ivec2 m = ivec2(floor(p / ms));
  m = clamp(m,
      ivec2(0),
      ivec2(int(pc.n_macrotiles_x - 1),
        int(pc.n_macrotiles_y - 1)));

  return uint(m.y) * pc.n_macrotiles_x + uint(m.x);
}

void main()
{
  uint tid   = gl_LocalInvocationID.x;
  uint lane  = gl_SubgroupInvocationID;
  uint warp  = tid / SG_SIZE;

  for(uint t = lane; t < N_MICRO; t += SG_SIZE)
    warp_mask[warp][t] = 0;

  barrier();

  uint macro_x = gl_WorkGroupID.x;
  uint macro_y = gl_WorkGroupID.y;

  uint macro_id = macro_y * pc.n_macrotiles_x + macro_x;

  ivec2 start = ivec2(0);
  ivec2 end   = ivec2(-1);

  uint base = macrotile_offsets[macro_id];
  uint count = macrotile_count[macro_id];


  ivec2 macro_tile_base =
    ivec2(int(macro_x) * 8,
        int(macro_y) * 8);

  ivec2 macro_tile_end = macro_tile_base + ivec2(7, 7);

  for(uint i = tid; i < count; i += WG_SIZE)
  {
    uint seg_id = macrotile_segments[base + i]; 

    Segment seg = segments[seg_id];

    vec2 seg_min = min(seg.p0, seg.p1);
    vec2 seg_max = max(seg.p0, seg.p1);

    float size = float(pc.tile_size);

    start = ivec2(floor(seg_min / size));
    end   = ivec2(floor(seg_max / size));
    ivec2 tile_max =
      ivec2(int(pc.n_tiles_x - 1),
          int(pc.n_tiles_y - 1));

    ivec2 macro_end =
      min(macro_tile_end, tile_max);

    start = clamp(start, macro_tile_base, macro_end);
    end   = clamp(end,   macro_tile_base, macro_end);

    for(int ty = start.y; ty <= end.y; ty++) { 
      for(int tx = start.x; tx <= end.x; tx++) {
        ivec2 local_tile = ivec2(tx, ty) - macro_tile_base; 
        uint local_idx = local_tile.y * 8 + local_tile.x; 
        atomicOr(warp_mask[warp][local_idx], 1u << lane);
      }
    }
  }

  barrier();
  if(tid < 64)
  {
    uint c = 0;
    for(uint w=0; w<N_WARPS; w++)
      c += bitCount(warp_mask[w][tid]);

    uint tile = tid;

    uint gx = uint(macro_tile_base.x) + (tile % 8);
    uint gy = uint(macro_tile_base.y) + (tile / 8);

    if(gx < pc.n_tiles_x && gy < pc.n_tiles_y)
    {
      uint global_tile_id = gy * pc.n_tiles_x + gx;

      tile_n_segments[global_tile_id] = c;
      tile_counts_micro[macro_id*64 + tile] = c;
    }
  }
}

