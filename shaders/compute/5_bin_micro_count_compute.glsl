#version 450

layout(local_size_x = 64) in;

#define WG_SIZE  64
#define N_MICRO  64

struct Segment {
  vec2 p0;
  vec2 p1;
};

layout(set=0,binding=0,std430) readonly buffer Segments {
  Segment segments[];
};

layout(set = 0, binding = 1, std430) readonly buffer macrotileNSegments {
  uint macrotile_n_segments[];
};

layout(set = 0, binding = 2, std430) readonly buffer macrotileOffsets {
  uint macrotile_offsets[];
};

layout(set = 0, binding = 3, std430) readonly buffer macrotileSegments {
  uint macrotile_segments[];
};

layout(set = 0, binding = 5, std430) buffer TileNSegments {
  uint tile_n_segments[];
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

shared uint tile_counts_local[N_MICRO];

void main()
{
  uint tid = gl_LocalInvocationID.x;

  uint macro_x = gl_WorkGroupID.x;
  uint macro_y = gl_WorkGroupID.y;

  if (macro_x >= pc.n_macrotiles_x || macro_y >= pc.n_macrotiles_y)
    return;

  uint macro_id = macro_y * pc.n_macrotiles_x + macro_x;

  ivec2 macro_tile_base = ivec2(int(macro_x) * 8, int(macro_y) * 8);
  ivec2 macro_tile_end  = macro_tile_base + ivec2(7, 7);

  if (tid < N_MICRO)
    tile_counts_local[tid] = 0u;

  barrier();

  uint base  = macrotile_offsets[macro_id];
  uint count = macrotile_n_segments[macro_id];

  for (uint i = tid; i < count; i += WG_SIZE)
  {
    uint seg_id = macrotile_segments[base + i];
    Segment seg = segments[seg_id];

    vec2 seg_min = min(seg.p0, seg.p1);
    vec2 seg_max = max(seg.p0, seg.p1);

    float tile_size_f = float(pc.tile_size);

    ivec2 start = ivec2(floor(seg_min / tile_size_f));
    ivec2 end   = ivec2(floor(seg_max / tile_size_f));

    ivec2 tile_max  = ivec2(int(pc.n_tiles_x - 1u), int(pc.n_tiles_y - 1u));
    ivec2 macro_end = min(macro_tile_end, tile_max);

    start = clamp(start, macro_tile_base, macro_end);
    end   = clamp(end,   macro_tile_base, macro_end);

    for (int ty = start.y; ty <= end.y; ++ty) {
      for (int tx = start.x; tx <= end.x; ++tx) {
        ivec2 local_tile = ivec2(tx, ty) - macro_tile_base;
        uint local_idx = uint(local_tile.y * 8 + local_tile.x);
        atomicAdd(tile_counts_local[local_idx], 1u);
      }
    }
  }

  barrier();

  if (tid < N_MICRO)
  {
    uint gx = uint(macro_tile_base.x) + (tid % 8u);
    uint gy = uint(macro_tile_base.y) + (tid / 8u);

    if (gx < pc.n_tiles_x && gy < pc.n_tiles_y)
    {
      uint global_tile_id = gy * pc.n_tiles_x + gx;
      tile_n_segments[global_tile_id] = tile_counts_local[tid];
    }
  }
}
