#version 450

#extension GL_KHR_shader_subgroup_basic  : enable

layout(local_size_x = 64) in;

struct Segment {
    vec2 p0;
    vec2 p1;
};

layout(set = 0, binding = 0, std430) readonly buffer Segments {
    Segment segments[];
};

layout(set = 0, binding = 1, std430) buffer macrotileNSegments {
    uint macrotile_n_segments[];
};

layout(set = 0, binding = 2, std430) buffer macrotileOffsets {
    uint macrotile_offsets[];
};

layout(set = 0, binding = 3, std430) buffer macrotileSegments {
    uint macrotile_segments[];
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

#define WG_SIZE 64 
#define SG_SIZE 32
shared uint warp_tile_counts[WG_SIZE / SG_SIZE][64];

void main() {
  if(gl_WorkGroupID.x >= pc.n_macrotiles_x ||
   gl_WorkGroupID.y >= pc.n_macrotiles_y)
    return;
  uint macro_id = gl_WorkGroupID.y * pc.n_macrotiles_x + gl_WorkGroupID.x;

  uint count = macrotile_n_segments[macro_id];
  uint base  = macrotile_offsets[macro_id];

  uint lane = gl_SubgroupInvocationID;
  uint warp = gl_LocalInvocationID.x / SG_SIZE;

  if(lane < SG_SIZE) {
    for(uint i = 0; i < uint(WG_SIZE / SG_SIZE); i++) {
      warp_tile_counts[warp][lane + (SG_SIZE * i)] = 0;
    }
  }

  barrier();

  float tile_size = float(pc.tile_size);

  ivec2 macro_base = ivec2(int(gl_WorkGroupID.x) * 8,
                           int(gl_WorkGroupID.y) * 8);

  ivec2 macro_end = macro_base + ivec2(7);

  ivec2 tile_min = ivec2(0);
  ivec2 tile_max = ivec2(int(pc.n_tiles_x - 1),
                         int(pc.n_tiles_y - 1));

  for(uint i = gl_LocalInvocationID.x; i < count; i += WG_SIZE) {

    Segment seg = segments[macrotile_segments[base + i]];

    ivec2 tile_start = ivec2(floor(min(seg.p0, seg.p1) / tile_size));
    ivec2 tile_end   = ivec2(floor(max(seg.p0, seg.p1) / tile_size));

    tile_start = clamp(tile_start, tile_min, tile_max);
    tile_end   = clamp(tile_end,   tile_min, tile_max);

    tile_start = clamp(tile_start, macro_base, macro_end);
    tile_end   = clamp(tile_end,   macro_base, macro_end);

    for(int ty = tile_start.y; ty <= tile_end.y; ty++) {
      for(int tx = tile_start.x; tx <= tile_end.x; tx++) {

        uint local_tile =
          uint((ty - macro_base.y) * 8 +
               (tx - macro_base.x));

        atomicAdd(warp_tile_counts[warp][local_tile], 1);
      }
    }
  }

  barrier();

  if(gl_LocalInvocationID.x < 64) {

    uint tile = gl_LocalInvocationID.x;
    uint sum = 0;
    for(uint i = 0; i < uint(WG_SIZE / SG_SIZE); i++) {
      sum += warp_tile_counts[i][tile];
    }

    uint gx = uint(macro_base.x) + (tile % 8);
    uint gy = uint(macro_base.y) + (tile / 8);

    if(gx >= pc.n_tiles_x || gy >= pc.n_tiles_y)
      return;

    uint global_tile_id = gy * pc.n_tiles_x + gx;

    tile_n_segments[global_tile_id] = sum;
    tile_counts_micro[macro_id*64+tile] = sum;
  }
}
