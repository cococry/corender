#version 450

layout(local_size_x = 32) in;

#extension GL_KHR_shader_subgroup_basic      : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

struct Segment {
  vec2 p0;
  vec2 p1;
};

layout(set = 0, binding = 0, std430) readonly buffer Segments {
  Segment segments[];
};

layout(set = 0, binding = 5, std430) buffer TileNSegments {
    uint tile_n_segments[];
};

layout(set = 0, binding = 7, std430) readonly buffer TileOffsets {
  uint tile_offsets[];
};

layout(set = 0, binding = 9, std430) buffer TileSegments {
  uint tile_segments[];
};

layout(set = 0, binding = 10, std430) writeonly buffer PrefixParity {
  uint prefix_parity[];
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

void main()
{
  uvec2 tile = gl_WorkGroupID.xy;
  uint lane  = gl_LocalInvocationID.x;

  uint tile_id = tile.y * pc.n_tiles_x + tile.x;

  ivec2 macro_xy = ivec2(tile / 8);
  uint macro_id =
    uint(macro_xy.y) * pc.n_macrotiles_x +
    uint(macro_xy.x);

  uint base  = tile_offsets[tile_id];
  uint count = tile_n_segments[tile_id];

  float tile_x0 = float(tile.x) * float(pc.tile_size);
  float tile_x1 = tile_x0 + float(pc.tile_size);

  float tile_y0 = float(tile.y) * float(pc.tile_size);

  uint mask = 0u;

  uint sgSize = gl_SubgroupSize;

  for (uint i = lane; i < count; i += sgSize)
  {
    uint seg_id = tile_segments[base + i];
    Segment s   = segments[seg_id];

    float x0 = s.p0.x;
    float y0 = s.p0.y;
    float x1 = s.p1.x;
    float y1 = s.p1.y;

    if (y0 > y1)
    {
      float tmp;
      tmp = y0; y0 = y1; y1 = tmp;
      tmp = x0; x0 = x1; x1 = tmp;
    }

    float dy = y1 - y0;

    if (y0 == y1) continue;

    float slope = (x1 - x0) / dy;

    for (int j = 0; j < 32; j++) {
      float scan_y = tile_y0 + float(j) + 0.5;

      if (scan_y < y0 || scan_y >= y1)
        continue;

      float x = x0 + (scan_y - y0) * slope;

      if (x >= tile_x0 && x < tile_x1) {
        mask ^= (1u << uint(j));
      }
    }
  }

  uint final_mask = subgroupXor(mask);

  if (subgroupElect())
  {
    prefix_parity[tile_id] = final_mask;
  }
}

