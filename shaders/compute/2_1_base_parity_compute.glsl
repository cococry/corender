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

layout(set = 0, binding = 1, std430) readonly buffer TileNSegments {
  uint tile_n_segments[];
};

layout(set = 0, binding = 2, std430) readonly buffer TileOffsets {
  uint tile_offsets[];
};

layout(set = 0, binding = 4, std430) readonly buffer TileSegments {
  uint tile_segments[];
};

layout(set = 0, binding = 6, std430) writeonly buffer PrefixParity {
  uint prefix_parity[];
};

layout(push_constant) uniform PC {
  uint screen_w, screen_h;
  uint n_tiles_x, n_tiles_y;
  uint tile_size;
} pc;
void main()
{
  uvec2 tile = gl_WorkGroupID.xy;
  uint lane  = gl_LocalInvocationID.x;

  uint tile_id = tile.y * pc.n_tiles_x + tile.x;

  uint base  = tile_offsets[tile_id];
  uint count = tile_n_segments[tile_id];

  if(count == 0) return;

  float tile_x0 = float(tile.x) * float(pc.tile_size);
  float tile_x1 = tile_x0 + float(pc.tile_size);

  float tile_y0 = float(tile.y) * float(pc.tile_size);

  uint mask = 0u;
  for (uint i = lane; i < count; i += 32) {
    uint seg_id = tile_segments[base + i];
    Segment s   = segments[seg_id];

    float y0 = s.p0.y;
    float y1 = s.p1.y;
    
    float x0 = s.p0.x;
    float x1 = s.p1.x;
    
    if (y0 > y1) {
      float tmp_y = y0;
      y0 = y1;
      y1 = tmp_y;

      float tmp_x = x0;
      x0 = x1;
      x1 = tmp_x;
    }

    if (y0 == y1) continue;

    float ymin_f = min(y0, y1) - tile_y0;
    float ymax_f = max(y0, y1) - tile_y0 - 0.5;

    int ymin = clamp(int(floor(ymin_f)), 0, 31);
    int ymax = clamp(int(ceil (ymax_f)), 0, 32);

    float dy = y1 - y0;
    float slope = (x1 - x0) / dy;
    float scan_y = tile_y0 + float(ymin) + 0.5;

    float x = x0 + (scan_y - y0) * slope;

    for (int j = ymin; j < ymax; j++) {
      if (x>= tile_x0 && x < tile_x1) {
        mask ^= (1u << uint(j));
      }
      x += slope;
    }
  }

  uint final_mask = subgroupXor(mask);

  if (subgroupElect())
  {
    prefix_parity[tile_id] = final_mask;
  }
}

