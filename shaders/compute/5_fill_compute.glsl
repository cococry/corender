#version 450
layout(local_size_x = 32, local_size_y = 32) in;

#extension GL_KHR_shader_subgroup_basic      : enable
struct Segment {
  vec2 p0;
  vec2 p1;
};


layout(set = 0, binding = 0, std430) readonly buffer Segments {
  Segment segments[];
};

layout(set = 0, binding = 6, std430) buffer TilenSegments {
    uint tile_n_segments[];
};

layout(set = 0, binding = 7, std430) buffer TileOffsets {
    uint tile_offsets[];
};

layout(set = 0, binding = 5, std430) buffer TileSegments {
    uint tile_segments[];
};

layout(set = 0, binding = 9, std430) readonly buffer PrefixParity {
  uint prefix_parity[]; 
};

layout(set = 0, binding = 11, rgba8) uniform image2D outImage;

layout(push_constant) uniform push_constant {
  uint screen_w, screen_h;
  uint n_tiles_x, n_tiles_y;
  uint n_macrotiles_x, n_macrotiles_y;
  uint tile_size, macrotile_size;

  uint n_segments;
  uint n_paths;
  uint fill_rule;
} pc;


const uint MAX_SEGMENTS_PER_TILE = 32;
const uint SUBGROUP_SIZE = 32;
const uint MAX_SUBGROUPS = 32;

void main() {
    uvec2 tile = gl_WorkGroupID.xy;
  uint scan  = gl_LocalInvocationID.y;

  int y = int(tile.y * pc.tile_size + scan);
  if (y < 0 || y >= int(pc.screen_h))
    return;

  uint tile_id = tile.y * pc.n_tiles_x + tile.x;

  uint count = tile_n_segments[tile_id]; 
  int x0 = int(tile.x * pc.tile_size);

  int coverage[32];
  for (int i = 0; i < pc.tile_size; i++)
    coverage[i] = 0;

  if (count > 0) {
    uint base = tile_offsets[tile_id]; 

    for (uint i = 0; i < count; i++) {
      uint seg_id = tile_segments[base + i];
      Segment s = segments[seg_id];

      float y0 = s.p0.y;
      float y1 = s.p1.y;
      if (y0 == y1)
        continue;

      float scan_y = float(y) + 0.5;

      bool hit =
        (y0 <= scan_y && scan_y < y1) ||
        (y1 <= scan_y && scan_y < y0);

      if (!hit)
        continue;

      float t = (scan_y - y0) / (y1 - y0);
      float x_hit = mix(s.p0.x, s.p1.x, t);

      if (x_hit >= float(x0) &&
          x_hit <  float(x0 + int(pc.tile_size))) {

        int xi = int(floor(x_hit)) - x0;
        if (xi >= 0 && xi < int(pc.tile_size))
          coverage[xi] ^= 1;
      }
    }
  }

  int row = int(tile.y * pc.tile_size + scan);
  int idx = int(row * pc.n_tiles_x + tile.x);

  int parity = int((prefix_parity[tile_id] >> scan) & 1u);

  for (int dx = 0; dx < pc.tile_size; dx++) {
    parity ^= coverage[dx];

    if (parity == 1) {
      imageStore(
          outImage,
          ivec2(x0 + dx, y),
          vec4(count >= 1 ? 1.0 : 0.0, 0.0,  1.0, 1.0)
          );
    }
  }
}

