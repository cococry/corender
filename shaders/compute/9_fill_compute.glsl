#version 450
layout(local_size_x = 32) in;

#extension GL_KHR_shader_subgroup_basic      : enable
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

layout(set = 0, binding = 10, std430) buffer PrefixParity {
  uint prefix_parity[];
};
layout(set = 0, binding = 17, rgba8) uniform image2D outImage;

struct MacrotileMetadata {
    uint off;
    uint count;
};
layout(set = 0, binding = 16, std430) readonly buffer MacrotileMeta {
    MacrotileMetadata macrotile_metas[];
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


const uint MAX_SEGMENTS_PER_TILE = 32;
const uint SUBGROUP_SIZE = 32;
const uint MAX_SUBGROUPS = 32;


void main() {
  uvec2 tile = gl_WorkGroupID.xy;
  uint scan = gl_LocalInvocationID.x;

  int y = int(tile.y * pc.tile_size + scan);
  if (y < 0 || y >= int(pc.screen_h))
    return;

  uint tile_id = tile.y * pc.n_tiles_x + tile.x;

  uint count = tile_n_segments[tile_id];

  int coverage[32];
  for (int i = 0; i < pc.tile_size; i++)
    coverage[i] = 0;

  int tile_x0 = int(tile.x * pc.tile_size);
  int tile_x1 = int(tile_x0 + pc.tile_size);

  if (count > 0) {
    uint base = tile_offsets[tile_id];

    for (uint i = 0; i < count; i++) {
      uint seg_id = tile_segments[base + i];
      Segment s = segments[seg_id];

      float x0 = s.p0.x;
      float y0 = s.p0.y;
      float x1 = s.p1.x;
      float y1 = s.p1.y;

      if (y0 > y1) {
        float tmp;
        tmp = y0; y0 = y1; y1 = tmp;
        tmp = x0; x0 = x1; x1 = tmp;
      }

      float dy = y1 - y0;
      if (y0 == y1) continue;

      float slope = (x1 - x0) / dy;
      float scan_y = float(y) + 0.5;

      if (scan_y < y0 || scan_y >= y1)
        continue;

      float x = x0 + (scan_y - y0) * slope;

      if (x >= float(tile_x0) && x < float(tile_x1)) {
        int xi = int(floor(x)) - int(tile_x0);
        if (xi >= 0 && xi < 32) {
          coverage[xi] ^= 1;
        }
      }
    }
  }

  int parity = int((prefix_parity[tile_id] >> scan) & 1u);

  for (int dx = 0; dx < pc.tile_size; dx++) {
    int px = tile_x0 + dx;
    if (px < 0 || px >= int(pc.screen_w))
      continue;

    parity ^= coverage[dx];

    vec4 color = vec4(0.1, 0.1, 0.1, 1.0);  // background

    if (parity == 1) {
      color = vec4(1.0, 0.0, 0.0, 1.0);     // fill
    }

    int macro_x = int(px / pc.macrotile_size);
    int macro_y = int(y / pc.macrotile_size);

    // assuming partition 0
    if(macrotile_metas[macro_y * pc.n_macrotiles_x + macro_x].count > 0) {
        color = vec4(0.2, 0.3, 0.8, 1.0);
    }

    /* microtile grid: every tile_size pixels */
    bool micro_v = (px % int(pc.tile_size)) == 0;
    bool micro_h = (y  % int(pc.tile_size)) == 0;

    /* macrotile grid: every macrotile_size pixels */
    bool macro_v = (px % int(pc.macrotile_size)) == 0;
    bool macro_h = (y  % int(pc.macrotile_size)) == 0;

    /* draw macro grid over micro grid */
    if (micro_v || micro_h) {
      color = vec4(1.0, 1.0, 1.0, 1.0);     // white microtile lines
    }

    if (macro_v || macro_h) {
      color = vec4(0.0, 1.0, 1.0, 1.0);     // cyan macrotile lines
    }

    imageStore(outImage, ivec2(px, y), color);
  }
}
