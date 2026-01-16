#version 450

layout(local_size_x = 1, local_size_y = 32) in;

struct Segment {
    vec2 p0;
    vec2 p1;
};

struct TileHeader {
    uint n_segments;
};

layout(set = 0, binding = 0, std430) readonly buffer Segments {
    Segment segments[];
};

layout(set = 0, binding = 1, std430) readonly buffer TileHeaders {
    TileHeader tile_headers[];
};

layout(set = 0, binding = 2, std430) readonly buffer TileSegmentIndices {
    uint tile_segment_indices[];
};

layout(set = 0, binding = 4, std430) buffer TileParity{
  uint tile_parity[];
};


layout(push_constant) uniform PC {
    uint screen_w, screen_h;
    uint n_tiles_x, n_tiles_y;
    uint tile_size;
    uint n_segments;
    uint n_paths;
    uint fill_rule;
} pc;

const uint MAX_SEGMENTS_PER_TILE = 32;

void main() {
  uvec2 tile = gl_WorkGroupID.xy;
  uint scan = gl_LocalInvocationID.y;

  int y = int(tile.y * pc.tile_size + scan);
  if (y < 0 || y >= int(pc.screen_h)) return;

  uint tile_id = tile.y * pc.n_tiles_x + tile.x;

  uint count = min(tile_headers[tile_id].n_segments,
      MAX_SEGMENTS_PER_TILE);

  if(count == 0) return;

  int x0 = int(tile.x * pc.tile_size);
  int parity_bit = 0;

  uint base  = tile_id * MAX_SEGMENTS_PER_TILE;
  for (uint i = 0; i < count; i++) {
    uint seg_id = tile_segment_indices[base + i];
    Segment s = segments[seg_id];
    float y0 = s.p0.y;
    float y1 = s.p1.y;

    if (y0 == y1) continue; // ignore horizontal edges

    float scan_y = float(y) + 0.5;

    bool hit =
      (y0 <= scan_y && scan_y < y1) ||
      (y1 <= scan_y && scan_y < y0);

    if(!hit) continue;

    float t = (scan_y - y0) / (y1 - y0);
    float x_hit = mix(s.p0.x, s.p1.x, t);
    if (x_hit >= float(x0) &&
        x_hit <  float(x0 + int(pc.tile_size))) {
      parity_bit ^= 1;
    }
  }

  uint parity_idx = (tile.y * pc.n_tiles_x + tile.x) * pc.tile_size + scan;
  tile_parity[parity_idx] = uint(parity_bit);

}
