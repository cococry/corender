#version 450

layout(local_size_x = 256) in;

struct Segment {
    vec2 p0;
    vec2 p1;
};

layout(set = 0, binding = 0, std430) readonly buffer Segments {
    Segment segments[];
};

layout(set = 0, binding = 1, std430) buffer TileNSegments {
    uint tile_n_segments[];
};

layout(set = 0, binding = 2, std430) buffer TileOffsets {
    uint tile_offsets[];
};

layout(set = 0, binding = 4, std430) buffer TileSegments {
    uint tile_segments[];
};

layout(set = 0, binding = 5, std430) buffer TileCursor {
    uint tile_cursor[];
};

layout(push_constant) uniform push_constant {
    uint screen_w, screen_h;
    uint n_tiles_x, n_tiles_y;
    uint tile_size;

    uint n_segments;
    uint n_paths;
    uint fill_rule;
} pc;

void bin_to_tile(uint tile_idx, uint segment_id) {
    uint local = atomicAdd(tile_cursor[tile_idx], 1);
    if (local >= tile_n_segments[tile_idx])
        return;

    uint out_idx = tile_offsets[tile_idx] + local;
    tile_segments[out_idx] = segment_id;
}

void main() {
  uint seg_id = gl_GlobalInvocationID.x;

  if(seg_id >= pc.n_segments)  return;

  Segment seg = segments[seg_id];
  vec2 p0 = seg.p0;
  vec2 p1 = seg.p1;

  float tile_size = float(pc.tile_size);

  ivec2 tile_start = ivec2(floor(p0 / tile_size));
  ivec2 tile_end = ivec2(floor(p1 / tile_size));
  tile_start = clamp(tile_start, ivec2(0), ivec2(int(pc.n_tiles_x - 1), int(pc.n_tiles_y - 1)));
  tile_end = clamp(tile_end, ivec2(0), ivec2(int(pc.n_tiles_x - 1), int(pc.n_tiles_y - 1)));

  vec2 d = p1 - p0;

  if(length(d) < 1e-6) {
    uint tile_idx = uint(tile_start.y) * pc.n_tiles_x + uint(tile_start.x);
    atomicAdd(tile_n_segments[tile_idx], 1);
    return;
  }
  int x_step = (d.x > 0.0) ? 1 : ((d.x < 0.0) ? -1 : 0); 
  int y_step = (d.y > 0.0) ? 1 : ((d.y < 0.0) ? -1 : 0); 

  float dx_inv = (d.x != 0.0f) ? 1.0f / abs(d.x) : 1e30;
  float dy_inv = (d.y != 0.0f) ? 1.0f / abs(d.y) : 1e30;

  float next_boundary_x = x_step > 0 ? (tile_start.x + 1) * tile_size : tile_start.x * tile_size;
  float next_boundary_y = y_step > 0 ? (tile_start.y + 1) * tile_size : tile_start.y * tile_size;

  float slope_max_x =
    (x_step != 0) ? abs(next_boundary_x - p0.x) * dx_inv : 1e30;

  float slope_max_y =
    (y_step != 0) ? abs(next_boundary_y - p0.y) * dy_inv : 1e30;

  float slope_delta_x = tile_size * dx_inv;
  float slope_delta_y = tile_size * dy_inv;

  ivec2 tile = tile_start;

  while (true) {
    uint tile_idx = uint(tile.y) * pc.n_tiles_x + uint(tile.x);

    bin_to_tile(tile_idx, seg_id);

    if (tile == tile_end) break;

    if (slope_max_x < slope_max_y) {
      slope_max_x += slope_delta_x;
      tile.x += x_step;
    } else if (slope_max_x > slope_max_y) {
      slope_max_y += slope_delta_y;
      tile.y += y_step;
    }
    else {
      slope_max_x += slope_delta_x;
      slope_max_y += slope_delta_y;
      tile.x += x_step;
      tile.y += y_step;
    }
    if (tile.x < 0 || tile.y < 0 ||
        tile.x >= int(pc.n_tiles_x) ||
        tile.y >= int(pc.n_tiles_y))
      break;
  }

}


