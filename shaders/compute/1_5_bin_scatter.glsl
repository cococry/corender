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
  uint screen_w,  screen_h;
  uint n_tiles_x, n_tiles_y;
  uint tile_size;
  
  uint n_segments;
  uint n_paths;

  uint fill_rule;
} pc;


void main() {
  uint segment_id = gl_GlobalInvocationID.x;

  if(segment_id >= pc.n_segments) return;

  Segment seg = segments[segment_id]; 

  vec2 seg_min = min(seg.p0, seg.p1);
  vec2 seg_max = max(seg.p0, seg.p1);

  ivec2 tile_min = ivec2(floor(seg_min / float(pc.tile_size)));
  ivec2 tile_max = ivec2(floor((seg_max) / float(pc.tile_size)));

  tile_min = clamp(tile_min, ivec2(0), ivec2(int(pc.n_tiles_x - 1), int(pc.n_tiles_y - 1)));
  tile_max = clamp(tile_max, ivec2(0), ivec2(int(pc.n_tiles_x - 1), int(pc.n_tiles_y - 1)));

  for (int ty = tile_min.y; ty <= tile_max.y; ty++) {
    for (int tx = tile_min.x; tx <= tile_max.x; tx++) {
      uint tile_idx = uint(ty) * pc.n_tiles_x + uint(tx);

      uint local = atomicAdd(tile_cursor[tile_idx], 1);
      if (local >= tile_n_segments[tile_idx]) continue;
      uint out_idx = tile_offsets[tile_idx] + local;

      tile_segments[out_idx]  = segment_id; 
    }
  }
}
