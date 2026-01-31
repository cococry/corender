#version 450

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

layout(set = 0, binding = 4, std430) buffer macrotileCursor {
    uint macrotile_cursor[];
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

void bin_to_macrotile(uint macrotile_idx, uint segment_id) {
    uint local = atomicAdd(macrotile_cursor[macrotile_idx], 1);
    if (local >= macrotile_n_segments[macrotile_idx])
        return;

    uint out_idx = macrotile_offsets[macrotile_idx] + local;
    macrotile_segments[out_idx] = segment_id;
}

void main() {
  uint seg_id = gl_GlobalInvocationID.x;

  if(seg_id >= pc.n_segments)  return;

  Segment seg = segments[seg_id];
  vec2 p0 = seg.p0;
  vec2 p1 = seg.p1;

  vec2 seg_max = max(p0, p1);
  vec2 seg_min = min(p0, p1);
  float macrotile_size = float(pc.macrotile_size);

  ivec2 macrotile_start = ivec2(floor(seg_min / macrotile_size));
  ivec2 macrotile_end = ivec2(floor(seg_max/ macrotile_size));
  macrotile_start = clamp(macrotile_start, ivec2(0), ivec2(int(pc.n_macrotiles_x - 1), int(pc.n_macrotiles_y - 1)));
  macrotile_end = clamp(macrotile_end, ivec2(0), ivec2(int(pc.n_macrotiles_x - 1), int(pc.n_macrotiles_y - 1)));

  for(int ty = macrotile_start.y; ty <= macrotile_end.y; ty++) {
    for(int tx = macrotile_start.x; tx <= macrotile_end.x; tx++) {
      uint idx = atomicAdd(macrotile_cursor[ty * pc.n_macrotiles_x + tx], 1);
      if(idx >= macrotile_n_segments[ty * pc.n_macrotiles_x + tx]) continue;
      uint base = macrotile_offsets[ty * pc.n_macrotiles_x + tx];
      macrotile_segments[base + idx] = seg_id;
    }
  }
}


