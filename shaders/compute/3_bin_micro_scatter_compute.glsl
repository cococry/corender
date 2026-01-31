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


layout(set = 0, binding = 5, std430) buffer TileSegments {
    uint tile_segments[];
};

layout(set = 0, binding = 6, std430) buffer TileNSegments{
    uint tile_n_segments[];
};


layout(set = 0, binding = 7, std430) buffer TileOffsets {
    uint tile_offsets[];
};

layout(set = 0, binding = 10, std430) buffer TileCursor {
    uint tile_cursor[];
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

void main() {
  uint macro_id = gl_WorkGroupID.y * pc.n_macrotiles_x + gl_WorkGroupID.x;

  uint count = macrotile_n_segments[macro_id];
  uint base = macrotile_offsets[macro_id];

  for(uint i = gl_LocalInvocationID.x; i < count; i += gl_WorkGroupSize.x) {
    uint seg_id = macrotile_segments[base + i];
    Segment seg = segments[seg_id];

    vec2 p0 = seg.p0;
    vec2 p1 = seg.p1;

    vec2 seg_max = max(p0, p1);
    vec2 seg_min = min(p0, p1);
    float tile_size = float(pc.tile_size);

    ivec2 tile_start = ivec2(floor(seg_min / tile_size));
    ivec2 tile_end = ivec2(floor(seg_max/ tile_size));

    ivec2 macro_base = ivec2(int(gl_WorkGroupID.x) * 8, int(gl_WorkGroupID.y) * 8);
    ivec2 macro_end = macro_base + ivec2(7); 

    tile_start = clamp(tile_start, macro_base, macro_end); 
    tile_end = clamp(tile_end, macro_base, macro_end); 

    for(uint ty = tile_start.y; ty <= tile_end.y; ty++) {
      for(uint tx = tile_start.x; tx <= tile_end.x; tx++) {
        uint tile_id = ty * pc.n_tiles_x + tx;
        uint idx = atomicAdd(tile_cursor[tile_id], 1);
        if(idx >= tile_n_segments[tile_id]) continue;
        uint micro_base = tile_offsets[tile_id];
        tile_segments[micro_base + idx] = seg_id; 
      }
    }
  }
}
