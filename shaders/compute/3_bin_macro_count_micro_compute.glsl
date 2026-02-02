#version 450

#extension GL_KHR_shader_subgroup_basic  : enable
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_vote : enable

layout(local_size_x = 64) in;

struct Segment {
    vec2 p0;
    vec2 p1;
};

layout(set = 0, binding = 11, std430) buffer macrotileNSegmentsMicro {
    uint macrotile_n_segments_micro[];
};
layout(set = 0, binding = 13, std430) buffer TileCountsMacro {
    uint tile_counts_micro[];
};

layout(set = 0, binding = 6, std430) buffer TileCounts {
    uint tile_counts[];
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


shared uint tmp[64];

void main() {
  if(gl_WorkGroupID.x >= pc.n_macrotiles_x ||
      gl_WorkGroupID.y >= pc.n_macrotiles_y)
    return;

  uint macro_id =
    gl_WorkGroupID.y * pc.n_macrotiles_x +
    gl_WorkGroupID.x;

  uint lid = gl_LocalInvocationID.x;

  uint tile = lid;


  ivec2 macro_tile_base =
    ivec2(int(gl_WorkGroupID.x) * 8,
        int(gl_WorkGroupID.y) * 8);

  uint gx = uint(macro_tile_base.x) + (tile % 8);
  uint gy = uint(macro_tile_base.y) + (tile / 8);
      
  uint global_tile_id = gy * pc.n_tiles_x + gx;

  tmp[lid] = tile_counts[global_tile_id];

  barrier();

  if(lid < 64) {
    for(uint offset = 32; offset > 0; offset >>= 1) {
      if(lid < offset)
        tmp[lid] += tmp[lid + offset];
      barrier();
    }
  }

  if(lid == 0)
    macrotile_n_segments_micro[macro_id] = tmp[0];

  
}

