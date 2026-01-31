#version 450 core

#define PREFIX_OUTPUT

/* #undef DO_XOR */

layout (local_size_x = 64) in;

#ifdef PREFIX_OUTPUT
layout(set = 0, binding = 2, std430) buffer TileOffsets {
  uint macrotile_offsets[];
};
#else 
layout(set = 0, binding = 1, std430) buffer TileOffsets {
  uint macrotile_n_segments[];
};
#endif

layout(set = 0, binding = 8, std430) buffer SubgroupTmp {
  uint subgroup_tmp[];
};

layout(push_constant) uniform push_constant {
  uint screen_w,  screen_h;
  uint n_tiles_x, n_tiles_y;
  uint n_macrotiles_x, n_macrotiles_y;
  uint tile_size, macrotile_size;

  uint n_segments;
  uint n_paths;

  uint fill_rule;
} pc;


void main() {
  uint gid = gl_GlobalInvocationID.x;
  uint wid = gl_WorkGroupID.x;

  bool is_active = gid < pc.n_macrotiles_x * pc.n_macrotiles_y;

  if(is_active) {
#ifdef DO_XOR 
  macrotile_offsets[gid] ^= subgroup_tmp[gid / 64];
#else 
  macrotile_offsets[gid] += subgroup_tmp[gid / 64];
#endif 
  }
}
