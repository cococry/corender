#version 450 core

/* #undef PREFIX_OUTPUT */
#define DO_XOR 

layout (local_size_x = 64) in;

layout(set = 0, binding = 8, std430) buffer SubgroupTmp {
    uint subgroup_tmp[];
};

#ifdef PREFIX_OUTPUT
layout(set = 0, binding = 10, std430) buffer TileOffsets {
  uint prefix_parity[];
};
#else 
layout(set = 0, binding = 10, std430) buffer TileOffsets {
  uint prefix_parity[];
};
#endif

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
  uint block = gl_WorkGroupID.x;
  uint row = gl_WorkGroupID.y;
  uint lane = gl_LocalInvocationID.x;
  uint blocks_per_row = (pc.n_tiles_x + 63) / 64;

  uint x = block * 64 + lane;

  uint idx = row * pc.n_tiles_x + x;

  uint block_idx = row * blocks_per_row + block;

  bool is_active = block_idx < blocks_per_row * pc.n_tiles_y && x < pc.n_tiles_x;
  if(is_active) {
#ifdef DO_XOR
    prefix_parity[idx] ^= subgroup_tmp[block_idx];
#else 
    prefix_parity[idx] += subgroup_tmp[block_idx];
#endif 
  }
}

