#version 450 core

layout (local_size_x = 256) in;

layout(set = 0, binding = 7, std430) buffer SubgroupTmp {
    uint subgroup_tmp[];
};

layout(set = 0, binding = 6, std430) buffer PrefixParity {
  uint prefix_parity[];
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
  uint block = gl_WorkGroupID.x;
  uint row = gl_WorkGroupID.y;
  uint lane = gl_LocalInvocationID.x;
  uint blocks_per_row = (pc.n_tiles_x + 255) / 256;

  uint x = block*256 + lane;

  if(x >= pc.n_tiles_x) return;

  uint idx = row * pc.n_tiles_x + x;

  uint block_idx = row * blocks_per_row + block;
  prefix_parity[idx] ^= subgroup_tmp[block_idx];
}
