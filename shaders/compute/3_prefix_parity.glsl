#version 450

#extension GL_KHR_shader_subgroup_basic      : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 32) in;

layout(set = 0, binding = 3, std430) buffer BaseParity {
    uint base_parity[];
};

layout(set = 0, binding = 4, std430) buffer PrefixParity {
    uint subgroup_prefix_parity[];
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
  uint row     = gl_WorkGroupID.x;
  uint group_x = gl_WorkGroupID.y;
  uint lane    = gl_SubgroupInvocationID;

  uint tile_x = group_x * gl_SubgroupSize + lane;
  if (row >= pc.n_tiles_y * pc.tile_size ||
      tile_x >= pc.n_tiles_x)
    return;

  uint idx = row * pc.n_tiles_x + tile_x;
  uint val = base_parity[idx] & 1u;

  uint excl = subgroupExclusiveXor(val);

  subgroup_prefix_parity[idx] = excl;
}

