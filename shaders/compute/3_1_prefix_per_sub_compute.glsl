#version 450

#extension GL_KHR_shader_subgroup_basic      : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 32 * 32) in;

layout(set = 0, binding = 3, std430) buffer PrefixParity {
    uint prefix_parity[];
};

layout(set = 0, binding = 4, std430) buffer SubgroupTmp {
    uint subgroup_tmp[];
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
  uint row      = gl_WorkGroupID.x;
  uint lane     = gl_SubgroupInvocationID;
  uint subgroup = gl_SubgroupID;

  uint tile_x = subgroup * gl_SubgroupSize + lane;
  if (tile_x >= pc.n_tiles_x)
    return;

  uint idx = row * pc.n_tiles_x + tile_x;

  uint val = prefix_parity[idx];
  uint prefix = subgroupExclusiveXor(val);
  prefix_parity[idx] = prefix; 

  uint subgroups_per_row = (pc.n_tiles_x + gl_SubgroupSize - 1) / gl_SubgroupSize;

  if(subgroups_per_row > 1) {
    uint subgroup_total = subgroupXor(val);
    if(subgroupElect()) {
      uint subgroup_id = row * subgroups_per_row + subgroup;

      subgroup_tmp[subgroup_id] = subgroup_total;
    }
  }
}

