#version 450

#extension GL_KHR_shader_subgroup_basic      : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 32) in;

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
  uint row = gl_WorkGroupID.x;
  uint lane = gl_LocalInvocationID.x;
  uint subgroups_per_row = (pc.n_tiles_x + gl_SubgroupSize - 1) / gl_SubgroupSize;
  if (lane >= subgroups_per_row) return;

  uint gid = row * subgroups_per_row + lane;
  uint val = subgroup_tmp[gid];
  uint prefix = subgroupExclusiveXor(val);
  subgroup_tmp[gid] = prefix;
}

