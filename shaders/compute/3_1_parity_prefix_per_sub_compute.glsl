#version 450

#extension GL_KHR_shader_subgroup_basic      : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 256) in;

layout(set = 0, binding = 6, std430) buffer PrefixParity {
  uint prefix_parity[];
};
layout(set = 0, binding = 7, std430) buffer SubgroupTmp {
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

shared uint subgroup_totals[8]; 

void main() {
  uint lid = gl_LocalInvocationID.x;
  uint block = gl_WorkGroupID.x;
  uint row = gl_WorkGroupID.y;

  uint x = block * 256 + lid;
  uint blocks_per_row = (pc.n_tiles_x + 255) / 256;
  
  uint gid = row * pc.n_tiles_x + x; 

  uint prefix_val = 0u;
  if (x < pc.n_tiles_x)
    prefix_val = prefix_parity[gid];

  uint prefix = subgroupExclusiveXor(prefix_val);

  uint sg_sum = subgroupXor(prefix_val);

  if (subgroupElect())
    subgroup_totals[gl_SubgroupID] = sg_sum;

  barrier();

  uint sg_offset = 0;

  if (gl_SubgroupID == 0)
  {
    uint lane = gl_SubgroupInvocationID;
    uint val = (lane < 8) ? subgroup_totals[lane] : 0;

    uint off = subgroupExclusiveXor(val);

    if (lane < 8)
      subgroup_totals[lane] = off;
  }


  barrier();

  sg_offset = subgroup_totals[gl_SubgroupID];
  prefix ^= sg_offset;

  if (x < pc.n_tiles_x)
    prefix_parity[gid] = prefix;

  if (lid == 255 || x == pc.n_tiles_x - 1)
  {
    subgroup_tmp[row * blocks_per_row + block] = prefix ^ prefix_val;
  }
}

