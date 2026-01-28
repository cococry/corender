#version 450

#extension GL_KHR_shader_subgroup_basic      : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 256) in;

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
  uint row   = gl_WorkGroupID.y;
  uint lane  = gl_LocalInvocationID.x;

  uint blocks_per_row = (pc.n_tiles_x + 255) / 256;
  uint gid = row * blocks_per_row + lane;

  if (lane >= blocks_per_row) return;

  uint x = subgroup_tmp[gid];

  // ---------------------------------------------------
  // Step 1: subgroup exclusive prefix
  // ---------------------------------------------------
  uint prefix = subgroupExclusiveXor(x);

  // ---------------------------------------------------
  // Step 2: subgroup sum
  // ---------------------------------------------------
  uint sg_sum = subgroupXor(x);

  // One lane writes subgroup total
  if (subgroupElect())
    subgroup_totals[gl_SubgroupID] = sg_sum;

  barrier();

  // ---------------------------------------------------
  // Step 3: subgroup 0 scans subgroup totals
  // ---------------------------------------------------
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

  // ---------------------------------------------------
  // Step 4: add subgroup offset
  // ---------------------------------------------------
  sg_offset = subgroup_totals[gl_SubgroupID];
  prefix ^= sg_offset;

  // ---------------------------------------------------
  // Write tile prefix
  // ---------------------------------------------------
  subgroup_tmp[gid] = prefix;

}

