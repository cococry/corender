
#version 450

#extension GL_KHR_shader_subgroup_basic      : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 256) in;

layout(set = 0, binding = 3, std430) buffer SubgroupTmpBinning {
    uint subgroup_tmp_binning[];
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
  uint gid = gl_GlobalInvocationID.x;
  uint lid = gl_LocalInvocationID.x;
  uint groupID = gl_WorkGroupID.x;

  uint n_tiles  = pc.n_tiles_x * pc.n_tiles_y;
  uint n_groups = (n_tiles + 255) / 256;

  if (gid >= n_groups) return;


  uint x = subgroup_tmp_binning[gid];

  // ---------------------------------------------------
  // Step 1: subgroup exclusive prefix
  // ---------------------------------------------------
  uint prefix = subgroupExclusiveAdd(x);

  // ---------------------------------------------------
  // Step 2: subgroup sum
  // ---------------------------------------------------
  uint sg_sum = subgroupAdd(x);

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

    uint off = subgroupExclusiveAdd(val);

    if (lane < 8)
      subgroup_totals[lane] = off;
  }

  barrier();

  // ---------------------------------------------------
  // Step 4: add subgroup offset
  // ---------------------------------------------------
  sg_offset = subgroup_totals[gl_SubgroupID];
  prefix += sg_offset;

  // ---------------------------------------------------
  // Write tile prefix
  // ---------------------------------------------------
  subgroup_tmp_binning[gid] = prefix;

}

