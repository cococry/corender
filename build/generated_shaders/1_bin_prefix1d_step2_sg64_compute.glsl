#version 450

#extension GL_KHR_shader_subgroup_basic      : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 64) in;

/* #undef DO_XOR */

layout(set = 0, binding = 8, std430) buffer SubgroupTmpBinning {
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

#define SG_SIZE 64

#define SGS_PER_WG 64 / SG_SIZE

shared uint subgroup_totals[SGS_PER_WG]; 

void main() {
  uint gid = gl_GlobalInvocationID.x;
  uint lid = gl_LocalInvocationID.x;
  uint groupID = gl_WorkGroupID.x;

  uint n_tiles  = pc.n_macrotiles_x * pc.n_macrotiles_y;
  uint n_groups = (n_tiles + 63) / 64;

  bool is_active = gid < n_groups;

  uint x = is_active ? subgroup_tmp[gid] : 0;

#ifdef DO_XOR
  uint prefix = subgroupExclusiveXor(x);
#else 
  uint prefix = subgroupExclusiveAdd(x);
#endif

#ifdef DO_XOR
  uint sg_sum = subgroupXor(x);
#else 
  uint sg_sum = subgroupAdd(x);
#endif

  if (subgroupElect())
    subgroup_totals[gl_SubgroupID] = sg_sum;

  barrier();

  uint sg_offset = 0;

  if (gl_SubgroupID == 0)
  {
    uint lane = gl_SubgroupInvocationID;

    uint val = (lane < SGS_PER_WG) ? subgroup_totals[lane] : 0;

#ifdef DO_XOR
    uint off = subgroupExclusiveXor(val);
#else 
    uint off = subgroupExclusiveAdd(val);
#endif

    if (lane < SGS_PER_WG)
      subgroup_totals[lane] = off;
  }

  barrier();

  sg_offset = subgroup_totals[gl_SubgroupID];

#ifdef DO_XOR
  prefix ^= sg_offset;
#else 
  prefix += sg_offset;
#endif 

  if(is_active)
    subgroup_tmp[gid] = prefix;
}

