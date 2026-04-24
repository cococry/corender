#version 450

/* #undef PREFIX_OUTPUT */
/* #undef DO_XOR */

#extension GL_KHR_shader_subgroup_basic      : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 64) in;

#ifdef PREFIX_OUTPUT
layout(set = 0, binding = 6, std430) buffer TileOffsets {
    uint macrotile_n_segments_micro[];
};
#endif

layout(set = 0, binding = 6, std430) buffer TileNSegments {
    uint macrotile_n_segments_micro[];
};

layout(set = 0, binding = 8, std430) buffer SubgroupTmpBinning {
    uint subgroup_tmp[];
};

layout(push_constant) uniform push_constant {
  uint screen_w,  screen_h;
  uint n_tiles_x, n_tiles_y;
  uint n_macrotiles_x, n_macrotiles_y;
  uint n_seg_blocks;
  uint n_bins;
  uint tile_size, macrotile_size;
  uint n_segments;
  uint n_paths;
  uint fill_rule;
} pc;

#define SG_SIZE 4
#define SGS_PER_WG (64 / SG_SIZE)

shared uint subgroup_totals[SGS_PER_WG];

void main() {
  uint lid = gl_LocalInvocationID.x;
  if (lid < SGS_PER_WG)
    subgroup_totals[lid] = 0u;
  barrier();

  uint group_id = gl_WorkGroupID.x;
  uint gid      = gl_GlobalInvocationID.x;
  uint n_total  = pc.n_macrotiles_x * pc.n_macrotiles_y;

  uint x = (gid < n_total) ? macrotile_n_segments_micro[gid] : 0u;

#ifdef DO_XOR
  uint prefix = subgroupExclusiveXor(x);
  uint sg_sum = subgroupXor(x);
#else
  uint prefix = subgroupExclusiveAdd(x);
  uint sg_sum = subgroupAdd(x);
#endif

  if (subgroupElect())
    subgroup_totals[gl_SubgroupID] = sg_sum;

  barrier();

  uint sg_offset = 0u;

  if (gl_SubgroupID == 0) {
    uint lane = gl_SubgroupInvocationID;
    uint val  = (lane < SGS_PER_WG) ? subgroup_totals[lane] : 0u;

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

#ifdef PREFIX_OUTPUT
  if (gid < n_total)
    macrotile_n_segments_micro[gid] = prefix;
#else
  if (gid < n_total)
    macrotile_n_segments_micro[gid] = prefix;
#endif

  if (subgroupElect() && gl_SubgroupID == SGS_PER_WG - 1u && gid < n_total) {
#ifdef DO_XOR
    subgroup_tmp[group_id] = sg_offset ^ sg_sum;
#else
    subgroup_tmp[group_id] = sg_offset + sg_sum;
#endif
  }
}
