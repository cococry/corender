#version 450

#define PREFIX_OUTPUT

/* #undef DO_XOR */

#extension GL_KHR_shader_subgroup_basic      : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 64) in;

#ifdef PREFIX_OUTPUT
layout(set = 0, binding = 12, std430) buffer TileOffsets {
    uint macro_block_offsets[];
};
#endif

layout(set = 0, binding = 11, std430) buffer TileNSegments {
    uint macro_block_counts[];
};

layout(set = 0, binding = 8, std430) buffer SubgroupTmp {
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


#define SG_SIZE 8

#define SGS_PER_WG 64 / SG_SIZE

shared uint subgroup_totals[SGS_PER_WG]; 

void main() {
  uint lid = gl_LocalInvocationID.x;
  uint block = gl_WorkGroupID.x;
  uint row = gl_WorkGroupID.y;

  uint x = block * 64+ lid;
  uint blocks_per_row = (pc.n_seg_blocks + 63) / 64;
  
  uint gid = row * pc.n_seg_blocks + x; 

  uint prefix_val = 0u;
  if (x < pc.n_seg_blocks)
    prefix_val = macro_block_counts[gid];

#ifdef DO_XOR
  uint prefix = subgroupInclusiveXor(prefix_val);
#else 
  uint prefix = subgroupExclusiveAdd(prefix_val);
#endif

#ifdef DO_XOR
  uint sg_sum = subgroupXor(prefix_val);
#else
  uint sg_sum = subgroupAdd(prefix_val);
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

#ifdef PREFIX_OUTPUT 
  if (x < pc.n_seg_blocks)
    macro_block_offsets[gid] = prefix;
#else 
  if (x < pc.n_seg_blocks)
    macro_block_counts[gid] = prefix;
#endif 

  if(block < blocks_per_row && (lid == 63 || x == pc.n_seg_blocks - 1))
  {
#ifdef DO_XOR 
    subgroup_tmp[row * blocks_per_row + block] = prefix ^ prefix_val;
#else 
    subgroup_tmp[row * blocks_per_row + block] = prefix + prefix_val;
#endif
  }
}

