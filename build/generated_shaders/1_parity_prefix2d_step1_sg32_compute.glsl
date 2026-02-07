#version 450

/* #undef PREFIX_OUTPUT */

#define DO_XOR 

#extension GL_KHR_shader_subgroup_basic      : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 64) in;

#ifdef PREFIX_OUTPUT
layout(set = 0, binding = 10, std430) buffer TileOffsets {
    uint prefix_parity[];
};
#endif

layout(set = 0, binding = 10, std430) buffer TileNSegments {
    uint prefix_parity[];
};

layout(set = 0, binding = 8, std430) buffer SubgroupTmp {
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

#define SG_SIZE 32

#define SGS_PER_WG 64 / SG_SIZE

shared uint subgroup_totals[SGS_PER_WG]; 

void main() {
  uint lid = gl_LocalInvocationID.x;
  uint block = gl_WorkGroupID.x;
  uint row = gl_WorkGroupID.y;

  uint x = block * 64+ lid;
  uint blocks_per_row = (pc.n_tiles_x + 63) / 64;
  
  uint gid = row * pc.n_tiles_x + x; 

  uint prefix_val = 0u;
  if (x < pc.n_tiles_x)
    prefix_val = prefix_parity[gid];

#ifdef DO_XOR
  uint prefix = subgroupExclusiveXor(prefix_val);
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
  if (x < pc.n_tiles_x)
    prefix_parity[gid] = prefix;
#else 
  if (x < pc.n_tiles_x)
    prefix_parity[gid] = prefix;
#endif 

  if(block < blocks_per_row && (lid == 63 || x == pc.n_tiles_x - 1))
  {
#ifdef DO_XOR 
    subgroup_tmp[row * blocks_per_row + block] = prefix ^ prefix_val;
#else 
    subgroup_tmp[row * blocks_per_row + block] = prefix + prefix_val;
#endif
  }
}

