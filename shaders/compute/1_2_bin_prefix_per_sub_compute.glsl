#version 450

#extension GL_KHR_shader_subgroup_basic      : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 256) in;

layout(set = 0, binding = 1, std430) buffer TileNSegments {
    uint tile_n_segments[];
};

layout(set = 0, binding = 2, std430) buffer TileOffsets {
    uint tile_offsets[];
};

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

  uint x = (gid < pc.n_tiles_x * pc.n_tiles_y) ? tile_n_segments[gid] : 0;

  uint prefix = subgroupExclusiveAdd(x);

  uint sg_sum = subgroupAdd(x);

  if (subgroupElect())
    subgroup_totals[gl_SubgroupID] = sg_sum;

  barrier();

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

  sg_offset = subgroup_totals[gl_SubgroupID];
  prefix += sg_offset;

  if (gid < pc.n_tiles_x * pc.n_tiles_y)
    tile_offsets[gid] = prefix;

  if (lid == 255)
  {
    subgroup_tmp_binning[groupID] = prefix + x;
  }
}

