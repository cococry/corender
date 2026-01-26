#version 450 core

layout (local_size_x = 256) in;

layout(set = 0, binding = 3, std430) buffer SubgroupTmp {
  uint subgroup_tmp_binning[];
} ;

layout(set = 0, binding = 2, std430) buffer TileOffsets {
  uint tile_offsets[];
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
  uint gid = gl_GlobalInvocationID.x;
  uint wid = gl_WorkGroupID.x;

  if(gid >= pc.n_tiles_x * pc.n_tiles_y) return;

  tile_offsets[gid] += subgroup_tmp_binning[gid / 256];
}
