#version 450 core

/* #undef PREFIX_OUTPUT */
/* #undef DO_XOR */

layout(local_size_x = 64) in;

#ifdef PREFIX_OUTPUT
layout(set = 0, binding = 2, std430) buffer TileOffsets {
  uint macrotile_offsets[];
};
#else
layout(set = 0, binding = 2, std430) buffer TileOffsets {
  uint macrotile_offsets[];
};
#endif

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

void main() {
  uint gid     = gl_GlobalInvocationID.x;
  uint n_total = pc.n_bins * pc.n_paths;

  bool is_active = gid < n_total;

  if (is_active) {
#ifdef DO_XOR
    macrotile_offsets[gid] ^= subgroup_tmp[gid / 64u];
#else
    macrotile_offsets[gid] += subgroup_tmp[gid / 64u];
#endif
  }
}
