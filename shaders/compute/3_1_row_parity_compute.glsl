#version 450
#extension GL_KHR_shader_subgroup_basic      : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 32) in;


layout(set = 0, binding = 3, std430) buffer BaseParity {
    uint base_parity[];
};


layout(set = 0, binding = 4, std430) buffer SubgroupPrefix {
  uint subgroup_prefix_parity[];   // [row * n_tiles_x + tile_x]
};

layout(set = 0, binding = 5, std430) buffer RowPrefixParity {
  uint row_prefix_parity[];        // [row * MAX_SUBGROUPS + subgroup]
};

layout(push_constant) uniform PC {
  uint screen_w, screen_h;
  uint n_tiles_x, n_tiles_y;
  uint tile_size;
} pc;

const uint SUBGROUP_SIZE = 32;
const uint MAX_SUBGROUPS = 32;
void main() {
  uint row  = gl_WorkGroupID.x;
  uint lane = gl_SubgroupInvocationID;

  uint total_rows = pc.n_tiles_y * pc.tile_size;
  if (row >= total_rows)
    return;

  uint subgroup_count =
    (pc.n_tiles_x + SUBGROUP_SIZE - 1) / SUBGROUP_SIZE;

  uint val = 0;
  if (lane < subgroup_count) {
    uint last_tile_x =
      min(lane * SUBGROUP_SIZE + SUBGROUP_SIZE - 1,
          pc.n_tiles_x - 1);

    uint idx = row * pc.n_tiles_x + last_tile_x;

    val = int(subgroup_prefix_parity[idx]); 
  }

  uint excl = subgroupExclusiveXor(val);

  if (lane < subgroup_count) {
    row_prefix_parity[row * MAX_SUBGROUPS + lane] = excl;
  }

}
