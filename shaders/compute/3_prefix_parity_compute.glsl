#version 450

#extension GL_KHR_shader_subgroup_basic      : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 32) in;

layout(set = 0, binding = 3, std430) buffer BaseParity {
    uint num_crossings[];
};

layout(set = 0, binding = 4, std430) buffer PrefixParity {
    uint prefix[];
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
    uint row     = gl_WorkGroupID.x;
    uint lane    = gl_SubgroupInvocationID;

    if (row >= pc.n_tiles_y * pc.tile_size)
        return;

    uint idx = row * pc.n_tiles_x + lane;

    uint val = 0;
    if (lane < pc.n_tiles_x)
        val = uint(num_crossings[idx]) & 1;

    uint excl = subgroupExclusiveXor(val);

    if (lane < pc.n_tiles_x)
        prefix[idx] = excl; 
}

