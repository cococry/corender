#version 450
#extension GL_GOOGLE_include_directive : require
#include "../../shared/pc.glsl"

/* #undef PREFIX_OUTPUT */
/* #undef DO_XOR */
#define PREFIX_SIGNED

layout(local_size_x = 256) in;

#if defined(DO_XOR) && defined(PREFIX_SIGNED)
#error "DO_XOR and PREFIX_SIGNED cannot be used together"
#endif

#ifdef PREFIX_SIGNED
#define T int
#else
#define T uint
#endif

#define WG_SIZE 256u

layout(set = 0, binding = SUBGROUP_TMP_BINDING, std430) buffer SubgroupTmp {
    T subgroup_tmp[];
};

#ifdef PREFIX_OUTPUT
layout(set = 0, binding = 12, std430) buffer PrefixOutput {
    T tile_events[];
};
#else
layout(set = 0, binding = 12, std430) buffer PrefixInput {
    T tile_events[];
};
#endif

T op_apply(T a, T b) {
#ifdef DO_XOR
    return a ^ b;
#else
    return a + b;
#endif
}

void main() {
    if (bump.failed != 0u) return;
    uint block = gl_WorkGroupID.x;
    uint row = gl_WorkGroupID.y;
    uint lid = gl_LocalInvocationID.x;

    if (row >= pc.n_tiles_y) {
        return;
    }

    uint dim_x = pc.n_tiles_x;
    uint blocks_per_row = (dim_x + WG_SIZE - 1u) / WG_SIZE;

    uint x = block * WG_SIZE + lid;

    if (x >= dim_x || block >= blocks_per_row) {
        return;
    }

    uint idx = row * dim_x + x;
    uint block_idx = row * blocks_per_row + block;

    T block_offset = subgroup_tmp[block_idx];

#ifdef PREFIX_OUTPUT
    tile_events[idx] = op_apply(block_offset, tile_events[idx]);
#else
    tile_events[idx] = op_apply(block_offset, tile_events[idx]);
#endif
}

#undef T
#undef WG_SIZE
