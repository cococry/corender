#version 450
#extension GL_GOOGLE_include_directive : require
#include "../../shared/pc.glsl"

/* #undef PREFIX_OUTPUT */
/* #undef DO_XOR */
#define PREFIX_INCLUSIVE
#define PREFIX_SIGNED

#extension GL_KHR_shader_subgroup_basic      : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

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
#define SG_SIZE 4
#define SGS_PER_WG (WG_SIZE / SG_SIZE)

#ifdef PREFIX_OUTPUT
layout(set = 0, binding = 7, std430) buffer PrefixOutput {
    T tile_events[];
};
#endif

layout(set = 0, binding = 7, std430) buffer PrefixInput {
    T tile_events[];
};

layout(set = 0, binding = SUBGROUP_TMP_BINDING, std430) buffer SubgroupTmp {
    T subgroup_tmp[];
};

shared T subgroup_totals[SGS_PER_WG];

T op_identity() {
#ifdef DO_XOR
    return 0u;
#else
    return T(0);
#endif
}

T op_apply(T a, T b) {
#ifdef DO_XOR
    return a ^ b;
#else
    return a + b;
#endif
}

T subgroup_exclusive_op(T v) {
#ifdef DO_XOR
    return subgroupExclusiveXor(v);
#else
    return subgroupExclusiveAdd(v);
#endif
}

T subgroup_reduce_op(T v) {
#ifdef DO_XOR
    return subgroupXor(v);
#else
    return subgroupAdd(v);
#endif
}

void main() {
    if (bump.failed != 0u) return;
    uint lid = gl_LocalInvocationID.x;
    uint block = gl_WorkGroupID.x;
    uint row = gl_WorkGroupID.y;

    if (row >= pc.n_tiles_y) {
        return;
    }

    uint dim_x = pc.n_tiles_x;
    uint blocks_per_row = (dim_x + WG_SIZE - 1u) / WG_SIZE;

    uint x = block * WG_SIZE + lid;
    uint gid = row * dim_x + x;

    T v = op_identity();

    if (x < dim_x) {
        v = tile_events[gid];
    }

    // Prefix inside each subgroup.
    T sg_excl = subgroup_exclusive_op(v);
    T sg_total = subgroup_reduce_op(v);

    if (subgroupElect()) {
        subgroup_totals[gl_SubgroupID] = sg_total;
    }

    barrier();

    // Prefix the subgroup totals using subgroup 0.
    if (gl_SubgroupID == 0u) {
        uint sg_lane = gl_SubgroupInvocationID;

        T val = op_identity();
        if (sg_lane < SGS_PER_WG) {
            val = subgroup_totals[sg_lane];
        }

        T off = subgroup_exclusive_op(val);

        if (sg_lane < SGS_PER_WG) {
            subgroup_totals[sg_lane] = off;
        }
    }

    barrier();

    T sg_offset = subgroup_totals[gl_SubgroupID];

    T local_excl = op_apply(sg_offset, sg_excl);
    T local_incl = op_apply(local_excl, v);

#ifdef PREFIX_INCLUSIVE
    T out_v = local_incl;
#else
    T out_v = local_excl;
#endif

#ifdef PREFIX_OUTPUT
    if (x < dim_x) {
        tile_events[gid] = out_v;
    }
#else
    if (x < dim_x) {
        tile_events[gid] = out_v;
    }
#endif

    bool is_last_active =
        (x < dim_x) &&
        ((lid == WG_SIZE - 1u) || (x + 1u == dim_x));

    if (block < blocks_per_row && is_last_active) {
        subgroup_tmp[row * blocks_per_row + block] = local_incl;
    }
}

#undef T
#undef WG_SIZE
#undef SG_SIZE
#undef SGS_PER_WG
