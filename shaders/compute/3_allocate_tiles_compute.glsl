#version 450
#extension GL_GOOGLE_include_directive : require
#include "../../shared/pc.glsl"

layout(local_size_x = 256) in;

// COMP_PIPELINE_BINDING_BASE + 4 is subgroup_tmp, thus + 5
layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 4, std430) buffer TileSegments {
    uint tile_segments[];
};

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 5, std430) buffer ActiveTiles {
    uint active_tiles[];
};


void main() {
    if (bump.failed != 0u) return;

    uint tile_id = gl_GlobalInvocationID.x;

    if(tile_id >= pc.n_tiles_x * pc.n_tiles_y) return;

    // tile_segments here holds the number of segments 
    // for each tile. this was produced by walk_segments.
    uint n_segments = tile_segments[tile_id];

    if(n_segments == 0u) {
        tile_segments[tile_id] = 0u;
        return;
    }
    
    uint base = atomicAdd(bump.n_tile_segment_slots, n_segments);

    const uint max_touches = pc.n_tiles_x * pc.n_tiles_y * AVG_TOUCHES_PER_TILE; 
    if (base + n_segments > max_touches) {
        atomicOr(bump.failed, FAIL_TILE_SEGMENT_OVERFLOW);
        return;
    }

    tile_segments[tile_id] = base;

    uint active_idx = atomicAdd(bump.n_active_tiles, 1u);

    active_tiles[active_idx] = tile_id;
}
