#version 450
#extension GL_GOOGLE_include_directive : require
#include "../../shared/pc.glsl"

layout(local_size_x = 256) in;

struct TileTouchRecord {
    uint seg_id;
    uint pack;
};

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 1, std430) readonly buffer TileTouchRecords {
    TileTouchRecord tile_touch_records[];
};

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 4, std430) buffer TileSegments {
    uint tile_segments[];
};

uint unpack_tile_id(uint packed_tile_rank) {
    return packed_tile_rank & 0xffffu;
}

uint unpack_rank(uint packed_tile_rank) {
    return packed_tile_rank >> 16u;
}

void main() {
    if (bump.failed != 0u) return;

    uint touch_idx = gl_GlobalInvocationID.x;

    if(touch_idx >= bump.n_touches) return;

    TileTouchRecord touch = tile_touch_records[touch_idx];

    uint tile_id = unpack_tile_id(touch.pack);
    uint rank = unpack_rank(touch.pack);

    // the tile_segments buffer holds the global offsets 
    // in the first section and the actual segments per 
    // tile after this section. this is why we do n_tiles + [...]
    uint n_tiles = pc.n_tiles_x * pc.n_tiles_y;
    uint dst = n_tiles + tile_segments[tile_id] + rank;


    const uint max_touches = pc.n_tiles_x * pc.n_tiles_y * AVG_TOUCHES_PER_TILE; 
    if (dst >= max_touches) {
        atomicOr(bump.failed, FAIL_SCATTER_OOB);
        return;
    }

    tile_segments[dst] = touch.seg_id;
}
