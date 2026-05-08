#version 450
#extension GL_GOOGLE_include_directive : require

#include "../../shared/pc.glsl"

layout(local_size_x = 256) in;

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 2, std430) readonly buffer TileEvents {
    int tile_events[];
};

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 4, std430) readonly buffer TileSegments {
    uint tile_n_segments[];
};

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 5, std430) writeonly buffer TileInfos {
    TileInfo tile_infos[];
};

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 6, std430) writeonly buffer ActiveTiles {
    uint active_tiles[];
};

void main() {
    uint tile_id = gl_GlobalInvocationID.x;

    if (tile_id >= pc.n_tiles) {
        return;
    }

    uint n_segments = tile_n_segments[tile_id];

    int winding = tile_events[tile_id];

    bool has_edges = n_segments != 0u;

    uint flags = TILE_EMPTY;

    if (has_edges) {
        flags |= TILE_FINE;
    } else if ((winding & 1) != 0) {
        flags |= TILE_SOLID;
    }

    uint base = 0u;

    if (has_edges) {
        base = atomicAdd(bump.n_tile_segment_slots, n_segments);

        if (base + n_segments > pc.max_tile_storage) {
            atomicOr(bump.failed, FAIL_TILE_SEGMENT_OVERFLOW);
            return;
        }
    }

    tile_infos[tile_id].base = base;
    tile_infos[tile_id].count = n_segments;
    tile_infos[tile_id].winding = winding;
    tile_infos[tile_id].flags = flags;

    if (flags != TILE_EMPTY) {
        uint active_idx = atomicAdd(bump.n_active_tiles, 1u);

        if (active_idx >= pc.n_tiles) {
            atomicOr(bump.failed, FAIL_ACTIVE_TILE_OVERFLOW);
            return;
        }

        active_tiles[active_idx] = tile_id;
    }
}
