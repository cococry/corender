#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_ballot : require

#include "../../shared/pc.glsl"
#include "../../shared/segment_walking.glsl"

layout(local_size_x = 256) in;

layout(set = 0, binding = SEGMENTS_BINDING, std430) readonly buffer Segments {
    Segment segments[];
};

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 4, std430) buffer TileSegments {
    uint tile_segments[];
};

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 1, std430) writeonly buffer TileTouchRecords {
    TileTouchRecord tile_touch_records[];
};

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 2, std430) buffer TileEvents {
    int tile_events[];
};

bool in_tile_bounds(ivec2 tile) {
    return
        tile.x >= 0 &&
        tile.y >= 0 &&
        tile.x < int(pc.n_tiles_x) &&
        tile.y < int(pc.n_tiles_y);
}


uint reserve_touch_idx() {
    /*
        Subgroup-batched global touch allocation.

        Every active lane calling visit_tile participates. This turns:
            one atomicAdd(bump.n_touches) per touch
        into approximately:
            one atomicAdd per active subgroup batch
    */
    uvec4 mask = subgroupBallot(true);
    uint count = subgroupBallotBitCount(mask);
    uint off = subgroupBallotExclusiveBitCount(mask);

    uint base = 0u;

    if (subgroupElect()) {
        base = atomicAdd(bump.n_touches, count);
    }

    base = subgroupBroadcastFirst(base);

    return base + off;
}


TileTouchRecord pack_touch_record(
    uint seg_id,
    uint tile_id,
    uint rank,
    uint local_tile
) {
    TileTouchRecord r;

    r.a = 0u;
    r.b = 0u;

    // enable GPU stats is used more as a 
    // debug/sanitizing flag here
#if CR_ENABLE_GPU_STATS
    if (seg_id > TOUCH_SEG_MASK) {
        atomicOr(bump.failed, FAIL_TOUCH_OVERFLOW);
        return r;
    }

    if (tile_id > TOUCH_TILE_MASK) {
        atomicOr(bump.failed, FAIL_TILE_ID_OVERFLOW);
        return r;
    }

    if (local_tile > TOUCH_LOCAL_TILE_MASK) {
        atomicOr(bump.failed, FAIL_TOUCH_OVERFLOW);
        return r;
    }
#endif
    
    if (rank > TOUCH_RANK_MASK) {
        atomicOr(bump.failed, FAIL_RANK_OVERFLOW);
        return r;
    }

    r.a =
        (seg_id & TOUCH_SEG_MASK) |
        ((rank & TOUCH_RANK_MASK) << TOUCH_SEG_BITS);

    r.b =
        (tile_id & TOUCH_TILE_MASK) |
        ((local_tile & TOUCH_LOCAL_TILE_MASK) << TOUCH_TILE_BITS);

    return r;
}

uint touch_seg_id(TileTouchRecord r) {
    return r.a & TOUCH_SEG_MASK;
}

uint touch_rank(TileTouchRecord r) {
    return (r.a >> TOUCH_SEG_BITS) & TOUCH_RANK_MASK;
}

uint touch_tile_id(TileTouchRecord r) {
    return r.b & TOUCH_TILE_MASK;
}

uint touch_local_tile(TileTouchRecord r) {
    return (r.b >> TOUCH_TILE_BITS) & TOUCH_LOCAL_TILE_MASK;
}

void visit_tile(ivec2 tile, uint seg_id, uint local_tile) {

    uint tile_id = uint(tile.y) * pc.n_tiles_x + uint(tile.x);

#if CR_ENABLE_GPU_STATS
    if (tile_id >= pc.n_tiles) {
        atomicOr(bump.failed, FAIL_BAD_TOUCH_TILE);
        return;
    }
#endif

    // most heavily contented atomic... grr atomics
    uint rank = atomicAdd(tile_segments[tile_id], 1u);  

    uint touch_idx = reserve_touch_idx();

    if (touch_idx >= pc.max_tile_storage) {
        atomicOr(bump.failed, FAIL_TOUCH_OVERFLOW);
        return;
    }
    
    TileTouchRecord rec = pack_touch_record(seg_id, tile_id, rank, local_tile);

    tile_touch_records[touch_idx] = rec;
}

void emit_backdrop_delta(int y, int x, int delta) {
    if (y < 0 || y >= int(pc.n_tiles_y)) {
        return;
    }

    if (x < 0) {
        x = 0;
        // tiles left of the screen still need to 
        // emit backdrop events, no return
    }

    if (x >= int(pc.n_tiles_x)) {
        return;
    }

    uint tile_id = uint(y) * pc.n_tiles_x + uint(x);

    atomicAdd(tile_events[tile_id], delta);
}

bool starts_new_row(LineWalkParams lp, uint local_tile, float current_x_steps, float last_n_x_steps) {
    // x did not change between previous tile and current tile
    return
        (local_tile == 0u)
        ? (lp.y0 == lp.s0.y)
        : (last_n_x_steps == current_x_steps);
}

void walk_segment_tiles(LineWalkParams lp, uint seg_id) {
    float last_n_x_steps = floor(lp.x_step_rate * -1.0 + lp.x_step_start_offset);

    for (uint local_tile = 0u; local_tile < lp.count; local_tile++) {
        float n_x_steps;
        ivec2 tile = tile_for_local_tile(lp, local_tile, n_x_steps);

        bool changed_row = starts_new_row(lp, local_tile, n_x_steps, last_n_x_steps);

        // only emit backdrop events when we crossed a
        // tile-row boundary
        if (changed_row) {
            emit_backdrop_delta(tile.y, tile.x + 1, lp.delta);
        }

        // visit every tile this segment overlaps and 
        // insert touch records.
        if (in_tile_bounds(tile)) {
            visit_tile(tile, seg_id, local_tile);
        }

        last_n_x_steps = n_x_steps;
    }
}

void walk_segment(vec2 p0, vec2 p1, uint seg_id) {
    if (pc.n_tiles_x == 0u || pc.n_tiles_y == 0u) {
        return;
    }

    LineWalkParams lp;

    if (!make_line_walk_params(p0, p1, lp)) {
        return;
    }

    float xmin = min(lp.s0.x, lp.s1.x);
    float xmax = max(lp.s0.x, lp.s1.x);

    if (
            lp.s0.y >= float(pc.n_tiles_y) ||
            lp.s1.y <= 0.0 ||
            xmin >= float(pc.n_tiles_x)
       ) {
        return;
    }

    walk_segment_tiles(lp, seg_id);
}

// dispatched over all segments globally. each 
// workgroup handles 256 segments.
void main() {
    uint seg_id = gl_GlobalInvocationID.x;

    if (seg_id >= pc.n_segments) {
        return;
    }

    Segment seg = segments[seg_id];

    walk_segment(seg.p0, seg.p1, seg_id);
}


