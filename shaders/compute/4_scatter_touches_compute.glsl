#version 450
#extension GL_GOOGLE_include_directive : require

#include "../../shared/pc.glsl"

layout(local_size_x = 256) in;

struct TileTouchRecord {
    uint seg_id;
    uint pack;
};

const uint TILE_EDGE_VALID       = 1u;
const uint TILE_EDGE_NEGATIVE    = 2u;
const uint TILE_EDGE_YEDGE_POS   = 4u;
const uint TILE_EDGE_YEDGE_NEG   = 8u;

const float TILE_EDGE_COORD_SCALE = 256.0;
const uint TILE_EDGE_COORD_MAX = 65535u;

const float EPS = 1e-6;
const float TILE_BOUNDARY_EPS = 1e-5;

layout(set = 0, binding = SEGMENTS_BINDING, std430) readonly buffer Segments {
    Segment segments[];
};

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 1, std430) readonly buffer TileTouchRecords {
    TileTouchRecord tile_touch_records[];
};

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 5, std430) readonly buffer TileInfos {
    TileInfo tile_infos[];
};

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 7, std430) writeonly buffer TileEdges {
    TileEdge tile_edges[];
};

uint unpack_tile_id(uint packed_tile_rank) {
    return packed_tile_rank & 0xffffu;
}

uint unpack_rank(uint packed_tile_rank) {
    return packed_tile_rank >> 16u;
}

bool clip_test(float p, float q, inout float t0, inout float t1) {
    const float CLIP_EPS = 1e-8;

    if (abs(p) < CLIP_EPS) {
        return q >= 0.0;
    }

    float r = q / p;

    if (p < 0.0) {
        if (r > t1) return false;
        if (r > t0) t0 = r;
    } else {
        if (r < t0) return false;
        if (r < t1) t1 = r;
    }

    return true;
}

bool clip_segment_to_rect(
    inout vec2 p0,
    inout vec2 p1,
    vec2 rect_mn,
    vec2 rect_mx
) {
    vec2 d = p1 - p0;

    float t0 = 0.0;
    float t1 = 1.0;

    if (!clip_test(-d.x, p0.x - rect_mn.x, t0, t1)) return false;
    if (!clip_test( d.x, rect_mx.x - p0.x, t0, t1)) return false;
    if (!clip_test(-d.y, p0.y - rect_mn.y, t0, t1)) return false;
    if (!clip_test( d.y, rect_mx.y - p0.y, t0, t1)) return false;

    vec2 old_p0 = p0;

    p0 = old_p0 + d * t0;
    p1 = old_p0 + d * t1;

    return true;
}

uint pack_coord(float v) {
    float max_coord = float(TILE_EDGE_COORD_MAX) / TILE_EDGE_COORD_SCALE;
    uint q = uint(round(clamp(v, 0.0, max_coord) * TILE_EDGE_COORD_SCALE));
    return min(q, TILE_EDGE_COORD_MAX);
}

uint pack_point(vec2 p) {
    uint x = pack_coord(p.x);
    uint y = pack_coord(p.y);
    return x | (y << 16u);
}

TileEdge make_invalid_edge() {
    TileEdge e;
    e.p0 = 0u;
    e.p1 = 0u;
    e.meta = 0u;
    e.y_edge = pack_coord(float(pc.tile_size));
    return e;
}

bool is_integerish(float v) {
    return abs(v - round(v)) < TILE_BOUNDARY_EPS;
}

float compute_y_edge_for_left_boundary(
    vec2 seg_p0,
    vec2 seg_p1,
    vec2 tile_mn,
    vec2 tile_mx
) {
    vec2 d = seg_p1 - seg_p0;

    float ts = tile_mx.y - tile_mn.y;
    float sentinel = ts;

    if (abs(d.x) <= EPS) {
        return sentinel;
    }

    float t = (tile_mn.x - seg_p0.x) / d.x;

    // Half-open segment ownership.
    if (t < 0.0 || t >= 1.0) {
        return sentinel;
    }

    float y = seg_p0.y + t * d.y;

    if (y <= tile_mn.y + TILE_BOUNDARY_EPS ||
        y >= tile_mx.y - TILE_BOUNDARY_EPS) {
        return sentinel;
    }

    return y - tile_mn.y;
}

void main() {
    uint touch_idx = gl_GlobalInvocationID.x;

    if (touch_idx >= bump.n_touches) {
        return;
    }

    TileTouchRecord touch = tile_touch_records[touch_idx];

    uint tile_id = unpack_tile_id(touch.pack);
    uint rank = unpack_rank(touch.pack);

    if (tile_id >= pc.n_tiles) {
        atomicOr(bump.failed, FAIL_SCATTER_OOB);
        return;
    }

    TileInfo info = tile_infos[tile_id];

    if (rank >= info.count) {
        atomicOr(bump.failed, FAIL_RANK_OOB);
        return;
    }

    uint dst = info.base + rank;

    if (dst >= pc.max_tile_storage) {
        atomicOr(bump.failed, FAIL_SCATTER_OOB);
        return;
    }

    if (touch.seg_id >= pc.n_segments) {
        atomicOr(bump.failed, FAIL_SCATTER_OOB);
        return;
    }

    uint tile_x = tile_id % pc.n_tiles_x;
    uint tile_y = tile_id / pc.n_tiles_x;

    float ts = float(pc.tile_size);

    vec2 tile_mn = vec2(float(tile_x), float(tile_y)) * ts;
    vec2 tile_mx = tile_mn + vec2(ts);

    Segment seg = segments[touch.seg_id];

    vec2 p0 = seg.p0;
    vec2 p1 = seg.p1;

    TileEdge edge = make_invalid_edge();

    if (!clip_segment_to_rect(p0, p1, tile_mn, tile_mx)) {
        tile_edges[dst] = edge;
        return;
    }

    vec2 clipped_d = p1 - p0;

    if (dot(clipped_d, clipped_d) < EPS * EPS) {
        tile_edges[dst] = edge;
        return;
    }

    vec2 local_p0 = p0 - tile_mn;
    vec2 local_p1 = p1 - tile_mn;

    local_p0 = clamp(local_p0, vec2(0.0), vec2(ts));
    local_p1 = clamp(local_p1, vec2(0.0), vec2(ts));

    vec2 local_d = local_p1 - local_p0;

    if (dot(local_d, local_d) < EPS * EPS) {
        tile_edges[dst] = edge;
        return;
    }

    bool local_horizontal = abs(local_d.y) < EPS;
    bool local_grid_aligned =
        is_integerish(local_p0.y) &&
        is_integerish(local_p1.y);

    vec2 original_d = seg.p1 - seg.p0;

    uint meta = TILE_EDGE_VALID;

    if (abs(original_d.y) >= EPS) {
        int winding_delta = original_d.y > 0.0 ? -1 : 1;

        if (winding_delta < 0) {
            meta |= TILE_EDGE_NEGATIVE;
        }
    }
    float y_edge = compute_y_edge_for_left_boundary(
        seg.p0,
        seg.p1,
        tile_mn,
        tile_mx
    );

    bool has_y_edge = y_edge < ts;

    if (has_y_edge) {
        if (original_d.x > EPS) {
            meta |= TILE_EDGE_YEDGE_POS;
        } else if (original_d.x < -EPS) {
            meta |= TILE_EDGE_YEDGE_NEG;
        }
    }
    if (local_horizontal && local_grid_aligned && !has_y_edge) {
        tile_edges[dst] = edge;
        return;
    }

    edge.p0 = pack_point(local_p0);
    edge.p1 = pack_point(local_p1);
    edge.y_edge = pack_coord(y_edge);
    edge.meta = meta;

    tile_edges[dst] = edge;
}
