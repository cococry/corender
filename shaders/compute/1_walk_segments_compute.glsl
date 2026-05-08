#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_debug_printf : enable

#include "../../shared/pc.glsl"

layout(local_size_x = 256) in;

struct TileTouchRecord {
    uint seg_id;
    uint pack;
};

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

const float EPS = 1e-6;
const float TILE_BOUNDARY_EPS = 1e-6;

void visit_tile(ivec2 tile, uint seg_id, uint local_idx, uint max_touches) {
    if (
        tile.x < 0 ||
        tile.y < 0 ||
        tile.x >= int(pc.n_tiles_x) ||
        tile.y >= int(pc.n_tiles_y)
    ) {
        atomicOr(bump.failed, FAIL_BAD_TOUCH_TILE);
        return;
    }

    uint tile_id = uint(tile.y) * pc.n_tiles_x + uint(tile.x);

    if (tile_id >= pc.n_tiles) {
        atomicOr(bump.failed, FAIL_BAD_TOUCH_TILE);
        return;
    }

    uint rank = atomicAdd(tile_segments[tile_id], 1u);
    uint touch_idx = atomicAdd(bump.n_touches, 1u);

    if (touch_idx >= max_touches) {
        atomicOr(bump.failed, FAIL_TOUCH_OVERFLOW);
        return;
    }

    if (rank >= 65536u) {
        atomicOr(bump.failed, FAIL_RANK_OVERFLOW);
        return;
    }

    if (tile_id >= 65536u) {
        atomicOr(bump.failed, FAIL_TILE_ID_OVERFLOW);
        return;
    }

    tile_touch_records[touch_idx].seg_id = seg_id;
    tile_touch_records[touch_idx].pack = (rank << 16u) | (tile_id & 0xffffu);
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

ivec2 tile_of_point_for_direction(vec2 p, vec2 dir, float inv_tile_size) {
    vec2 tc = p * inv_tile_size;
    vec2 ftc = floor(tc);
    ivec2 t = ivec2(ftc);

    if (dir.x < 0.0 && abs(tc.x - ftc.x) < TILE_BOUNDARY_EPS) {
        t.x -= 1;
    }

    if (dir.y < 0.0 && abs(tc.y - ftc.y) < TILE_BOUNDARY_EPS) {
        t.y -= 1;
    }

    return t;
}

bool tile_y_boundary_row(float y, float inv_ts, out int row) {
    float ty = y * inv_ts;
    float nearest = round(ty);

    if (abs(ty - nearest) > TILE_BOUNDARY_EPS) {
        row = 0;
        return false;
    }

    row = int(nearest);
    return true;
}

int first_tile_strictly_right_of_crossing_float(float x, float inv_ts) {
    float tx = x * inv_ts;
    float nearest = round(tx);

    if (abs(tx - nearest) < TILE_BOUNDARY_EPS) {
        tx = nearest;
    }

    return max(int(floor(tx)) + 1, 0);
}

void emit_top_edge_event_at_tile(uint event_y, uint event_x, vec2 d) {
    if (event_y >= pc.n_tiles_y) {
        return;
    }

    if (event_x >= pc.n_tiles_x) {
        return;
    }

    if (abs(d.y) <= EPS) {
        return;
    }

    int delta = d.y > 0.0 ? -1 : 1;

    uint event_tile_id = event_y * pc.n_tiles_x + event_x;
    atomicAdd(tile_events[event_tile_id], delta);
}

void emit_top_edge_event_for_y_crossing(
    ivec2 from_tile,
    ivec2 to_tile,
    vec2 p0,
    vec2 d,
    float t_cross,
    bool crossed_x_too,
    float inv_ts
) {
    if (from_tile.y == to_tile.y) {
        return;
    }

    int event_y = max(from_tile.y, to_tile.y);

    if (event_y < 0 || event_y >= int(pc.n_tiles_y)) {
        return;
    }

    float x_cross = p0.x + d.x * t_cross;

    int event_x = first_tile_strictly_right_of_crossing_float(
        x_cross,
        inv_ts
    );

    if (event_x < 0 || event_x >= int(pc.n_tiles_x)) {
        return;
    }

    emit_top_edge_event_at_tile(uint(event_y), uint(event_x), d);
}

void emit_endpoint_top_edge_event(vec2 p, vec2 d, float inv_ts) {
    int row = 0;

    if (!tile_y_boundary_row(p.y, inv_ts, row)) {
        return;
    }

    if (row < 0 || row >= int(pc.n_tiles_y)) {
        return;
    }

    int event_x = first_tile_strictly_right_of_crossing_float(p.x, inv_ts);

    if (event_x < 0 || event_x >= int(pc.n_tiles_x)) {
        return;
    }

    emit_top_edge_event_at_tile(uint(row), uint(event_x), d);
}

void walk_segment_global_tiles(vec2 p0_in, vec2 p1_in, uint seg_id, uint max_touches) {
    vec2 p0 = p0_in;
    vec2 p1 = p1_in;

    float ts = float(pc.tile_size);
    float inv_ts = 1.0 / ts;

    vec2 screen_mn = vec2(0.0);
    vec2 screen_mx = vec2(float(pc.screen_w), float(pc.screen_h));

    if (!clip_segment_to_rect(p0, p1, screen_mn, screen_mx)) {
        return;
    }

    vec2 d = p1 - p0;

    if (dot(d, d) < EPS * EPS) {
        return;
    }

    ivec2 tile = tile_of_point_for_direction(p0, d, inv_ts);
    ivec2 end_tile = tile_of_point_for_direction(p1, -d, inv_ts);

    ivec2 tile_mn = ivec2(0);
    ivec2 tile_mx = ivec2(int(pc.n_tiles_x) - 1, int(pc.n_tiles_y) - 1);

    tile = clamp(tile, tile_mn, tile_mx);
    end_tile = clamp(end_tile, tile_mn, tile_mx);

    if (d.y > EPS) {
        emit_endpoint_top_edge_event(p0, d, inv_ts);
    } else if (d.y < -EPS) {
        emit_endpoint_top_edge_event(p1, d, inv_ts);
    }

    uint local_tile_idx = 0u;

    visit_tile(tile, seg_id, local_tile_idx, max_touches);

    if (tile == end_tile) {
        return;
    }

    int step_x = d.x > 0.0 ? 1 : (d.x < 0.0 ? -1 : 0);
    int step_y = d.y > 0.0 ? 1 : (d.y < 0.0 ? -1 : 0);

    const float INF = 3.402823466e+38;
    float t_max_x = INF;
    float t_max_y = INF;
    float t_delta_x = INF;
    float t_delta_y = INF;

    if (step_y == 0) {
        for (int x = tile.x + step_x; x != end_tile.x + step_x; x += step_x) {
            local_tile_idx++;
            visit_tile(ivec2(x, tile.y), seg_id, local_tile_idx, max_touches);
        }
        return;
    }

    if (step_x == 0) {
        for (int y = tile.y + step_y; y != end_tile.y + step_y; y += step_y) {
            ivec2 prev_tile = ivec2(tile.x, y - step_y);
            ivec2 next_tile = ivec2(tile.x, y);

            float boundary_y =
                step_y > 0
                ? float(y) * ts
                : float(y + 1) * ts;

            float t_cross = (boundary_y - p0.y) / d.y;

            if (t_cross > EPS && t_cross < 1.0 - EPS) {
                emit_top_edge_event_for_y_crossing(
                    prev_tile,
                    next_tile,
                    p0,
                    d,
                    t_cross,
                    false,
                    inv_ts
                );
            }

            local_tile_idx++;
            visit_tile(next_tile, seg_id, local_tile_idx, max_touches);
        }

        return;
    }

    float next_boundary_x =
        step_x > 0
        ? float(tile.x + 1) * ts
        : float(tile.x) * ts;

    float inv_dx = 1.0 / d.x;
    t_max_x = (next_boundary_x - p0.x) * inv_dx;
    t_delta_x = ts * abs(inv_dx);

    float next_boundary_y =
        step_y > 0
        ? float(tile.y + 1) * ts
        : float(tile.y) * ts;

    float inv_dy = 1.0 / d.y;
    t_max_y = (next_boundary_y - p0.y) * inv_dy;
    t_delta_y = ts * abs(inv_dy);

    uint max_iters =
        (pc.n_tiles_x > 0u ? pc.n_tiles_x - 1u : 0u) +
        (pc.n_tiles_y > 0u ? pc.n_tiles_y - 1u : 0u);

    for (uint iter = 0u; iter < max_iters; iter++) {
        if (tile == end_tile) {
            break;
        }

        ivec2 prev_tile = tile;

        if (t_max_x < t_max_y) {
            tile.x += step_x;
            t_max_x += t_delta_x;
        } else if (t_max_y < t_max_x) {
            float t_cross = t_max_y;

            tile.y += step_y;
            t_max_y += t_delta_y;

            if (t_cross > EPS && t_cross < 1.0 - EPS) {
                emit_top_edge_event_for_y_crossing(
                    prev_tile,
                    tile,
                    p0,
                    d,
                    t_cross,
                    false,
                    inv_ts
                );
            }
        } else {
            float t_cross = t_max_y;

            tile.x += step_x;
            tile.y += step_y;

            t_max_x += t_delta_x;
            t_max_y += t_delta_y;

            if (t_cross > EPS && t_cross < 1.0 - EPS) {
                emit_top_edge_event_for_y_crossing(
                    prev_tile,
                    tile,
                    p0,
                    d,
                    t_cross,
                    true,
                    inv_ts
                );
            }
        }

        if (
            tile.x < tile_mn.x ||
            tile.y < tile_mn.y ||
            tile.x > tile_mx.x ||
            tile.y > tile_mx.y
        ) {
            break;
        }

        local_tile_idx++;
        visit_tile(tile, seg_id, local_tile_idx, max_touches);
    }
}

void main() {
    uint seg_id = gl_GlobalInvocationID.x;

    if (seg_id >= pc.n_segments) {
        return;
    }

    Segment seg = segments[seg_id];

    walk_segment_global_tiles(seg.p0, seg.p1, seg_id, pc.max_tile_storage);
}
