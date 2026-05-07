#version 450
#extension GL_GOOGLE_include_directive : require
#include "../../shared/pc.glsl"

layout(local_size_x = 256) in;

struct Segment {
    vec2 p0;
    vec2 p1;
};

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

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 2, std430) writeonly buffer TileEvents {
    int tile_events[];
};

void visit_tile(ivec2 tile, uint seg_id, uint local_idx) {

    uint tile_id = uint(tile.y) * pc.n_tiles_x + uint(tile.x);

    // get a unique local slot inside this tile for 
    // the touch record
    uint rank = atomicAdd(tile_segments[tile_id], 1u);

    // get a unique global slot for the touch record  
    uint touch_idx = atomicAdd(bump.n_touches , 1u);

    const uint max_touches = pc.n_tiles_x * pc.n_tiles_y * AVG_TOUCHES_PER_TILE; 
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

    // write the touch
    tile_touch_records[touch_idx].seg_id = seg_id;
    tile_touch_records[touch_idx].pack = (rank << 16u) | (tile_id & 0xffffu); 
}

bool clip_test(float p, float q, inout float t0, inout float t1) {
    const float EPS = 1e-8;

    if (abs(p) < EPS) {
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

// Liang-Barsky clip against 
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

    // If the point lies exactly on a grid boundary and the segment travels
    // negatively, ownership should move to the tile on the negative side.
    const float EPS = 1e-5;

    if (dir.x < 0.0 && abs(tc.x - ftc.x) < EPS) {
        t.x -= 1;
    }

    if (dir.y < 0.0 && abs(tc.y - ftc.y) < EPS) {
        t.y -= 1;
    }

    return t;
}

void emit_events_for_segment_tiles(vec2 p0, vec2 p1) {
    float x0 = p0.x, y0 = p0.y;
    float x1 = p1.x, y1 = p1.y;

    // horizontal segments don't contribute
    if(y0 == y1) return;

    int delta = y1 > y0 ? -1 : 1;
    
    if (y0 > y1) {
        float tmp = y0;
        y0 = y1; 
        y1 = tmp;

        tmp = x0; 
        x0 = x1; 
        x1 = tmp;
    }

    const float EPS = 1e-5;
    float dy = y1 - y0;
    // TODO: probably remove
    if(dy < EPS) return;

    float slope = (x1 - x0) / dy;

    float y_min = max(y0, 0.0);
    float y_max = min(y1, float(pc.screen_h));

    // TODO: probably remove
    if(y_min >= y_max) return;

    float ts = float(pc.tile_size);

    uint tile_y_min = uint(clamp(
            int(floor(y_min / ts)),
            0,
            pc.n_tiles_y - 1
            ));

    uint tile_y_max = uint(clamp(
            int(floor((y_max - EPS) / ts)),
            0,
            pc.n_tiles_y - 1
            ));

    for (uint ty = tile_y_min; ty <= tile_y_max; ty++) {
        float tile_y_start = float(ty) * ts;

        float event_y = max(tile_y_start, y0);

        // the global x value at which the segment crosses this tile 
        float x = x0 + (event_y - y0) * slope;

        // if crossing is left of the screen, it affects the row from tile 0.
        // if it is not, the event tile is the first tile to the right of the 
        // tile that the crossing occured in (thus + 1). 
        uint event_tile_x = (x < 0.0) ? 0 : int(floor(x / ts)) + 1;

        if(event_tile_x < pc.n_tiles_x) {
            uint event_tile_id = uint(ty) * pc.n_tiles_x + uint(event_tile_x);
            atomicAdd(tile_events[event_tile_id], delta);
        } 
    }
}

void walk_segment_global_tiles(vec2 p0_in, vec2 p1_in, uint seg_id) {
    vec2 p0 = p0_in;
    vec2 p1 = p1_in;

    vec2 screen_mn = vec2(0.0);
    vec2 screen_mx = vec2(float(pc.screen_w), float(pc.screen_h));

    emit_events_for_segment_tiles(p0_in, p1_in);

    // This modifies p0 and p1 so that the segment lies entirely inside the 
    // screen rectangle. We don't want to process the parts of the segment 
    // that are not inside the render target.
    if (!clip_segment_to_rect(p0, p1, screen_mn, screen_mx)) {
        return;
    }

    vec2 d = p1 - p0;

    // If after clipping, the segment is entirely gone, it is not part 
    // of our tile grid.
    const float EPS = 1e-5;
    if (dot(d, d) < EPS * EPS) {
        return;
    }

    float ts = float(pc.tile_size);
    float inv_ts = 1.0 / ts;

    ivec2 tile = tile_of_point_for_direction(p0, d, inv_ts);
    ivec2 end_tile = tile_of_point_for_direction(p1, d, inv_ts);

    ivec2 tile_mn = ivec2(0);
    ivec2 tile_mx = ivec2(int(pc.n_tiles_x) - 1, int(pc.n_tiles_y) - 1);

    // Clamp to screen tile region 
    tile = clamp(tile, tile_mn, tile_mx);
    end_tile = clamp(end_tile, tile_mn, tile_mx);

    uint local_tile_idx = 0u;

    // count the first tile. The segment starts inside 
    // 'tile', so we count it immediately.
    visit_tile(tile, seg_id, local_tile_idx);

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
            visit_tile(ivec2(x, tile.y), seg_id, local_tile_idx);
        }
        return;
    }
    if (step_x == 0) {
        for (int y = tile.y + step_y; y != end_tile.y + step_y; y += step_y) {
            local_tile_idx++;
            visit_tile(ivec2(tile.x, y), seg_id, local_tile_idx);
        }
        return;
    }

    if (step_x != 0) {
        float next_boundary_x =
            step_x > 0
            ? float(tile.x + 1) * ts
            : float(tile.x) * ts;

        float inv_dx = 1.0 / d.x;
        t_max_x = (next_boundary_x - p0.x) * inv_dx;
        t_delta_x = ts * abs(inv_dx);
    }

    if (step_y != 0) {
        float next_boundary_y =
            step_y > 0
            ? float(tile.y + 1) * ts
            : float(tile.y) * ts;

        float inv_dy = 1.0 / d.y;
        t_max_y = (next_boundary_y - p0.y) * inv_dy;
        t_delta_y = ts * abs(inv_dy);
    }

    // A clipped segment inside the screen can cross at most
    // (pc.n_tiles_x - 1) + (pc.n_tiles_y - 1) tile boundaries.
    uint max_iters =
        (pc.n_tiles_x > 0u ? pc.n_tiles_x - 1u : 0u) +
        (pc.n_tiles_y > 0u ? pc.n_tiles_y - 1u : 0u);

    for (uint iter = 0u; iter < max_iters; iter++) {
        if (tile == end_tile) {
            break;
        }

        if (t_max_x < t_max_y) {
            // The segment crosses into the tile left or right.
            tile.x += step_x;
            t_max_x += t_delta_x;
        } else if (t_max_y < t_max_x) {
            // The segment crosses into the tile above or below.
            tile.y += step_y;
            t_max_y += t_delta_y;
        } else {
            // Exact corner crossing, visit both dimensions once.
            tile.x += step_x;
            tile.y += step_y;
            t_max_x += t_delta_x;
            t_max_y += t_delta_y;
        }

        if (tile.x < tile_mn.x || tile.y < tile_mn.y ||
                tile.x > tile_mx.x  || tile.y > tile_mx.y) {
            break;
        }

        local_tile_idx++;
        visit_tile(tile, seg_id, local_tile_idx);
    }
}

void main() {
    uint seg_id = gl_GlobalInvocationID.x;

    if (seg_id >= pc.n_segments) {
        return;
    }

    Segment seg = segments[seg_id];

    walk_segment_global_tiles(seg.p0, seg.p1, seg_id);
}
