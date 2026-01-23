#version 450

layout(local_size_x = 8) in;

struct Segment {
    vec2 p0;
    vec2 p1;
};

struct TileHeader {
    uint n_segments;
};

layout(set = 0, binding = 0, std430) readonly buffer Segments {
    Segment segments[];
};

layout(set = 0, binding = 1, std430) buffer TileHeaders {
    TileHeader tile_headers[];
};

layout(set = 0, binding = 2, std430) buffer TileSegmentIndices {
    uint tile_segment_indices[];
};

layout(push_constant) uniform push_constant {
    uint screen_w,  screen_h;
    uint n_tiles_x, n_tiles_y;
    uint tile_size;

    uint n_segments;
    uint n_paths;

    uint fill_rule;
} pc;

const uint MAX_SEGMENTS_PER_TILE = 32;

/* Segment vs axis-aligned rectangle intersection */
bool segmentIntersectsAABB(vec2 p0, vec2 p1, vec2 bmin, vec2 bmax)
{
    vec2 d = p1 - p0;

    // Handle degenerate segments
    if (all(lessThan(abs(d), vec2(1e-6))))
        return all(greaterThanEqual(p0, bmin)) &&
               all(lessThanEqual(p0, bmax));

    vec2 inv_d = 1.0 / d;

    vec2 t0 = (bmin - p0) * inv_d;
    vec2 t1 = (bmax - p0) * inv_d;

    vec2 tmin = min(t0, t1);
    vec2 tmax = max(t0, t1);

    float t_enter = max(tmin.x, tmin.y);
    float t_exit  = min(tmax.x, tmax.y);

    return t_enter <= t_exit && t_exit >= 0.0 && t_enter <= 1.0;
}

void main()
{
    uint segment_id = gl_GlobalInvocationID.x;
    if (segment_id >= pc.n_segments)
        return;

    Segment seg = segments[segment_id];

    // Segment AABB (coarse tile range)
    vec2 seg_min = min(seg.p0, seg.p1);
    vec2 seg_max = max(seg.p0, seg.p1);

    ivec2 tile_min = ivec2(floor(seg_min / float(pc.tile_size)));
    ivec2 tile_max = ivec2(floor(seg_max / float(pc.tile_size)));

    tile_min = clamp(tile_min,
                     ivec2(0),
                     ivec2(int(pc.n_tiles_x - 1), int(pc.n_tiles_y - 1)));

    tile_max = clamp(tile_max,
                     ivec2(0),
                     ivec2(int(pc.n_tiles_x - 1), int(pc.n_tiles_y - 1)));

    for (int ty = tile_min.y; ty <= tile_max.y; ty++) {
        for (int tx = tile_min.x; tx <= tile_max.x; tx++) {

            vec2 tile_min_px = vec2(tx, ty) * float(pc.tile_size);
            vec2 tile_max_px = tile_min_px + vec2(pc.tile_size);

            // Exact intersection test
            if (!segmentIntersectsAABB(seg.p0, seg.p1,
                                       tile_min_px, tile_max_px))
                continue;

            uint tile_idx = uint(ty) * pc.n_tiles_x + uint(tx);

            uint write_idx =
                atomicAdd(tile_headers[tile_idx].n_segments, 1);

            if (write_idx < MAX_SEGMENTS_PER_TILE) {
                tile_segment_indices[
                    tile_idx * MAX_SEGMENTS_PER_TILE + write_idx
                ] = segment_id;
            }
        }
    }
}

