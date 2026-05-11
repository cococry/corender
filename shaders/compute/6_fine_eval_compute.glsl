#version 450
#extension GL_GOOGLE_include_directive : require

#include "../../shared/pc.glsl"

layout(local_size_x = 4, local_size_y = 16, local_size_z = 1) in;

const uint FINE_TILE_SIZE = 16u;
const uint PIXELS_PER_THREAD = 4u;

const uint WORKGROUP_SIZE = 64u;
const uint EDGE_CHUNK_SIZE = WORKGROUP_SIZE;

const float EPS = 1e-6;
const float FINE_TILE_SIZE_F = 16.0;

const bool USE_EVEN_ODD_FILL = true;

const vec4 CLEAR_COLOR = vec4(0.0, 0.0, 0.0, 1.0);
const vec4 FILL_COLOR  = vec4(1.0, 1.0, 1.0, 1.0);

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 5, std430) readonly buffer TileInfos {
    TileInfo tile_infos[];
};

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 6, std430) readonly buffer ActiveTiles {
    uint active_tiles[];
};

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 7, std430) readonly buffer TileEdges {
    TileEdge tile_edges[];
};

layout(set = 0, binding = IMG_BINDING, rgba8) uniform writeonly image2D img;

shared TileEdge sh_edges[EDGE_CHUNK_SIZE];

uint packed_x(uint packed_point) {
    return packed_point & 0xffffu;
}

uint packed_y(uint packed_point) {
    return packed_point >> 16u;
}

float unpack_coord(uint q) {
    return float(q) * (float(pc.tile_size) / 65535.0);
}

vec2 unpack_point(uint packed_point) {
    return vec2(
        unpack_coord(packed_x(packed_point)),
        unpack_coord(packed_y(packed_point))
    );
}

float get_left_y_edge_from_packed(uint packed_p0, uint packed_p1) {
    float ts = float(pc.tile_size);

    uint p0_x = packed_x(packed_p0);
    uint p0_y = packed_y(packed_p0);

    uint p1_x = packed_x(packed_p1);
    uint p1_y = packed_y(packed_p1);

    bool p0_left = p0_x == 0u;
    bool p1_left = p1_x == 0u;

    // Scatter preserves this exact signal before grid-x nudging:
    // p0.x == 0 && p0.y != 0 -> left_y_edge = p0.y
    // p1.x == 0 && p1.y != 0 -> left_y_edge = p1.y
    // If both are left or neither is left, no left_y_edge correction.

    if (p0_left && !p1_left && p0_y != 0u) {
        return unpack_coord(p0_y);
    }

    if (p1_left && !p0_left && p1_y != 0u) {
        return unpack_coord(p1_y);
    }

    return ts;
}

bool is_grid_aligned_x(float x) {
    return x == floor(x) && x != 0.0;
}

void apply_grid_x_nudge(inout vec2 p0, inout vec2 p1) {
    if (is_grid_aligned_x(p0.x)) {
        p0.x -= EPS;
    }

    if (is_grid_aligned_x(p1.x)) {
        p1.x -= EPS;
    }
}

float resolve_even_odd(float a) {
    return abs(a - 2.0 * round(0.5 * a));
}

float resolve_nonzero(float a) {
    return min(abs(a), 1.0);
}

float resolve_coverage(float a) {
    float coverage = USE_EVEN_ODD_FILL
        ? resolve_even_odd(a)
        : resolve_nonzero(a);

    return clamp(coverage, 0.0, 1.0);
}

bool edge_can_affect_row(
    vec2 p0,
    vec2 p1,
    float left_y_edge,
    vec2 delta,
    float row_y
) {
    float ymin = min(p0.y, p1.y);
    float ymax = max(p0.y, p1.y);

    // is the row contained in the area 
    // integral of the edge?
    bool area_possible =
        abs(delta.y) > EPS &&
        row_y < ymax &&
        row_y + 1.0 > ymin;
   
    // y edge contribution is only possible
    // when clamp(xy.y - left_y_edge + 1.0, 0.0, 1.0);
    // is nonzero. this becomes non-zero xy.y > left_y_edge - 1.0.
    // since xy.y here is row_y, we do row_y > left_y_edge - 1.0.
    // if left_y_edge >= FINE_TILE_SIZE_F, there is no real 
    // y edge and thus no edge possible
    bool left_y_edge_possible =
        left_y_edge < FINE_TILE_SIZE_F &&
        row_y > left_y_edge - 1.0;

    return area_possible || left_y_edge_possible;
}

void add_left_y_edge_contribution(
    float left_y_edge,
    vec2 delta,
    vec2 xy,
    inout float area[PIXELS_PER_THREAD]
) {
    float left_y_edge_contrib =
        sign(delta.x) *
        clamp(xy.y - left_y_edge + 1.0, 0.0, 1.0);

    if (left_y_edge_contrib != 0.0) {
        area[0] += left_y_edge_contrib;
        area[1] += left_y_edge_contrib;
        area[2] += left_y_edge_contrib;
        area[3] += left_y_edge_contrib;
    }
}

void accumulate_edge(
    vec2 p0,
    vec2 p1,
    float left_y_edge,
    vec2 xy,
    inout float area[PIXELS_PER_THREAD]
) {
    vec2 delta = p1 - p0;

    // zero length edge
    if (dot(delta, delta) < EPS * EPS) {
        return;
    }

    float row_y = xy.y;

    if (!edge_can_affect_row(p0, p1, left_y_edge, delta, row_y)) {
        return;
    }

    // Horizontal edges still contribue through left_y_edge 
    if (abs(delta.y) <= EPS) {
        add_left_y_edge_contribution(left_y_edge, delta, xy, area);
        return;
    }

    float y = p0.y - xy.y;

    float y0 = clamp(y, 0.0, 1.0);
    float y1 = clamp(y + delta.y, 0.0, 1.0);

    float dy = y0 - y1;

    if (dy != 0.0) {
        float inv_dy = 1.0 / delta.y;

        float t0 = (y0 - y) * inv_dy;
        float t1 = (y1 - y) * inv_dy;

        float startx = p0.x - xy.x;

        float x0 = startx + t0 * delta.x;
        float x1 = startx + t1 * delta.x;

        float xmin0 = min(x0, x1);
        float xmax0 = max(x0, x1);

        {
            float xmin = min(xmin0 - 0.0, 1.0) - 1.0e-6;
            float xmax = xmax0 - 0.0;

            float b = min(xmax, 1.0);
            float c = max(b, 0.0);
            float d = max(xmin, 0.0);

            float a = (
                b +
                0.5 * (d * d - c * c) -
                xmin
            ) / (xmax - xmin);

            area[0] += a * dy;
        }

        {
            float xmin = min(xmin0 - 1.0, 1.0) - 1.0e-6;
            float xmax = xmax0 - 1.0;

            float b = min(xmax, 1.0);
            float c = max(b, 0.0);
            float d = max(xmin, 0.0);

            float a = (
                b +
                0.5 * (d * d - c * c) -
                xmin
            ) / (xmax - xmin);

            area[1] += a * dy;
        }

        {
            float xmin = min(xmin0 - 2.0, 1.0) - 1.0e-6;
            float xmax = xmax0 - 2.0;

            float b = min(xmax, 1.0);
            float c = max(b, 0.0);
            float d = max(xmin, 0.0);

            float a = (
                b +
                0.5 * (d * d - c * c) -
                xmin
            ) / (xmax - xmin);

            area[2] += a * dy;
        }

        {
            float xmin = min(xmin0 - 3.0, 1.0) - 1.0e-6;
            float xmax = xmax0 - 3.0;

            float b = min(xmax, 1.0);
            float c = max(b, 0.0);
            float d = max(xmin, 0.0);

            float a = (
                b +
                0.5 * (d * d - c * c) -
                xmin
            ) / (xmax - xmin);

            area[3] += a * dy;
        }
    }

    add_left_y_edge_contribution(left_y_edge, delta, xy, area);
}

void store_pixel_checked(ivec2 p, vec4 rgba) {
    if (
        p.x >= 0 &&
        p.y >= 0 &&
        uint(p.x) < pc.screen_w &&
        uint(p.y) < pc.screen_h
    ) {
        imageStore(img, p, rgba);
    }
}

void shade_solid_tile(uint tile_x, uint tile_y, uvec2 local_base) {
    ivec2 pixel = ivec2(
        int(tile_x * FINE_TILE_SIZE + local_base.x),
        int(tile_y * FINE_TILE_SIZE + local_base.y)
    );

    store_pixel_checked(pixel + ivec2(0, 0), FILL_COLOR);
    store_pixel_checked(pixel + ivec2(1, 0), FILL_COLOR);
    store_pixel_checked(pixel + ivec2(2, 0), FILL_COLOR);
    store_pixel_checked(pixel + ivec2(3, 0), FILL_COLOR);
}

void write_fine_pixels(
    uint tile_x,
    uint tile_y,
    uvec2 local_base,
    float area[PIXELS_PER_THREAD]
) {
    ivec2 pixel = ivec2(
        int(tile_x * FINE_TILE_SIZE + local_base.x),
        int(tile_y * FINE_TILE_SIZE + local_base.y)
    );

    float c0 = resolve_coverage(area[0]);
    float c1 = resolve_coverage(area[1]);
    float c2 = resolve_coverage(area[2]);
    float c3 = resolve_coverage(area[3]);

    vec4 color0 = vec4(mix(CLEAR_COLOR.rgb, FILL_COLOR.rgb, c0), 1.0);
    vec4 color1 = vec4(mix(CLEAR_COLOR.rgb, FILL_COLOR.rgb, c1), 1.0);
    vec4 color2 = vec4(mix(CLEAR_COLOR.rgb, FILL_COLOR.rgb, c2), 1.0);
    vec4 color3 = vec4(mix(CLEAR_COLOR.rgb, FILL_COLOR.rgb, c3), 1.0);

    store_pixel_checked(pixel + ivec2(0, 0), color0);
    store_pixel_checked(pixel + ivec2(1, 0), color1);
    store_pixel_checked(pixel + ivec2(2, 0), color2);
    store_pixel_checked(pixel + ivec2(3, 0), color3);
}

void shade_fine_tile(
    TileInfo info,
    uint tile_x,
    uint tile_y,
    uvec2 local_base
) {
    vec2 local_xy = vec2(local_base);

    float backdrop = float(info.winding);

    float area[PIXELS_PER_THREAD];
    area[0] = backdrop;
    area[1] = backdrop;
    area[2] = backdrop;
    area[3] = backdrop;

    uint lane = gl_LocalInvocationIndex;

    for (uint chunk_base = 0u; chunk_base < info.count; chunk_base += EDGE_CHUNK_SIZE) {
      // phase one: load edges into shared memory
        uint remaining = info.count - chunk_base;
        uint chunk_count = min(EDGE_CHUNK_SIZE, remaining);

        if (lane < chunk_count) {
            sh_edges[lane] = tile_edges[info.base + chunk_base + lane];
        }

        barrier();

        // phase two: all invocations process the shared chunk
        for (uint edge_ix = 0u; edge_ix < chunk_count; edge_ix++) {
            TileEdge edge = sh_edges[edge_ix];

            float left_y_edge = get_left_y_edge_from_packed(edge.p0, edge.p1);

            vec2 p0 = unpack_point(edge.p0);
            vec2 p1 = unpack_point(edge.p1);

            apply_grid_x_nudge(p0, p1);

            accumulate_edge(p0, p1, left_y_edge, local_xy, area);
        }

        barrier();
    }

    write_fine_pixels(tile_x, tile_y, local_base, area);
}

// dispatched over all active tiles 
// each workgroup handles one active tile, 
// Each workgroup contains 4(x) x 16(y) = 64 invocations.
void main() {
    uint active_ix = gl_WorkGroupID.x;

    if (active_ix >= bump.n_active_tiles) {
        return;
    }

    uint tile_id = active_tiles[active_ix];

    if (tile_id >= pc.n_tiles) {
        atomicOr(bump.failed, FAIL_FINE_OOB);
        return;
    }

    TileInfo info = tile_infos[tile_id];

    uint tile_x = tile_id % pc.n_tiles_x;
    uint tile_y = tile_id / pc.n_tiles_x;

    uvec2 local_base = uvec2(
        gl_LocalInvocationID.x * PIXELS_PER_THREAD, 
        gl_LocalInvocationID.y);

    if ((info.flags & TILE_SOLID) != 0u) {
        shade_solid_tile(tile_x, tile_y, local_base);
        return;
    }

    if ((info.flags & TILE_FINE) != 0u) {
        shade_fine_tile(info, tile_x, tile_y, local_base);
        return;
    }
}
