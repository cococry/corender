#version 450
#extension GL_GOOGLE_include_directive : require

#include "../../shared/pc.glsl"

layout(local_size_x = 4, local_size_y = 16, local_size_z = 1) in;

const uint FINE_TILE_SIZE = 16u;
const uint PIXELS_PER_THREAD = 4u;

const uint TILE_EDGE_VALID       = 1u;
const uint TILE_EDGE_NEGATIVE    = 2u;
const uint TILE_EDGE_YEDGE_POS   = 4u;
const uint TILE_EDGE_YEDGE_NEG   = 8u;

const float TILE_EDGE_COORD_SCALE = 256.0;
const float EPS = 1e-6;

const vec4 CLEAR_COLOR = vec4(0.0, 0.0, 0.0, 1.0);
const vec4 FILL_COLOR  = vec4(1.0, 1.0, 1.0, 1.0);
const vec4 TILE_BORDER_COLOR = vec4(1.0, 0.0, 0.0, 1.0);
const vec4 TILE_BORDER_COLOR_SOLID = vec4(0.0, 1.0, 0.0, 1.0);

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

float unpack_edge_coord(uint q) {
    return float(q & 0xffffu) / TILE_EDGE_COORD_SCALE;
}

vec2 unpack_edge_point(uint p) {
    return vec2(
        unpack_edge_coord(p),
        unpack_edge_coord(p >> 16u)
    );
}

bool edge_is_valid(TileEdge e) {
    return (e.meta & TILE_EDGE_VALID) != 0u;
}

bool approx_eq(float a, float b) {
    return abs(a - b) < EPS;
}

float even_odd_coverage(float winding_area) {
    if (isnan(winding_area) || isinf(winding_area)) {
        return 0.0;
    }

    float m = mod(abs(winding_area), 2.0);
    return clamp(min(m, 2.0 - m), 0.0, 1.0);
}

float row_overlap(float a, float b, float py) {
    float mn = min(a, b);
    float mx = max(a, b);

    float lo = max(mn, py);
    float hi = min(mx, py + 1.0);

    return clamp(hi - lo, 0.0, 1.0);
}

void accumulate_edge_even_odd(
    vec2 p0,
    vec2 p1,
    vec2 xy,
    inout float area[PIXELS_PER_THREAD],
    TileEdge edge
) {
    vec2 delta = p1 - p0;

    bool on_left_boundary =
        approx_eq(p0.x, 0.0) &&
        approx_eq(p1.x, 0.0) &&
        abs(delta.y) > EPS;

    if (on_left_boundary) {
        float row_toggle = row_overlap(p0.y, p1.y, xy.y);

        if (row_toggle > 0.0) {
            for (uint i = 0u; i < PIXELS_PER_THREAD; i++) {
                area[i] += row_toggle;
            }
        }

        return;
    }

    if (abs(delta.y) > EPS) {
        float y = p0.y - xy.y;

        float y0 = clamp(y, 0.0, 1.0);
        float y1 = clamp(y + delta.y, 0.0, 1.0);

        float dy = y0 - y1;

        if (abs(dy) > EPS) {
            float inv_dy = 1.0 / delta.y;

            float t0 = (y0 - y) * inv_dy;
            float t1 = (y1 - y) * inv_dy;

            float startx = p0.x - xy.x;

            float x0 = startx + t0 * delta.x;
            float x1 = startx + t1 * delta.x;

            float xmin0 = min(x0, x1);
            float xmax0 = max(x0, x1);

            float width = xmax0 - xmin0;

            for (uint i = 0u; i < PIXELS_PER_THREAD; i++) {
                float i_f = float(i);
                float a;

                if (width < EPS) {
                    float x = xmin0 - i_f;

                    if (x <= 0.0) {
                        a = 1.0;
                    } else if (x >= 1.0) {
                        a = 0.0;
                    } else {
                        a = 1.0 - x;
                    }
                } else {
                    float xmin = min(xmin0 - i_f, 1.0) - 1.0e-6;
                    float xmax = xmax0 - i_f;

                    float b = min(xmax, 1.0);
                    float c = max(b, 0.0);
                    float d = max(xmin, 0.0);

                    a = (b + 0.5 * (d * d - c * c) - xmin) / (xmax - xmin);
                }

                area[i] += a * dy;
            }
        }
    }

    float y_edge = unpack_edge_coord(edge.y_edge);

    if (y_edge < float(FINE_TILE_SIZE)) {
        float edge_sign = 0.0;

        if ((edge.meta & TILE_EDGE_YEDGE_POS) != 0u) {
            edge_sign = 1.0;
        } else if ((edge.meta & TILE_EDGE_YEDGE_NEG) != 0u) {
            edge_sign = -1.0;
        }

        if (edge_sign != 0.0) {
            float y_edge_contrib =
                edge_sign * clamp(xy.y - y_edge + 1.0, 0.0, 1.0);

            for (uint i = 0u; i < PIXELS_PER_THREAD; i++) {
                area[i] += y_edge_contrib;
            }
        }
    }
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

bool is_tile_border_pixel(uvec2 local_base, uint i) {
    uint lx = local_base.x + i;
    uint ly = local_base.y;

    return lx == 0u ||
           ly == 0u ||
           lx == FINE_TILE_SIZE - 1u ||
           ly == FINE_TILE_SIZE - 1u;
}

vec4 apply_tile_border(vec4 color, uvec2 local_base, uint i, vec4 border) {
    if (is_tile_border_pixel(local_base, i)) {
        return border;
    }

    return color;
}

void shade_solid_tile(uint tile_x, uint tile_y, uvec2 local_base) {
    ivec2 pixel = ivec2(
        int(tile_x * FINE_TILE_SIZE + local_base.x),
        int(tile_y * FINE_TILE_SIZE + local_base.y)
    );

    for (uint i = 0u; i < PIXELS_PER_THREAD; i++) {
        vec4 color = apply_tile_border(
            FILL_COLOR,
            local_base,
            i,
            TILE_BORDER_COLOR_SOLID
        );

        store_pixel_checked(pixel + ivec2(int(i), 0), color);
    }
}

void shade_fine_tile(
    uint tile_id,
    TileInfo info,
    uint tile_x,
    uint tile_y,
    uvec2 local_base
) {
    vec2 local_xy = vec2(local_base);

    float area[PIXELS_PER_THREAD];

    float backdrop = float(info.winding & 1);

    for (uint i = 0u; i < PIXELS_PER_THREAD; i++) {
        area[i] = backdrop;
    }

    for (uint edge_ix = 0u; edge_ix < info.count; edge_ix++) {
        TileEdge edge = tile_edges[info.base + edge_ix];

        if (!edge_is_valid(edge)) {
            continue;
        }

        vec2 p0 = unpack_edge_point(edge.p0);
        vec2 p1 = unpack_edge_point(edge.p1);

        p0 = clamp(p0, vec2(0.0), vec2(float(FINE_TILE_SIZE)));
        p1 = clamp(p1, vec2(0.0), vec2(float(FINE_TILE_SIZE)));

        vec2 d = p1 - p0;

        if (dot(d, d) < EPS * EPS) {
            continue;
        }

        accumulate_edge_even_odd(p0, p1, local_xy, area, edge);
    }

    ivec2 pixel = ivec2(
        int(tile_x * FINE_TILE_SIZE + local_base.x),
        int(tile_y * FINE_TILE_SIZE + local_base.y)
    );

    for (uint i = 0u; i < PIXELS_PER_THREAD; i++) {
        float coverage = even_odd_coverage(area[i]);

        if (isnan(coverage) || isinf(coverage)) {
            store_pixel_checked(
                pixel + ivec2(int(i), 0),
                vec4(1.0, 0.0, 1.0, 1.0)
            );
            continue;
        }

        coverage = clamp(coverage, 0.0, 1.0);

        vec3 rgb = mix(CLEAR_COLOR.rgb, FILL_COLOR.rgb, coverage);
        vec4 color = apply_tile_border(
            vec4(rgb, 1.0),
            local_base,
            i,
            TILE_BORDER_COLOR
        );

        store_pixel_checked(
            pixel + ivec2(int(i), 0),
            color
        );
    }
}

void main() {
    uint active_ix = gl_WorkGroupID.x;

    if (active_ix >= bump.n_active_tiles) {
        return;
    }

    if (pc.tile_size != FINE_TILE_SIZE) {
        atomicOr(bump.failed, FAIL_FINE_TILE_SIZE_MISMATCH);
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
        gl_LocalInvocationID.y
    );

    if ((info.flags & TILE_SOLID) != 0u) {
        shade_solid_tile(tile_x, tile_y, local_base);
        return;
    }

    if ((info.flags & TILE_FINE) != 0u) {
        shade_fine_tile(tile_id, info, tile_x, tile_y, local_base);
        return;
    }
}
