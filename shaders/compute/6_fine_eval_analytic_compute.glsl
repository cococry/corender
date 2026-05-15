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


const vec4 CLEAR_COLOR = vec4(0.0, 0.0, 0.0, 1.0);
const vec4 FILL_COLOR  = vec4(1.0, 1.0, 1.0, 1.0);

const vec4 PIXEL_OFFSETS = vec4(0.0, 1.0, 2.0, 3.0);

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 5, std430) readonly buffer TileInfos {
    TileInfo tile_infos[];
};

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 6, std430) readonly buffer ActiveTilesSparse {
    uint active_tiles_sparse[];
};

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 7, std430) readonly buffer ActiveTilesDense {
    uint active_tiles_dense[];
};

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 8, std430) readonly buffer TileEdges {
    TileEdge tile_edges[];
};

layout(set = 0, binding = IMG_BINDING, rgba8) uniform writeonly image2D img;

shared vec2 sh_p0[EDGE_CHUNK_SIZE];
shared vec2 sh_delta[EDGE_CHUNK_SIZE];

shared float sh_left_y_edge[EDGE_CHUNK_SIZE];
shared float sh_inv_delta_y[EDGE_CHUNK_SIZE];

shared uint sh_row_mask[EDGE_CHUNK_SIZE];

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

vec4 resolve_even_odd(vec4 a) {
    return abs(a - 2.0 * round(0.5 * a));
}

vec4 resolve_nonzero(vec4 a) {
    return min(abs(a), vec4(1.0));
}

vec4 resolve_coverage(vec4 a) {
    vec4 coverage = CR_FILL_RULE_NONZERO 
        ? resolve_even_odd(a)
        : resolve_nonzero(a);

    return clamp(coverage, vec4(0.0), vec4(1.0));
}

uint mask_low_bits(uint n) {
    if (n >= FINE_TILE_SIZE) {
        return 0xffffu;
    }

    return (1u << n) - 1u;
}

uint row_mask_range(uint first, uint last_exclusive) {
    first = min(first, FINE_TILE_SIZE);
    last_exclusive = min(last_exclusive, FINE_TILE_SIZE);

    if (first >= last_exclusive) {
        return 0u;
    }

    return mask_low_bits(last_exclusive) & ~mask_low_bits(first);
}

uint compute_edge_row_mask(
    vec2 p0,
    vec2 p1,
    float left_y_edge,
    vec2 delta
) {
    if (dot(delta, delta) < EPS * EPS) {
        return 0u;
    }

    uint mask = 0u;

    // is the row contained in the area 
    // integral of the edge?
    if (abs(delta.y) > EPS) {
        float ymin = clamp(min(p0.y, p1.y), 0.0, FINE_TILE_SIZE_F);
        float ymax = clamp(max(p0.y, p1.y), 0.0, FINE_TILE_SIZE_F);

        uint first = uint(clamp(floor(ymin), 0.0, FINE_TILE_SIZE_F));
        uint last_exclusive = uint(clamp(ceil(ymax), 0.0, FINE_TILE_SIZE_F));

        mask |= row_mask_range(first, last_exclusive);
    }
   
    // y edge contribution is only possible
    // when clamp(xy.y - left_y_edge + 1.0, 0.0, 1.0);
    // is nonzero. this becomes non-zero xy.y > left_y_edge - 1.0.
    // since xy.y here is row_y, we do row_y > left_y_edge - 1.0.
    // if left_y_edge >= FINE_TILE_SIZE_F, there is no real 
    // y edge and thus no edge possible
    if (left_y_edge < FINE_TILE_SIZE_F) {
        uint first = uint(clamp(floor(left_y_edge), 0.0, FINE_TILE_SIZE_F));

        mask |= row_mask_range(first, FINE_TILE_SIZE);
    }

    return mask;
}

void add_left_y_edge_contribution(
    float left_y_edge,
    vec2 delta,
    vec2 xy,
    inout vec4 area
) {
    float left_y_edge_contrib =
        sign(delta.x) *
        clamp(xy.y - left_y_edge + 1.0, 0.0, 1.0);

    if (left_y_edge_contrib != 0.0) {
        area += vec4(left_y_edge_contrib);
    }
}

void accumulate_edge(
    vec2 p0,
    vec2 delta,
    float left_y_edge,
    float inv_delta_y,
    vec2 xy,
    inout vec4 area
) {
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
        float t0 = (y0 - y) * inv_delta_y;
        float t1 = (y1 - y) * inv_delta_y;

        float startx = p0.x - xy.x;

        float x0 = startx + t0 * delta.x;
        float x1 = startx + t1 * delta.x;

        float xmin0 = min(x0, x1);
        float xmax0 = max(x0, x1);

        if (xmax0 <= 0.0) {
            area += vec4(dy);
        } else if (xmin0 < float(PIXELS_PER_THREAD)) {
            vec4 xmin = min(vec4(xmin0) - PIXEL_OFFSETS, vec4(1.0)) - vec4(1.0e-6);
            vec4 xmax = vec4(xmax0) - PIXEL_OFFSETS;

            vec4 b = min(xmax, vec4(1.0));
            vec4 c = max(b, vec4(0.0));
            vec4 d = max(xmin, vec4(0.0));

            vec4 a = (
                b +
                0.5 * (d * d - c * c) -
                xmin
            ) / (xmax - xmin);

            area += a * dy;
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

void store_pixel_unchecked(ivec2 p, vec4 rgba) {
    imageStore(img, p, rgba);
}

void store_pixel_maybe_checked(ivec2 p, vec4 rgba, bool tile_fully_inside) {
    if (tile_fully_inside) {
        store_pixel_unchecked(p, rgba);
    } else {
        store_pixel_checked(p, rgba);
    }
}

bool tile_is_fully_inside_screen(uint tile_x, uint tile_y) {
    return
        (tile_x + 1u) * FINE_TILE_SIZE <= pc.screen_w &&
        (tile_y + 1u) * FINE_TILE_SIZE <= pc.screen_h;
}

void shade_solid_tile(
    uint tile_x,
    uint tile_y,
    uvec2 local_base,
    bool tile_fully_inside
) {
    ivec2 pixel = ivec2(
        int(tile_x * FINE_TILE_SIZE + local_base.x),
        int(tile_y * FINE_TILE_SIZE + local_base.y)
    );

    store_pixel_maybe_checked(pixel + ivec2(0, 0), FILL_COLOR, tile_fully_inside);
    store_pixel_maybe_checked(pixel + ivec2(1, 0), FILL_COLOR, tile_fully_inside);
    store_pixel_maybe_checked(pixel + ivec2(2, 0), FILL_COLOR, tile_fully_inside);
    store_pixel_maybe_checked(pixel + ivec2(3, 0), FILL_COLOR, tile_fully_inside);
}

void write_fine_pixels(
    uint tile_x,
    uint tile_y,
    uvec2 local_base,
    vec4 area,
    bool tile_fully_inside
) {
    ivec2 pixel = ivec2(
        int(tile_x * FINE_TILE_SIZE + local_base.x),
        int(tile_y * FINE_TILE_SIZE + local_base.y)
    );

    vec4 c = resolve_coverage(area);

    vec4 color0 = vec4(mix(CLEAR_COLOR.rgb, FILL_COLOR.rgb, c.x), 1.0);
    vec4 color1 = vec4(mix(CLEAR_COLOR.rgb, FILL_COLOR.rgb, c.y), 1.0);
    vec4 color2 = vec4(mix(CLEAR_COLOR.rgb, FILL_COLOR.rgb, c.z), 1.0);
    vec4 color3 = vec4(mix(CLEAR_COLOR.rgb, FILL_COLOR.rgb, c.w), 1.0);

    store_pixel_maybe_checked(pixel + ivec2(0, 0), color0, tile_fully_inside);
    store_pixel_maybe_checked(pixel + ivec2(1, 0), color1, tile_fully_inside);
    store_pixel_maybe_checked(pixel + ivec2(2, 0), color2, tile_fully_inside);
    store_pixel_maybe_checked(pixel + ivec2(3, 0), color3, tile_fully_inside);
}

void decode_edge_for_chunk(
    uint lane,
    uint edge_base
) {
    TileEdge edge = tile_edges[edge_base + lane];

    float left_y_edge = get_left_y_edge_from_packed(edge.p0, edge.p1);

    vec2 p0 = unpack_point(edge.p0);
    vec2 p1 = unpack_point(edge.p1);

    apply_grid_x_nudge(p0, p1);

    vec2 delta = p1 - p0;

    sh_p0[lane] = p0;
    sh_delta[lane] = delta;
    sh_left_y_edge[lane] = left_y_edge;
    sh_inv_delta_y[lane] = abs(delta.y) > EPS ? 1.0 / delta.y : 0.0;
    sh_row_mask[lane] = compute_edge_row_mask(p0, p1, left_y_edge, delta);
}

void shade_fine_tile(
    TileInfo info,
    uint tile_x,
    uint tile_y,
    uvec2 local_base,
    bool tile_fully_inside
) {
    vec2 local_xy = vec2(local_base);

    float backdrop = float(info.winding);

    vec4 area = vec4(backdrop);

    uint lane = gl_LocalInvocationIndex;
    uint row_bit = 1u << local_base.y;

    for (uint chunk_base = 0u; chunk_base < info.count; chunk_base += EDGE_CHUNK_SIZE) {
      // phase one: load edges into shared memory
        uint remaining = info.count - chunk_base;
        uint chunk_count = min(EDGE_CHUNK_SIZE, remaining);

        if (lane < chunk_count) {
            decode_edge_for_chunk(lane, info.base + chunk_base);
        }

        barrier();

        // phase two: all invocations process the shared chunk
        for (uint edge_ix = 0u; edge_ix < chunk_count; edge_ix++) {
            if ((sh_row_mask[edge_ix] & row_bit) == 0u) {
                continue;
            }

            accumulate_edge(
                sh_p0[edge_ix],
                sh_delta[edge_ix],
                sh_left_y_edge[edge_ix],
                sh_inv_delta_y[edge_ix],
                local_xy,
                area
            );
        }

        barrier();
    }

    write_fine_pixels(tile_x, tile_y, local_base, area, tile_fully_inside);
}

uint active_tile_id_from_split_lists(uint active_ix) {
    uint n_sparse = bump.n_active_tiles_sparse;

    if (active_ix < n_sparse) {
        return active_tiles_sparse[active_ix];
    }

    return active_tiles_dense[active_ix - n_sparse];
}

// dispatched over all active tiles 
// each workgroup handles one active tile, 
// Each workgroup contains 4(x) x 16(y) = 64 invocations.
void main() {
    uint active_ix = gl_WorkGroupID.x;

    uint n_active =
        bump.n_active_tiles_sparse +
        bump.n_active_tiles_dense;

    if (active_ix >= n_active) {
        return;
    }

    uint tile_id = active_tile_id_from_split_lists(active_ix);

    if (tile_id >= pc.n_tiles) {
        atomicOr(bump.failed, FAIL_FINE_OOB);
        return;
    }

    TileInfo info = tile_infos[tile_id];

    uint tile_x = tile_id % pc.n_tiles_x;
    uint tile_y = tile_id / pc.n_tiles_x;

    bool tile_fully_inside = tile_is_fully_inside_screen(tile_x, tile_y);

    uvec2 local_base = uvec2(
        gl_LocalInvocationID.x * PIXELS_PER_THREAD, 
        gl_LocalInvocationID.y);

    if ((info.flags & TILE_SOLID) != 0u) {
        shade_solid_tile(tile_x, tile_y, local_base, tile_fully_inside);
        return;
    }

    if ((info.flags & TILE_FINE) != 0u) {
        shade_fine_tile(info, tile_x, tile_y, local_base, tile_fully_inside);
        return;
    }
}
