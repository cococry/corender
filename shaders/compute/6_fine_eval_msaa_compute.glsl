#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require

#include "../../shared/pc.glsl"

layout(local_size_x = 4, local_size_y = 16, local_size_z = 1) in;

const uint FINE_TILE_SIZE = 16u;
const uint PIXELS_PER_THREAD = 4u;

const uint WORKGROUP_SIZE = 64u;
const uint EDGE_CHUNK_SIZE = WORKGROUP_SIZE;

const uint ENFORCED_SUBGROUP_SIZE = 32u;
const uint N_SUBGROUPS = WORKGROUP_SIZE / ENFORCED_SUBGROUP_SIZE;

const uint MSAA_SAMPLE_COUNT = 16u;
const uint MSAA_FULL_MASK = 0xffffu;

const float EPS = 1e-6;
const float FINE_TILE_SIZE_F = 16.0;

const float ONE_MINUS_ULP = 0.99999994;
const float ROBUST_EPSILON = 2e-7;

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

layout(set = 0, binding = MSAA_LUT_BINDING, std430) readonly buffer MsaaMaskLut {
  uint msaa_mask_lut[];
};

layout(set = 0, binding = IMG_BINDING, rgba8) uniform writeonly image2D img;

const uint MASK_LUT_DIM = 256u;
const float MASK_LUT_X_MIN = -1.0;
const float MASK_LUT_X_MAX =  2.0;
const float MASK_LUT_X_SCALE =
float(MASK_LUT_DIM) / (MASK_LUT_X_MAX - MASK_LUT_X_MIN);

uint quantize_mask_lut_x(float x) {
  float q = floor((x - MASK_LUT_X_MIN) * MASK_LUT_X_SCALE);
  return uint(clamp(q, 0.0, float(MASK_LUT_DIM - 1u)));
}

uint lookup_msaa16_cell_mask(float x_top, float x_bottom) {
  uint top_ix = quantize_mask_lut_x(x_top);
  uint bottom_ix = quantize_mask_lut_x(x_bottom);

  uint entry_ix = top_ix * MASK_LUT_DIM + bottom_ix;

  uint word = msaa_mask_lut[entry_ix >> 1u];
  uint shift = (entry_ix & 1u) * 16u;

  return (word >> shift) & MSAA_FULL_MASK;
}

shared TileEdge sh_edges[EDGE_CHUNK_SIZE];

shared uint sh_count[WORKGROUP_SIZE];
shared uint sh_subgroup_sums[N_SUBGROUPS];

// One bit per row of this tile.
// Bit y means row parity toggles starting at row y
shared uint sh_winding_y;

// One 16-bit event mask per row
shared uint sh_winding_x[FINE_TILE_SIZE];

// One 8-bit MSAA mask per pixel
shared uint sh_samples[FINE_TILE_SIZE * FINE_TILE_SIZE];

shared uint sh_scan_y;
shared uint sh_scan_x[FINE_TILE_SIZE];

struct MsaaLineWalkParams {
  vec2 xy0;
  vec2 xy1;

  float x_step_rate;
  float x_step_start_offset;
  float x_sign;
  float slope_x_per_y;
  float x0;
  float y0;

  float dx;
  float dy;

  uint count_x;
  uint count;
};

shared MsaaLineWalkParams sh_walk[EDGE_CHUNK_SIZE];

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

uint pixel_ix(uint x, uint y) {
  return y * FINE_TILE_SIZE + x;
}

uint span(float a, float b) {
  a = clamp(a, 0.0, FINE_TILE_SIZE_F);
  b = clamp(b, 0.0, FINE_TILE_SIZE_F);

  return uint(max(ceil(max(a, b)) - floor(min(a, b)), 1.0));
}

uint prefix_even_odd_16(uint events) {
  events ^= events << 1u;
  events ^= events << 2u;
  events ^= events << 4u;
  events ^= events << 8u;

  return events & 0xffffu;
}

bool is_zero_length(vec2 p0, vec2 p1) {
  vec2 delta = p1 - p0;
  return dot(delta, delta) < EPS * EPS;
}

bool is_horizontal_grid_line(vec2 p0, vec2 p1) {
  return abs(p0.y - p1.y) <= EPS &&
    abs(p0.y - floor(p0.y)) <= EPS;
}

uint n_pixel_cells_for_segment(vec2 p0, vec2 p1) {
  if (is_zero_length(p0, p1)) {
    return 0u;
  }

  if (is_horizontal_grid_line(p0, p1)) {
    return 0u;
  }

  return span(p0.x, p1.x) + span(p0.y, p1.y) - 1u;
}

bool tile_is_fully_inside_screen(uint tile_x, uint tile_y) {
  return
    (tile_x + 1u) * FINE_TILE_SIZE <= pc.screen_w &&
    (tile_y + 1u) * FINE_TILE_SIZE <= pc.screen_h;
}

void clear_msaa_shared(uint lane) {
  if (lane == 0u) {
    sh_winding_y = 0u;
  }

  for (uint i = lane; i < FINE_TILE_SIZE; i += WORKGROUP_SIZE) {
    sh_winding_x[i] = 0u;
  }

  for (uint i = lane; i < FINE_TILE_SIZE * FINE_TILE_SIZE; i += WORKGROUP_SIZE) {
    sh_samples[i] = 0u;
  }

  barrier();
}

void emit_left_y_edge_event(float left_y_edge) {
  if (left_y_edge <= EPS || left_y_edge >= FINE_TILE_SIZE_F) {
    return;
  }

  uint row = uint(ceil(left_y_edge));

  if (row < FINE_TILE_SIZE) {
    atomicXor(sh_winding_y, 1u << row);
  }
}

void emit_x_winding_event(uint x, uint y) {
  if (x >= FINE_TILE_SIZE - 1u || y >= FINE_TILE_SIZE) {
    return;
  }

  atomicXor(sh_winding_x[y], 2u << x);
}

uint compute_cell_sample_mask_lut(
    MsaaLineWalkParams lp,
    uint px,
    uint py
    ) {
  float pixel_top = float(py);
  float pixel_left = float(px);

  float x_top =
    lp.xy0.x +
    lp.slope_x_per_y * (pixel_top - lp.xy0.y) -
    pixel_left;

  float x_bottom = x_top + lp.slope_x_per_y;

  return lookup_msaa16_cell_mask(x_top, x_bottom);
}

bool make_msaa_line_walk_params(
    vec2 p0_in,
    vec2 p1_in,
    out MsaaLineWalkParams lp
    ) {
  // degenerate segments cannot touch any pixel cell
  if (is_zero_length(p0_in, p1_in)) {
    return false;
  }

  bool is_down = p1_in.y >= p0_in.y;

  lp.xy0 = is_down ? p0_in : p1_in;
  lp.xy1 = is_down ? p1_in : p0_in;

  vec2 d = lp.xy1 - lp.xy0;

  lp.dx = abs(d.x);
  lp.dy = d.y;

  float dx_plus_dy = lp.dx + lp.dy;

  // segment has no useful length => invalid
  if (dx_plus_dy <= EPS) {
    return false;
  }

  // horizontal grid-line segments do not contribute pixel-cell work
  if (is_horizontal_grid_line(lp.xy0, lp.xy1)) {
    return false;
  }

  // horizontal or nearly-horizontal segments do not produce MSAA cell coverage here
  if (lp.dy <= EPS) {
    return false;
  }

  float inverse_dx_plus_dy = 1.0 / dx_plus_dy;

  // how many vertical pixel boundaries does this segment cross
  lp.count_x = span(lp.xy0.x, lp.xy1.x) - 1u;

  // total traversed pixel cells for this segment
  lp.count = lp.count_x + span(lp.xy0.y, lp.xy1.y);

  // what fraction of the walk steps are x-steps
  lp.x_step_rate = lp.dx * inverse_dx_plus_dy;

  bool x_moves_right = lp.xy1.x >= lp.xy0.x;
  lp.x_sign = x_moves_right ? 1.0 : -1.0;

  // pixel-local x position, measured in the walk direction
  float pixel_local_x = fract(lp.xy0.x * lp.x_sign);

  lp.y0 = floor(lp.xy0.y);

  // next horizontal pixel boundary below xy0
  float ytop = lp.y0 + 1.0;

  // y distance to the next horizontal pixel boundary
  float y_to_next_boundary = ytop - lp.xy0.y;

  // starting offset of the edge for the line stepper
  lp.x_step_start_offset = min(
      (lp.dy * pixel_local_x + lp.dx * y_to_next_boundary) *
      inverse_dx_plus_dy,
      ONE_MINUS_ULP
      );

  // expected number of x-steps by the final walked cell
  float num_x_steps =
    floor(
        lp.x_step_rate * (float(lp.count) - 1.0) +
        lp.x_step_start_offset
        );

  // detect whether floating-point error would produce the wrong x-step count
  float x_step_err = num_x_steps - float(lp.count_x);

  // nudge the step rate away from the numerically wrong side
  if (x_step_err != 0.0) {
    lp.x_step_rate -= ROBUST_EPSILON * sign(x_step_err);
  }

  // starting pixel x-column, accounting for x walk direction
  lp.x0 =
    floor(lp.xy0.x * lp.x_sign) * lp.x_sign +
    (x_moves_right ? 0.0 : -1.0);

  // precompute x movement per y step for the MSAA coverage lookup
  lp.slope_x_per_y = d.x / lp.dy;

  return true;
}

void accumulate_msaa_cell(
    MsaaLineWalkParams lp,
    uint local_tile
    ) {
  // number of x-steps taken before this local pixel cell
  float n_x_steps =
    floor(lp.x_step_rate * float(local_tile) + lp.x_step_start_offset);

  // recover the pixel cell touched by this local walk step
  int x = int(lp.x0 + lp.x_sign * n_x_steps);

  // remaining steps are y-steps
  int y = int(lp.y0 + float(local_tile) - n_x_steps);

  // ignore cells outside this fine tile
  if (
      x < 0 ||
      y < 0 ||
      x >= int(FINE_TILE_SIZE) ||
      y >= int(FINE_TILE_SIZE)
     ) {
    return;
  }

  bool changed_pixel_row;

  if (local_tile == 0u) {
    changed_pixel_row = abs(lp.y0 - lp.xy0.y) <= EPS;
  } else {
    float prev_n_x_steps =
      floor(lp.x_step_rate * float(local_tile - 1u) + lp.x_step_start_offset);

    changed_pixel_row = n_x_steps == prev_n_x_steps;
  }

  // emit parity event only if row changed
  if (changed_pixel_row) {
    emit_x_winding_event(uint(x), uint(y));
  }

  // compute this edge's 16-sample coverage mask for the pixel cell
  uint mask = compute_cell_sample_mask_lut(
      lp,
      uint(x),
      uint(y)
      );

  // xor edge coverage into the shared per-pixel MSAA mask
  if (mask != 0u) {
    atomicXor(sh_samples[pixel_ix(uint(x), uint(y))], mask);
  }
}

void prefix_counts(uint lane, uint count) {
  uint subgroup_prefix = subgroupInclusiveAdd(count);

  uint subgroup_total =
    subgroupBroadcast(subgroup_prefix, ENFORCED_SUBGROUP_SIZE - 1u);

  if (gl_SubgroupInvocationID == 0u) {
    sh_subgroup_sums[gl_SubgroupID] = subgroup_total;
  }

  barrier();

  uint subgroup_base = gl_SubgroupID == 0u ? 0u : sh_subgroup_sums[0];

  sh_count[lane] = subgroup_base + subgroup_prefix;

  barrier();
}

uint binary_search_edge(uint goal, uint chunk_count) {
  uint lo = 0u;
  uint hi = chunk_count;

  while (hi > lo + 1u) {
    uint mid = (lo + hi) >> 1u;

    if (goal >= sh_count[mid - 1u]) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  return lo;
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

vec4 coverage_color(float c) {
  return vec4(mix(CLEAR_COLOR.rgb, FILL_COLOR.rgb, c), 1.0);
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

void prepare_msaa_scan_tables(uint lane) {
  if (lane == 0u) {
    sh_scan_y = prefix_even_odd_16(sh_winding_y);
  }

  if (lane < FINE_TILE_SIZE) {
    sh_scan_x[lane] = prefix_even_odd_16(sh_winding_x[lane]);
  }

  barrier();
}

void write_one_msaa_pixel(
    TileInfo info,
    ivec2 pixel,
    uint x,
    uint y,
    uint scan_y,
    uint scan_x,
    bool tile_fully_inside
    ) {
  uint row_parity =
    ((scan_y >> y) & 1u) ^
    (uint(info.winding) & 1u);

  uint pixel_parity = (scan_x >> x) & 1u;
  uint base_parity = row_parity ^ pixel_parity;
  uint base_mask = base_parity != 0u ? MSAA_FULL_MASK : 0u;

  uint mask = (sh_samples[pixel_ix(x, y)] ^ base_mask) & MSAA_FULL_MASK;
  float c = float(bitCount(mask)) * (1.0 / float(MSAA_SAMPLE_COUNT));

  vec4 color = coverage_color(c);
  store_pixel_maybe_checked(pixel, color, tile_fully_inside);
}

void write_msaa_pixels(
    TileInfo info,
    uint tile_x,
    uint tile_y,
    uvec2 local_base,
    bool tile_fully_inside
    ) {
  uint lane = gl_LocalInvocationIndex;

  prepare_msaa_scan_tables(lane);

  ivec2 pixel = ivec2(
      int(tile_x * FINE_TILE_SIZE + local_base.x),
      int(tile_y * FINE_TILE_SIZE + local_base.y)
      );

  uint y = local_base.y;
  uint x0 = local_base.x;

  uint scan_y = sh_scan_y;
  uint scan_x = sh_scan_x[y];

  {
    uint x = x0 + 0u;

    write_one_msaa_pixel(
        info,
        pixel + ivec2(0, 0),
        x,
        y,
        scan_y,
        scan_x,
        tile_fully_inside
        );
  }

  {
    uint x = x0 + 1u;

    write_one_msaa_pixel(
        info,
        pixel + ivec2(1, 0),
        x,
        y,
        scan_y,
        scan_x,
        tile_fully_inside
        );
  }

  {
    uint x = x0 + 2u;

    write_one_msaa_pixel(
        info,
        pixel + ivec2(2, 0),
        x,
        y,
        scan_y,
        scan_x,
        tile_fully_inside
        );
  }

  {
    uint x = x0 + 3u;

    write_one_msaa_pixel(
        info,
        pixel + ivec2(3, 0),
        x,
        y,
        scan_y,
        scan_x,
        tile_fully_inside
        );
  }
}

void shade_fine_tile_msaa(
    TileInfo info,
    uint tile_x,
    uint tile_y,
    uvec2 local_base,
    bool tile_fully_inside
    ) {
  uint lane = gl_LocalInvocationIndex;

  clear_msaa_shared(lane);

  for (uint chunk_base = 0u; chunk_base < info.count; chunk_base += EDGE_CHUNK_SIZE) {
    uint remaining = info.count - chunk_base;
    uint chunk_count = min(EDGE_CHUNK_SIZE, remaining);

    // load edges into shared mem
    if (lane < chunk_count) {
      sh_edges[lane] = tile_edges[info.base + chunk_base + lane];
    }

    barrier();

    uint pixel_count = 0u;

    if (lane < chunk_count) {
      TileEdge edge = sh_edges[lane];

      float left_y_edge = get_left_y_edge_from_packed(edge.p0, edge.p1);

      vec2 p0 = unpack_point(edge.p0);
      vec2 p1 = unpack_point(edge.p1);

      apply_grid_x_nudge(p0, p1);

      MsaaLineWalkParams lp;

      if (make_msaa_line_walk_params(p0, p1, lp)) {
        sh_walk[lane] = lp;
        pixel_count = lp.count;
      } else {
        pixel_count = 0u;
      }

      // emit the parity event per edge
      emit_left_y_edge_event(left_y_edge);
    }

    prefix_counts(lane, pixel_count);

    // total number of pixel-cell jobs in this edge chunk
    uint total_pixel_cells = 0u;

    if (chunk_count > 0u) {
      total_pixel_cells = sh_count[chunk_count - 1u];
    }

    barrier();

    for (uint work_cell = lane; work_cell < total_pixel_cells; work_cell += WORKGROUP_SIZE) {
      // the index of the edge that the current work
      // pixel-cell is on
      uint edge_idx = binary_search_edge(work_cell, chunk_count);

      uint previous_cell = 0u;

      if (edge_idx > 0u) {
        previous_cell = sh_count[edge_idx - 1u];
      }

      uint local_cell = work_cell - previous_cell;

      MsaaLineWalkParams lp = sh_walk[edge_idx];

      accumulate_msaa_cell(lp, local_cell);
    }

    barrier();
  }

  write_msaa_pixels(info, tile_x, tile_y, local_base, tile_fully_inside);
}

// dispatched over all active tiles
// each workgroup handles one active tile
// each workgroup contains 4(x) x 16(y) = 64 invocations
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

  bool tile_fully_inside = tile_is_fully_inside_screen(tile_x, tile_y);

  uvec2 local_base = uvec2(
      gl_LocalInvocationID.x * PIXELS_PER_THREAD,
      gl_LocalInvocationID.y
      );

  if ((info.flags & TILE_SOLID) != 0u) {
    shade_solid_tile(tile_x, tile_y, local_base, tile_fully_inside);
    return;
  }

  if ((info.flags & TILE_FINE) != 0u) {
    shade_fine_tile_msaa(info, tile_x, tile_y, local_base, tile_fully_inside);
    return;
  }
}
