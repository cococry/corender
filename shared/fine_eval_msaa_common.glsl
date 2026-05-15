layout(local_size_x = 4, local_size_y = 16, local_size_z = 1) in;

const uint FINE_TILE_SIZE = 16u;
const uint PIXELS_PER_THREAD = 4u;

const uint WORKGROUP_SIZE = 64u;
const uint EDGE_CHUNK_SIZE = WORKGROUP_SIZE;

const uint ENFORCED_SUBGROUP_SIZE = 32u;
const uint N_SUBGROUPS = WORKGROUP_SIZE / ENFORCED_SUBGROUP_SIZE;

#ifndef FINE_MSAA_SAMPLE_COUNT
#define FINE_MSAA_SAMPLE_COUNT 16
#endif


#if FINE_MSAA_SAMPLE_COUNT == 8

layout(set = 0, binding = MSAA8_LUT_BINDING, std430) readonly buffer MsaaMaskLut {
  uint msaa_mask_lut[];
};

const uint MSAA_SAMPLE_COUNT = 8u;
const uint MSAA_FULL_MASK = 0xffu;
const uint MASK_LUT_DIM = 128u;

#if CR_FILL_RULE_NONZERO
#define MSAA_NZ_WORDS_PER_PIXEL 4u
#endif

#elif FINE_MSAA_SAMPLE_COUNT == 16

layout(set = 0, binding = MSAA16_LUT_BINDING, std430) readonly buffer MsaaMaskLut {
  uint msaa_mask_lut[];
};

const uint MSAA_SAMPLE_COUNT = 16u;
const uint MSAA_FULL_MASK = 0xffffu;
const uint MASK_LUT_DIM = 256u;

#if CR_FILL_RULE_NONZERO
#define MSAA_NZ_WORDS_PER_PIXEL 8u
#endif

#else
#error Unsupported FINE_MSAA_SAMPLE_COUNT
#endif

const float EPS = 1e-6;
const float FINE_TILE_SIZE_F = 16.0;

const float ONE_MINUS_ULP = 0.99999994;
const float ROBUST_EPSILON = 2e-7;

const vec4 CLEAR_COLOR = vec4(0.0, 0.0, 0.0, 1.0);
const vec4 FILL_COLOR  = vec4(1.0, 1.0, 1.0, 1.0);

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 5, std430) readonly buffer TileInfos {
  TileInfo tile_infos[];
};

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 8, std430) readonly buffer TileEdges {
  TileEdge tile_edges[];
};

layout(set = 0, binding = IMG_BINDING, rgba8) uniform writeonly image2D img;

const float MASK_LUT_X_MIN = -1.0;
const float MASK_LUT_X_MAX =  2.0;
const float MASK_LUT_X_SCALE =
float(MASK_LUT_DIM) / (MASK_LUT_X_MAX - MASK_LUT_X_MIN);

uint quantize_mask_lut_x(float x) {
  float q = floor((x - MASK_LUT_X_MIN) * MASK_LUT_X_SCALE);
  return uint(clamp(q, 0.0, float(MASK_LUT_DIM - 1u)));
}

uint lookup_msaa_cell_mask(float x_top, float x_bottom) {
  uint top_ix = quantize_mask_lut_x(x_top);
  uint bottom_ix = quantize_mask_lut_x(x_bottom);

  uint entry_ix = top_ix * MASK_LUT_DIM + bottom_ix;

#if FINE_MSAA_SAMPLE_COUNT == 8
  uint word = msaa_mask_lut[entry_ix >> 2u];
  uint shift = (entry_ix & 3u) * 8u;

  return (word >> shift) & MSAA_FULL_MASK;
#else
  uint word = msaa_mask_lut[entry_ix >> 1u];
  uint shift = (entry_ix & 1u) * 16u;

  return (word >> shift) & MSAA_FULL_MASK;
#endif
}

#if !CR_FILL_RULE_NONZERO

// One bit per row of this tile.
// Bit y means row parity toggles starting at row y
shared uint sh_winding_y;

// One 16-bit event mask per row
shared uint sh_winding_x[FINE_TILE_SIZE];

// One 8-bit MSAA mask per pixel
shared uint sh_samples[FINE_TILE_SIZE * FINE_TILE_SIZE];

shared uint sh_scan_y;
shared uint sh_scan_x[FINE_TILE_SIZE];

#else

shared int sh_winding_y[FINE_TILE_SIZE];
shared int sh_winding_x[FINE_TILE_SIZE * FINE_TILE_SIZE];

shared uint sh_samples_pos[FINE_TILE_SIZE * FINE_TILE_SIZE * MSAA_NZ_WORDS_PER_PIXEL];
shared uint sh_samples_neg[FINE_TILE_SIZE * FINE_TILE_SIZE * MSAA_NZ_WORDS_PER_PIXEL];
shared uint sh_samples_touched[FINE_TILE_SIZE * FINE_TILE_SIZE];

shared int sh_scan_y[FINE_TILE_SIZE];
shared int sh_scan_x[FINE_TILE_SIZE * FINE_TILE_SIZE];

#endif

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

#if CR_FILL_RULE_NONZERO
  int winding_delta;
#endif
};

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

#if CR_FILL_RULE_NONZERO

int winding_delta_from_points(vec2 p0, vec2 p1) {
  return p1.y >= p0.y ? -1 : 1;
}

int left_y_edge_delta_from_points(vec2 p0, vec2 p1) {
  float dx = p1.x - p0.x;

  if (abs(dx) <= EPS) {
    return 0;
  }

  return dx > 0.0 ? 1 : -1;
}

#endif

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
#if !CR_FILL_RULE_NONZERO
  if (lane == 0u) {
    sh_winding_y = 0u;
  }

  for (uint i = lane; i < FINE_TILE_SIZE; i += WORKGROUP_SIZE) {
    sh_winding_x[i] = 0u;
  }

  for (uint i = lane; i < FINE_TILE_SIZE * FINE_TILE_SIZE; i += WORKGROUP_SIZE) {
    sh_samples[i] = 0u;
  }
#else
  for (uint i = lane; i < FINE_TILE_SIZE; i += WORKGROUP_SIZE) {
    sh_winding_y[i] = 0;
    sh_scan_y[i] = 0;
  }

  for (uint i = lane; i < FINE_TILE_SIZE * FINE_TILE_SIZE; i += WORKGROUP_SIZE) {
    sh_winding_x[i] = 0;
    sh_scan_x[i] = 0;
    sh_samples_touched[i] = 0u;
  }

  for (
      uint i = lane;
      i < FINE_TILE_SIZE * FINE_TILE_SIZE * MSAA_NZ_WORDS_PER_PIXEL;
      i += WORKGROUP_SIZE
      ) {
    sh_samples_pos[i] = 0u;
    sh_samples_neg[i] = 0u;
  }
#endif

  barrier();
}

#if !CR_FILL_RULE_NONZERO

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

#else

void emit_left_y_edge_event(float left_y_edge, int delta) {
  if (delta == 0) {
    return;
  }

  if (left_y_edge <= EPS || left_y_edge >= FINE_TILE_SIZE_F) {
    return;
  }

  uint row = uint(ceil(left_y_edge));

  if (row < FINE_TILE_SIZE) {
    atomicAdd(sh_winding_y[row], delta);
  }
}

void emit_x_winding_event(uint x, uint y, int delta) {
  if (x >= FINE_TILE_SIZE - 1u || y >= FINE_TILE_SIZE) {
    return;
  }

  uint event_x = x + 1u;

  atomicAdd(sh_winding_x[y * FINE_TILE_SIZE + event_x], delta);
}

uint spread2_to_halfwords(uint mask) {
  mask &= 0x3u;

  uint r = 0u;
  r |= ((mask >> 0u) & 1u) << 0u;
  r |= ((mask >> 1u) & 1u) << 16u;

  return r;
}

uint halfword_lane(uint word, uint lane) {
  return (word >> (lane * 16u)) & 0xffffu;
}

void emit_sample_winding(uint pix, uint mask, int delta) {
  mask &= MSAA_FULL_MASK;

  if (mask == 0u) {
    return;
  }

  atomicOr(sh_samples_touched[pix], mask);

  uint base = pix * MSAA_NZ_WORDS_PER_PIXEL;

  for (uint word_ix = 0u; word_ix < MSAA_NZ_WORDS_PER_PIXEL; word_ix++) {
    uint bits2 = (mask >> (word_ix * 2u)) & 0x3u;

    if (bits2 == 0u) {
      continue;
    }

    uint inc = spread2_to_halfwords(bits2);

    if (delta > 0) {
      atomicAdd(sh_samples_pos[base + word_ix], inc);
    } else {
      atomicAdd(sh_samples_neg[base + word_ix], inc);
    }
  }
}

#endif

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

  return lookup_msaa_cell_mask(x_top, x_bottom);
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

#if CR_FILL_RULE_NONZERO
  lp.winding_delta = is_down ? -1 : 1;
#endif

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
#if !CR_FILL_RULE_NONZERO
  if (lane == 0u) {
    sh_scan_y = prefix_even_odd_16(sh_winding_y);
  }

  if (lane < FINE_TILE_SIZE) {
    sh_scan_x[lane] = prefix_even_odd_16(sh_winding_x[lane]);
  }
#else
  if (lane == 0u) {
    int acc = 0;

    for (uint y = 0u; y < FINE_TILE_SIZE; y++) {
      acc += sh_winding_y[y];
      sh_scan_y[y] = acc;
    }
  }

  if (lane < FINE_TILE_SIZE) {
    uint y = lane;
    int acc = 0;

    for (uint x = 0u; x < FINE_TILE_SIZE; x++) {
      uint ix = y * FINE_TILE_SIZE + x;
      acc += sh_winding_x[ix];
      sh_scan_x[ix] = acc;
    }
  }
#endif

  barrier();
}

#if !CR_FILL_RULE_NONZERO

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

#else

uint resolve_nonzero_msaa_mask(uint pix, int base_winding) {
  uint touched = sh_samples_touched[pix];

  if (touched == 0u) {
    return base_winding != 0 ? MSAA_FULL_MASK : 0u;
  }

  uint base = pix * MSAA_NZ_WORDS_PER_PIXEL;
  uint mask = 0u;

  for (uint word_ix = 0u; word_ix < MSAA_NZ_WORDS_PER_PIXEL; word_ix++) {
    uint pos_word = sh_samples_pos[base + word_ix];
    uint neg_word = sh_samples_neg[base + word_ix];

    for (uint lane = 0u; lane < 2u; lane++) {
      uint sample_ix = word_ix * 2u + lane;

      int pos = int(halfword_lane(pos_word, lane));
      int neg = int(halfword_lane(neg_word, lane));

      int winding = base_winding + pos - neg;

      if (winding != 0) {
        mask |= 1u << sample_ix;
      }
    }
  }

  return mask & MSAA_FULL_MASK;
}

void write_one_msaa_pixel(
    TileInfo info,
    ivec2 pixel,
    uint x,
    uint y,
    bool tile_fully_inside
    ) {
  int base_winding =
    info.winding +
    sh_scan_y[y] +
    sh_scan_x[y * FINE_TILE_SIZE + x];

  uint mask = resolve_nonzero_msaa_mask(pixel_ix(x, y), base_winding);
  float c = float(bitCount(mask)) * (1.0 / float(MSAA_SAMPLE_COUNT));

  vec4 color = coverage_color(c);
  store_pixel_maybe_checked(pixel, color, tile_fully_inside);
}

#endif

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

#if !CR_FILL_RULE_NONZERO
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
#else
  {
    uint x = x0 + 0u;

    write_one_msaa_pixel(
        info,
        pixel + ivec2(0, 0),
        x,
        y,
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
        tile_fully_inside
        );
  }
#endif
}
