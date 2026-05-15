#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require

#include "../../shared/pc.glsl"
#include "../../shared/fine_eval_msaa_common.glsl"

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 7, std430) readonly buffer ActiveTiles {
  uint active_tiles[];
};

void accumulate_msaa_cell_direct(
    MsaaLineWalkParams lp,
    uint local_tile,
    float n_x_steps,
    bool changed_pixel_row
    ) {
  int x = int(lp.x0 + lp.x_sign * n_x_steps);

  int y = int(lp.y0 + float(local_tile) - n_x_steps);

  if (
      x < 0 ||
      y < 0 ||
      x >= int(FINE_TILE_SIZE) ||
      y >= int(FINE_TILE_SIZE)
     ) {
    return;
  }

  if (changed_pixel_row) {
#if CR_FILL_RULE_NONZERO
    emit_x_winding_event(uint(x), uint(y), lp.winding_delta);
#else
    emit_x_winding_event(uint(x), uint(y));
#endif
  }


    uint pix = pixel_ix(uint(x), uint(y));

#if CR_FILL_RULE_NONZERO
  uint x_mask = compute_cell_sample_x_mask_lut(
      lp,
      uint(x),
      uint(y)
      );

  uint y_mask = compute_cell_sample_y_mask_lut(
      lp,
      uint(x),
      uint(y)
      );

  if (x_mask != 0u) {
    emit_sample_winding(
        pix,
        x_mask,
        lp.winding_delta
        );
  }

  if (y_mask != 0u) {
    emit_sample_winding(
        pix,
        y_mask,
        lp.left_delta
        );
  }
#else
  uint mask = compute_cell_sample_mask_lut(
      lp,
      uint(x),
      uint(y)
      );

  if (mask != 0u) {
    atomicXor(sh_samples[pix], mask);
  }
#endif

}

void shade_fine_tile_msaa_dense(
    TileInfo info,
    uint tile_x,
    uint tile_y,
    uvec2 local_base,
    bool tile_fully_inside
    ) {
  uint lane = gl_LocalInvocationIndex;

  clear_msaa_shared(lane);

  // For dense tiles, edges are iterated directly without a pixel-cell scheduler.
  // This is useful for pathological dense tiles, as it removes binary_search_edge()
  // from every pixel-cell job, it removes doing prefix sums per edge chunk and it 
  // avoids lots of shared memory traffic.
  // Also, dense tiles already have enough natural parallelism. With info.count >= 64, 
  // every lane has at least one edge, often several, so load balance isn't really 
  // a problem here. 
  for (uint edge_local = lane; edge_local < info.count; edge_local += WORKGROUP_SIZE) {
    TileEdge edge = tile_edges[info.base + edge_local];

    float left_y_edge = get_left_y_edge_from_packed(edge.p0, edge.p1);

    vec2 p0 = unpack_point(edge.p0);
    vec2 p1 = unpack_point(edge.p1);

#if CR_FILL_RULE_NONZERO
    int left_delta = left_y_edge_delta_from_points(p0, p1);
#endif

    apply_grid_x_nudge(p0, p1);

    MsaaLineWalkParams lp;

    if (make_msaa_line_walk_params(p0, p1, lp)) {
#if CR_FILL_RULE_NONZERO
      lp.left_delta = left_delta;
#endif
      float prev_n_x_steps = 0.0;

      for (uint local_tile = 0u; local_tile < lp.count; local_tile++) {
        float n_x_steps =
          floor(lp.x_step_rate * float(local_tile) + lp.x_step_start_offset);

        bool changed_pixel_row;

        if (local_tile == 0u) {
          changed_pixel_row = abs(lp.y0 - lp.xy0.y) <= EPS;
        } else {
          changed_pixel_row = n_x_steps == prev_n_x_steps;
        }

        accumulate_msaa_cell_direct(
            lp,
            local_tile,
            n_x_steps,
            changed_pixel_row
            );

        prev_n_x_steps = n_x_steps;
      }
    }

#if CR_FILL_RULE_NONZERO
    emit_left_y_edge_event(left_y_edge, left_delta);
#else
    emit_left_y_edge_event(left_y_edge);
#endif
  }

  barrier();

  write_msaa_pixels(info, tile_x, tile_y, local_base, tile_fully_inside);
}


// dispatched over all active tiles
// each workgroup handles one active tile
// each workgroup contains 4(x) x 16(y) = 64 invocations
void main() {
  uint active_ix = gl_WorkGroupID.x;

  if (active_ix >= bump.n_active_tiles_dense) {
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

  shade_fine_tile_msaa_dense(info, tile_x, tile_y, local_base, tile_fully_inside);
}
