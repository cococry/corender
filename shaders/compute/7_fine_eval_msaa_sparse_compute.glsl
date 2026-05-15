#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require

#include "../../shared/pc.glsl"
#include "../../shared/fine_eval_msaa_common.glsl"

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 6, std430) readonly buffer ActiveTiles {
  uint active_tiles[];
};

shared TileEdge sh_edges[EDGE_CHUNK_SIZE];

shared uint sh_count[WORKGROUP_SIZE];
shared uint sh_subgroup_sums[N_SUBGROUPS];

shared MsaaLineWalkParams sh_walk[EDGE_CHUNK_SIZE];

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
#if CR_FILL_RULE_NONZERO
    emit_x_winding_event(uint(x), uint(y), lp.winding_delta);
#else
    emit_x_winding_event(uint(x), uint(y));
#endif
  }

  // compute this edge's 16 sample coverage mask for the pixel cell
  uint mask = compute_cell_sample_mask_lut(
      lp,
      uint(x),
      uint(y)
      );

#if CR_FILL_RULE_NONZERO
  if (mask != 0u) {
    emit_sample_winding(
        pixel_ix(uint(x), uint(y)),
        mask,
        lp.winding_delta
        );
  }
#else
  // xor edge coverage into the shared per-pixel MSAA mask
  if (mask != 0u) {
    atomicXor(sh_samples[pixel_ix(uint(x), uint(y))], mask);
  }
#endif
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

#if CR_FILL_RULE_NONZERO
      int left_delta = left_y_edge_delta_from_points(p0, p1);
#endif

      apply_grid_x_nudge(p0, p1);

      MsaaLineWalkParams lp;

      if (make_msaa_line_walk_params(p0, p1, lp)) {
        sh_walk[lane] = lp;
        pixel_count = lp.count;
      } else {
        pixel_count = 0u;
      }

      // emit the parity event per edge
#if CR_FILL_RULE_NONZERO
      emit_left_y_edge_event(left_y_edge, left_delta);
#else
      emit_left_y_edge_event(left_y_edge);
#endif
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

  if (active_ix >= bump.n_active_tiles_sparse) {
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
