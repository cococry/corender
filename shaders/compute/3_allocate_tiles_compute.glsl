#version 450
#extension GL_GOOGLE_include_directive : require

#include "../../shared/pc.glsl"

layout(local_size_x = 256) in;

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 2, std430) buffer TileEvents {
    int tile_events[];
};

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 4, std430) buffer TileSegments {
    uint tile_n_segments[];
};

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 5, std430) buffer TileInfos {
    TileInfo tile_infos[];
};

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 6, std430) buffer ActiveTilesSparse {
    uint active_tiles_sparse[];
};

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 7, std430) buffer ActiveTilesDense {
    uint active_tiles_dense[];
};

#if CR_ENABLE_GPU_STATS
layout(set = 0, binding = STATS_BINDING, std430) buffer StatsBuffer {
    GpuStats stats;
};
#endif

#if CR_ENABLE_GPU_STATS
uint hist_bin(uint v) {
    if (v == 0u) {
        return 0u;
    }

    int msb = findMSB(v);
    return uint(clamp(msb + 1, 1, int(STATS_HIST_BINS) - 1));
}
#endif

uint abs_i32_to_u32(int v) {
    if (v >= 0) {
        return uint(v);
    }

    return uint(-(v + 1)) + 1u;
}

// dispatched over all screen tiles. 
// each workgroup handles 256 tiles. 
void main() {
    uint tile_id = gl_GlobalInvocationID.x;

    if (tile_id >= pc.n_tiles) {
        return;
    }

    uint n_segments = tile_n_segments[tile_id];
    int winding = tile_events[tile_id];

#if CR_ENABLE_GPU_STATS
    if (n_segments >= 2u) {
        CR_STAT_ADD(contention_tiles, 1u);
        CR_STAT_ADD(contended_tile_segments, n_segments);
        CR_STAT_ADD(excess_tile_segments, n_segments - 1u);

        uint capped_n = min(n_segments, 65535u);
        uint pair_pressure = (capped_n * (capped_n - 1u)) >> 1u;

        CR_STAT_ADD(tile_atomic_pair_pressure, pair_pressure);
    }

    CR_STAT_ADD(n_tiles_seen, 1u);

    CR_STAT_ADD(total_tile_segments, n_segments);
    CR_STAT_MAX(max_tile_segments, n_segments);

    uint tile_bin = hist_bin(n_segments);
    CR_STAT_HIST_ADD(hist_tile_segments, tile_bin, 1u);
#endif

#if CR_ENABLE_GPU_STATS
    uint abs_w = abs_i32_to_u32(winding);
    CR_STAT_MAX(max_abs_winding, abs_w);
#endif

    bool has_edges = n_segments != 0u;

    uint flags = TILE_EMPTY;

#if EVEN_ODD_WINDING
    bool winding_inside = (winding & 1) != 0;
#else
    bool winding_inside = winding != 0;
#endif

    if (has_edges) {
      flags |= TILE_FINE;
    } else if (winding_inside) {
      // if the tile does not have edges touching it 
      // but the winding rule says it is inside, it is
      // a solid tile.
      flags |= TILE_SOLID;
    }

#if CR_ENABLE_GPU_STATS
    if (flags == TILE_EMPTY) {
      CR_STAT_ADD(empty_tiles, 1u);
    } else {
      CR_STAT_ADD(active_tiles, 1u);

      if ((flags & TILE_SOLID) != 0u) {
        CR_STAT_ADD(solid_tiles, 1u);
        }

        if ((flags & TILE_FINE) != 0u) {
            CR_STAT_ADD(fine_tiles, 1u);

            CR_STAT_ADD(total_fine_segments, n_segments);
            CR_STAT_MAX(max_fine_segments, n_segments);

            uint fine_bin = hist_bin(n_segments);
            CR_STAT_HIST_ADD(hist_fine_segments, fine_bin, 1u);

            CR_STAT_ADD(estimated_fine_edge_evals, n_segments * 64u);
        }
    }
#endif

    uint base = 0u;

    // if the tile has edges, we need a base/offset into the 
    // segments array for that tile to know which segments
    // belong to this tile later.
    if (has_edges) {
        base = atomicAdd(bump.n_tile_segment_slots, n_segments);

        if (base + n_segments > pc.max_tile_storage) {
            atomicOr(bump.failed, FAIL_TILE_SEGMENT_OVERFLOW);
            return;
        }
    }

    tile_infos[tile_id].base = base;
    tile_infos[tile_id].count = n_segments;
    tile_infos[tile_id].winding = winding;
    tile_infos[tile_id].flags = flags;

    // if the tile is not empty, it's active and we add it 
    // to the list of dispatched tiles for fine.
    if (flags != TILE_EMPTY) {
      if ((flags & TILE_FINE) != 0u && n_segments >= TILE_DENSE_THRESHOLD) {
        uint active_idx = atomicAdd(bump.n_active_tiles_dense, 1u);

        if (active_idx >= pc.n_tiles) {
          atomicOr(bump.failed, FAIL_ACTIVE_TILE_OVERFLOW);
          return;
        }

        active_tiles_dense[active_idx] = tile_id;
      } else {
        uint active_idx = atomicAdd(bump.n_active_tiles_sparse, 1u);

        if (active_idx >= pc.n_tiles) {
          atomicOr(bump.failed, FAIL_ACTIVE_TILE_OVERFLOW);
          return;
        }

        active_tiles_sparse[active_idx] = tile_id;
      }
    }
}
