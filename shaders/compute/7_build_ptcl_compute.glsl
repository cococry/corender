#version 450
#extension GL_GOOGLE_include_directive : require

#include "../../shared/pc.glsl"

#define TILES_PER_MACROTILE_X 16
#define TILES_PER_MACROTILE_Y 16 
#define TILES_PER_MACROTILE TILES_PER_MACROTILE_X * TILES_PER_MACROTILE_Y 

#extension GL_KHR_shader_subgroup_basic      : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

#define WG_SIZE TILES_PER_MACROTILE

#define N_PATH_SLICES         8 

#define PTCL_INIT_CAP           64
#define PTCL_HEADROOM           2
#define PTCL_OVERFLOW_INCREMENT 256

#define SGS_PER_WG WG_SIZE / 32

layout(local_size_x = WG_SIZE) in;

uint ptcmd_off;
uint ptcmd_cap;

shared uint path_tile_bitmap[N_PATH_SLICES][TILES_PER_MACROTILE];

shared uint partition_offsets         [WG_SIZE];
shared uint partition_counts          [WG_SIZE];
shared uint partition_prefix          [WG_SIZE];
shared uint merged_path_draws         [WG_SIZE];
shared uint path_tile_widths          [WG_SIZE];
shared uint path_tile_widths_clipped  [WG_SIZE];
shared uint path_tile_tops_clipped    [WG_SIZE];
shared uint path_tile_coord_bases     [WG_SIZE];
shared uint path_tile_counts          [WG_SIZE];

shared uint subgroup_totals   [SGS_PER_WG];

layout(set = 0, binding = PATHS_BINDING, std430) readonly buffer Paths {
  DrawPath paths[];
};

layout(set = 0, binding = PATH_BBOXS_BINDING, std430) readonly buffer PathBBOXes {
  PathBBOX path_bboxes[];
};

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 10, std430) readonly buffer MacrotileMeta {
    MacrotileMetadata macrotile_metas[];
};

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 9, std430) readonly buffer BinnedDraws {
    uint binned_path_draws[];
};

layout(set = 0, binding = PATH_DRAWS_BINDING, std430) readonly buffer Draws {
  Draw path_draws[];
};

layout(set = 0, binding = PATH_TILE_BBOXS_BINDING, std430) readonly buffer PathTileBBOXes {
  PathTileBBOX path_tile_bboxes[];
};

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 5, std430) buffer TileInfos {
  TileInfo tile_infos[];
};
layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 2, std430) buffer TileEvents {
    int tile_events[];
};

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 11, std430) buffer PTCLs {
    uint ptcls[];
};

void ptcmd_alloc(uint size) {
  if(ptcmd_off + size >= ptcmd_cap) {
    // splill and write jmp command 
    uint cmd = pc.ptcl_spill_offset + atomicAdd(bump.n_ptcl, PTCL_INCREMENT);
    // TODO: check for bump overflow
  
    // Jump commands have no valid opcode
    ptcls[ptcmd_off] = ~cmd;
    ptcmd_off = cmd;
    ptcmd_cap = ptcmd_off + (PTCL_INIT_CAP - PTCL_HEADROOM);
  }
}

void ptcmd_for_path(uint tile, uint tile_id, uint n_segs) {
  if(n_segs == 0) {
    uint packed_data = 0;
    ptcls[ptcmd_off] = TILE_CMD_SOLID;
    ptcmd_off++;
  }
  else {
    // if the tile has edges, we need a base/offset into the 
    // segments array for that tile to know which segments
    // belong to this tile later.
    uint base = atomicAdd(bump.n_tile_segment_slots, n_segments);

    if (base + n_segments > pc.max_tile_storage) {
      atomicOr(bump.failed, FAIL_TILE_SEGMENT_OVERFLOW);
      return;
    }

    uint packed_data = 0;

  }

}

uint wg_prefix_sum_inclusive(
    uint lidx,
    inout uint values[WG_SIZE],
    inout uint subgroup_totals[SGS_PER_WG]
) {
    uint x = values[lidx];

    uint subgroup_prefix = subgroupInclusiveAdd(x);
    uint subgroup_total  = subgroupAdd(x);

    if (subgroupElect()) {
        subgroup_totals[gl_SubgroupID] = subgroup_total;
    }

    barrier();

    if (gl_SubgroupID == 0) {
        uint lane = gl_SubgroupInvocationID;
        uint v = lane < SGS_PER_WG ? subgroup_totals[lane] : 0u;
        uint prefix = subgroupExclusiveAdd(v);

        if (lane < SGS_PER_WG) {
            subgroup_totals[lane] = prefix;
        }
    }

    barrier();

    uint wg_prefix = subgroup_totals[gl_SubgroupID] + subgroup_prefix;
    values[lidx] = wg_prefix;

    barrier();

    return wg_prefix;
}

void main() {
  uvec2 macrotile = gl_WorkGroupID.xy;
  uint macrotile_id = pc.n_macrotiles_x * macrotile.y + macrotile.x;

  uint lidx = gl_LocalInvocationID.x;
  uint local_tile_id = lidx; 

  uvec2 local_tile = uvec2(
      local_tile_id % TILES_PER_MACROTILE_X,
      local_tile_id / TILES_PER_MACROTILE_X
  );
  uvec2 top_left_in_tiles = uvec2(
        TILES_PER_MACROTILE_X * macrotile.x, 
        TILES_PER_MACROTILE_Y * macrotile.y 
      );
  
  uint tile_id = (top_left_in_tiles.y + local_tile.y) * 
    pc.n_tiles_x + top_left_in_tiles.x + local_tile.x;

  ptcmd_off = tile_id * PTCL_INIT_CAP;
  ptcmd_cap = command_offset + (PTCL_INIT_CAP - PTCL_HEADROOM); 

  const uint n_elements_per_batch = WG_SIZE; 

  uint part_idx = 0;
  uint n_parts  = (pc.n_path_draws + n_elements_per_batch - 1u) / n_elements_per_batch; 

  uint partition_start_offset = 0;
  uint available_elements     = 0;
  uint consumed_elements      = 0;
  uint written_elements       = 0;

  while(part_idx < n_parts || consumed_elements < available_elements)
  {
    barrier();

    for (uint slice = 0u; slice < N_PATH_SLICES; slice++) 
    {
      path_tile_bitmap[slice][local_tile_id] =  0u;
    }

    while(true)
    {
      // Load new elements from the stream if we already wrote 
      // all previously loaded elements and still have 
      // batches/partitions left.
      if(written_elements == available_elements && part_idx < n_parts)
      {
        partition_start_offset = available_elements;

        uint path_draws_partition = part_idx + lidx;
        uint path_count = 0u;

        // Load source-buffer path partition offsets 
        // into shared memory for this lane if participating
        if(path_draws_partition < n_parts) {
          uint meta_idx = path_draws_partition * pc.n_macrotiles + macrotile_id;

          MacrotileMetadata meta = macrotile_metas[meta_idx];

          partition_offsets[lidx] = meta.off;
          path_count = meta.count; 
        }

        // prefix sum the element counts of the partition and get total 
        // available elements of this partition 
        {
          partition_counts[lidx] = path_count; // is 0 for inactive lanes

          uint prefix = wg_prefix_sum_inclusive(
              lidx,
              partition_counts,
              subgroup_totals
              );

          // exclusive start offset for the partition
          partition_prefix[lidx] = partition_start_offset + prefix - path_count;
          
          // inclusive end offset for the partition
          partition_counts[lidx] = partition_start_offset + prefix;

          barrier();

          // each lane gets available_elements
          available_elements = partition_counts[WG_SIZE - 1]; 
        }

        // each refill loads up to WG_SIZE partitions at once
        part_idx += WG_SIZE;
      }

      // Each lane now corresponds to a local path index 
      // within the current batch. We then binary search
      // to find the partition the path is in. This is 
      // done to build a merged, flat list of path-path_draws within 
      // the current stream batch. 
      uint flat_path_idx = consumed_elements + lidx; 
      if(flat_path_idx >= written_elements && flat_path_idx < available_elements)
      {
        uint partition_idx = 0u;

        for (uint step = WG_SIZE >> 1u; step > 0u; step >>= 1u) {
          uint probe = partition_idx + step;

          if (flat_path_idx >= partition_counts[probe - 1u]) {
            partition_idx = probe;
          }
        }

        uint local_idx = flat_path_idx - partition_prefix[partition_idx];
        uint off = partition_offsets[partition_idx];
        merged_path_draws[lidx] = binned_path_draws[off + local_idx];
      }

      written_elements = min(consumed_elements + n_elements_per_batch, available_elements);
      if(written_elements - consumed_elements >= n_elements_per_batch || 
          (written_elements >= available_elements && part_idx >= n_parts)
          ) break;

      barrier();
    }

    // get touched tiles for each path draw  
    uint n_tiles      = 0u;
    uint op           = 0u;

    if(lidx + consumed_elements < written_elements) {
      Draw draw = path_draws[merged_path_draws[lidx]];
      op        = draw_op(draw);

      uint path_id      = draw.path_id; 
      PathTileBBOX bbox = path_tile_bboxes[path_id]; 

      uint tile_w_unclipped  = bbox.x1 - bbox.x0;
      path_tile_widths[lidx] = tile_w_unclipped; 

      ivec2 local_mn = ivec2(bbox.x0, bbox.y0) - ivec2(top_left_in_tiles); 
      ivec2 local_mx = ivec2(bbox.x1, bbox.y1) - ivec2(top_left_in_tiles); 

      uvec2 mn = uvec2(clamp(local_mn, ivec2(0), 
            ivec2(TILES_PER_MACROTILE_X, TILES_PER_MACROTILE_Y)));
      uvec2 mx = uvec2(clamp(local_mx, ivec2(0), 
            ivec2(TILES_PER_MACROTILE_X, TILES_PER_MACROTILE_Y)));

      uint tile_w   = mx.x - mn.x; 
      uint tile_h   = mx.y - mn.y;
      n_tiles = tile_w * tile_h; 

      path_tile_widths_clipped [lidx]  = tile_w; 
      path_tile_tops_clipped   [lidx]  = (mn.x | (mn.y << 16u));
      path_tile_coord_bases    [lidx]  = bbox.tiles_offset - 
        (local_mn.y * tile_w_unclipped + local_mn.x);
    }

    path_tile_counts[lidx] = n_tiles;

    uint prefix = wg_prefix_sum_inclusive(
        lidx,
        path_tile_counts,
        subgroup_totals
        );

    path_tile_counts[lidx] = prefix;

    barrier();

    uint n_tiles_touched_total = path_tile_counts[WG_SIZE - 1];

    for(uint i = lidx; i < n_tiles_touched_total; i += WG_SIZE) {
      uint path_draw_idx_local = 0u;

      // find path-draw of this tile touch  
      for (uint step = WG_SIZE >> 1u; step > 0u; step >>= 1u) {
        uint probe = path_draw_idx_local + step;

        if (i >= path_tile_counts[probe - 1u]) {
          path_draw_idx_local = probe;
        }
      }

      // 0-based index within that draw object’s tile subset.
      // Logically, local_tile_flat can never underflow because:
      //  if (i >= path_tile_counts[probe - 1u]) {
      //    path_draw_idx_local = probe;
      //  }
      // Meaning i will always be >= path_tile_counts[path_draw_idx_local - 1u]
      // (given path_draw_idx_local > 0)
      uint local_tile_flat = i - 
        (path_draw_idx_local > 0 ? path_tile_counts[path_draw_idx_local - 1u] : 0); 

      uint path_width = path_tile_widths_clipped[path_draw_idx_local];
      uint path_top   = path_tile_tops_clipped[path_draw_idx_local]; 

      uint local_x = (path_top & 0xffffu) + local_tile_flat % path_width;
      uint local_y = ((path_top >> 16u) & 0xffffu) + local_tile_flat / path_width;

      uint tile_id = path_tile_coord_bases[path_draw_idx_local] + 
        (path_tile_widths[path_draw_idx_local] * local_y) + local_x;

      TileInfo tile_info  = tile_infos[tile_id];
      bool has_edges      = tile_info_seg_count(tile_info) > 0; 
      uint path_draw_idx  = merged_path_draws[path_draw_idx_local];
      Draw path_draw      = path_draws[path_draw_idx];

      // TODO: Fill rule
      bool even_odd       = true;
      int winding         = tile_events[tile_id];
      bool winding_inside = !even_odd ? winding != 0 : (winding & 1) != 0;

      if(has_edges || winding_inside) {
        uint my_slice   = path_draw_idx / 32u;
        uint my_mask    = 1u << (path_draw_idx & 31u); 
        uint local_tile = TILES_PER_MACROTILE_X * local_y + local_x;
        atomicOr(path_tile_bitmap[my_slice][local_tile], my_mask);
      }
    }

    barrier();

    // write command lists for each relevant tile 
    for(uint slice = 0; slice < N_PATH_SLICES; slice++) {
      uint bitmap = path_tile_bitmap[slice][lidx];
      if(bitmap == 0u) continue;

      // consume one path draw of the current slice's bitmap 
      // at a time
      for(; bitmap != 0; bitmap &= bitmap - 1u) {
        uint path_draw_local  = slice * 32u + uint(findLSB(bitmap));

        uint tile_id = path_tile_coord_bases[path_draw_local] + 
          (path_tile_widths[path_draw_local] * local_tile.y) + local_tile.x;

        // WRITE the actual ptcls
        uint path_draw_idx    = merged_path_draws[path_draw_local];
        Draw path_draw        = path_draws[path_draw_idx];
      }
    }

    consumed_elements = written_elements; 
  }
}
