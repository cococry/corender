// Approach largely from: https://github.com/linebender/vello/blob/main/vello_shaders/shader/binning.wgsl  
#version 450 core

#extension GL_GOOGLE_include_directive : require
#include "../../shared/pc.glsl"

#define MAX_TILES 256 
#define WG_SIZE 256
#define N_SLICE (WG_SIZE / 32)
#define N_SUBSLICE 4

#define CONTRIBUTION_BITMAP(slice, tile) contribution_bitmap[(slice) * MAX_TILES + (tile)]
#define MACROTILE_COUNT(subslice, tile) tile_count[(subslice) * MAX_TILES + (tile)]

layout(local_size_x = WG_SIZE) in;

layout(set = 0, binding = PATHS_BINDING, std430) readonly buffer Paths {
  DrawPath paths[];
};

layout(set = 0, binding = PATH_DRAWS_BINDING, std430) readonly buffer Draws {
  Draw path_draws[];
};

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 9, std430) writeonly buffer BinnedPaths {
  uint binned_path_draws[];
};

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 10, std430) writeonly buffer MacrotileMeta {
  MacrotileMetadata macrotile_metas[];
};

layout(set = 0, binding = PATH_BBOXS_BINDING, std430) readonly buffer PathBBOXes {
  PathBBOX path_bboxes[];
};

shared uint contribution_bitmap[N_SLICE * MAX_TILES];
shared uint tile_count[N_SUBSLICE * MAX_TILES];
shared uint macrotile_offset[MAX_TILES];

void add_draw_to_macrotile(
    uint tile,
    uint my_slice,
    uint my_mask,
    uint draw_id
    ) {
  uint bitmap = CONTRIBUTION_BITMAP(my_slice, tile);

  /* Get the local rank for this lane. This rank is */             
  /* basically asking: within my lane's 32 lane slice, how many */        
  /* contributing lanes before me also wrote to this tile? */
  uint rank = bitCount(bitmap & (my_mask - 1u));

  if (my_slice > 0u) {
    /* Add prefix count from previous slices to get */                    
    /* a tile-local offset */
    uint prev_slice = my_slice - 1u;
    uint packed_count = MACROTILE_COUNT(prev_slice >> 1u, tile);
    uint shift = 16u * (prev_slice & 1u);
                              
    /* rank becomes the tile-local offset of this draw */                 
    /* inside that tile's list. */
    rank += (packed_count >> shift) & 0xffffu;
  }

  /* Now add the tile-local offset (rank) to the global macrotile offset */
  binned_path_draws[macrotile_offset[tile] + rank] = draw_id;
}

void main() {
  uint lidx = gl_LocalInvocationID.x;
  uint draw_id = gl_GlobalInvocationID.x;
    
  uint macrotile_page = gl_WorkGroupID.y;
  uint page_base = macrotile_page * WG_SIZE; 
  uint page_end = min(page_base + WG_SIZE, pc.n_macrotiles); 

  uint n_macrotiles = pc.n_macrotiles_x * pc.n_macrotiles_y;

  // clear the shared mem
  for (uint i = lidx; i < N_SLICE * MAX_TILES; i += WG_SIZE) {
    contribution_bitmap[i] = 0u;
  }

  for (uint i = lidx; i < N_SUBSLICE * MAX_TILES; i += WG_SIZE) {
    tile_count[i] = 0u;
  }

  for (uint i = lidx; i < MAX_TILES; i += WG_SIZE) {
    macrotile_offset[i] = 0u;
  }

  barrier();

  // First (1.) Pass: Populate the contribution bitmap within this workgroup.
  // The contribution_bitmap shows what lanes in a given slice contribute 
  // path_draws to a given tile.

  // Example:
  // CONTRIBUTION_BITMAP(0, 12) = [0, 0, 1, 1, 0, 1, ...] (simplified) 
  // Means: For macrotile 12, in slice 0, lanes 2, 3, and 5 contribute path_draws.

  // Since each lane corresponds to one draw in this workgroup, it tells us 
  // which path_draws (of the path_draws processed by this workgroup) overlap which macrotile

  ivec2 t0 = ivec2(0);
  ivec2 t1 = ivec2(-1);

  uint my_slice = lidx / 32u;
  uint my_mask = 1u << (lidx & 31u);

  bool single_tile = false;

  if (draw_id < pc.n_path_draws) {
    Draw draw = path_draws[draw_id];
    DrawPath path = paths[draw.path_id];

    PathBBOX bbox = path_bboxes[draw.path_id];

    vec2 mn = bbox.mn;
    vec2 mx = bbox.mx;

    ivec2 ip0 = ivec2(floor(mn));
    ivec2 ip1 = ivec2(floor(mx));

    // Divide by 256, as macrotiles will always be 256x256 pixels (safes div op) 
    t0 = ip0 >> 8;
    t1 = ip1 >> 8;

    // clamp to valid ranges
    t0 = clamp(
        t0, 
        ivec2(0), 
        ivec2(int(pc.n_macrotiles_x - 1u), int(pc.n_macrotiles_y - 1u)));

    t1 = clamp(
        t1,
        ivec2(0),
        ivec2(int(pc.n_macrotiles_x - 1u),int(pc.n_macrotiles_y - 1u)));


    single_tile = all(equal(t0, t1));

    if(single_tile) {
      uint tile = uint(t0.y) * pc.n_macrotiles_x + uint(t0.x);

      if(tile >= page_base && tile < page_end) {
        uint local_tile = tile - page_base;
        atomicOr(CONTRIBUTION_BITMAP(my_slice, local_tile), my_mask);
      }
    } else {
      // add this element's contribution to all macrotiles it overlaps 
      for (int y = t0.y; y <= t1.y; y++) {
        uint tile = uint(y) * pc.n_macrotiles_x + uint(t0.x);
        for (int x = t0.x; x <= t1.x; x++, tile++) {

          if(tile >= page_base && tile < page_end) {
            uint local_tile = tile - page_base;
            atomicOr(CONTRIBUTION_BITMAP(my_slice, local_tile), my_mask);
          }
        }
      }
    }
  }

  barrier();

  // Second (2.) Pass: The local invocation (lidx, lane) now acts as the 
  // flat macrotile index 

  uint local_tile = lidx;
  uint global_tile = page_base + lidx;
  if (global_tile < pc.n_macrotiles) {
    uint element_count = 0u;

    // prefix-sum the element contribution of all slices
    // - which was calculated for all slices of this workgroup 
    // in the first (1) pass - for this lane's tile.

    // this is done by packing the contribution of two slices into 
    // one 32 bit integer, to save memory and loop invocations
    for (uint i = 0u; i < N_SUBSLICE; i++) {
      element_count += bitCount(CONTRIBUTION_BITMAP(i * 2u, local_tile));
      uint count_lo = element_count;

      element_count += bitCount(CONTRIBUTION_BITMAP(i * 2u + 1u, local_tile));
      uint count_hi = element_count;

      uint count_packed = count_lo | (count_hi << 16u);
      MACROTILE_COUNT(i, local_tile) = count_packed;
    }

    // increment global bump to get global offset into macrotile data 
    uint off = atomicAdd(bump.n_binned_path_draws, element_count);

    macrotile_offset[lidx] = off;

    uint draw_partition = gl_WorkGroupID.x;
    uint meta_idx = draw_partition * pc.n_macrotiles + global_tile;

    macrotile_metas[meta_idx].count = element_count;
    macrotile_metas[meta_idx].off = off;
  }

  barrier();

  // Third (3.) Pass: Scatter the draw to all tiles it overlaps. 
  // We compute slice-local and bin-local ranks before adding the 
  // global offset that was calculated in pass two (2) to get a 
  // unique index into binned_path_draws. See add_draw_to_marotile() for more info.
  if (draw_id < pc.n_path_draws) {

    if(single_tile) {
      uint tile = uint(t0.y) * pc.n_macrotiles_x + uint(t0.x);
      if(tile >= page_base && tile < page_end) {
        uint local_tile = tile - page_base;
        add_draw_to_macrotile(local_tile, my_slice, my_mask, draw_id);
      }
    } else {
      for (int y = t0.y; y <= t1.y; y++) {
        uint tile = uint(y) * pc.n_macrotiles_x + uint(t0.x);
        for (int x = t0.x; x <= t1.x; x++, tile++) {
          if(tile >= page_base && tile < page_end) {
            uint local_tile = tile - page_base;
            add_draw_to_macrotile(local_tile, my_slice, my_mask, draw_id);
          }
        }
      }
    }
  }
}

#undef CONTRIBUTION_BITMAP
#undef MACROTILE_COUNT

