#version 450
#extension GL_GOOGLE_include_directive : require

#include "../../shared/pc.glsl"

#define TILES_PER_MACROTILE_X 16
#define TILES_PER_MACROTILE_Y 16 
#define TILES_PER_MACROTILE TILES_PER_MACROTILE_X * TILES_PER_MACROTILE_Y 

#define WG_SIZE TILES_PER_MACROTILE

#define N_PATH_SLICES         8 

#define PTCL_INIT_CAP         64
#define PTCL_HEADROOM         2

layout(local_size_x = WG_SIZE) in;

uint command_offset;
uint command_limit;

shared uint path_slices_bitmap_per_tile[N_PATH_SLICES][TILES_PER_MACROTILE];
shared uint partition_offsets[WG_SIZE];

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 10, std430) readonly buffer MacrotileMeta {
    MacrotileMetadata macrotile_metas[];
};

/*
uint binary_search_draw_object(uint goal, uint n_parts) {
  uint lo = 0u;
  uint hi = n_parts;

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
*/

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

  command_offset = tile_id * PTCL_INIT_CAP;
  command_limit = command_offset + (PTCL_INIT_CAP - PTCL_HEADROOM); 

  const uint n_elements_per_batch = WG_SIZE; 

  uint part_idx = 0;
  uint n_parts  = (pc.n_paths + n_elements_per_batch - 1u) / n_elements_per_batch; 

  uint partition_start_offset = 0;
  uint available_elements     = 0;
  uint consumed_elements      = 0;
  uint written_elements       = 0;


  while(part_idx < n_parts || consumed_elements < available_elements)
  {
    barrier();

    for (uint slice = 0u; slice < N_PATH_SLICES; slice++) 
    {
      path_slices_bitmap_per_tile[slice][local_tile_id] =  0u;
    }

    while(true)
    {
      if(written_elements == available_elements && part_idx < n_parts)
      {
        partition_start_offset = available_elements;
        uint paths_partition = part_idx + lidx;

        uint path_count = 0u;

        if(paths_partition < n_parts) {
          uint meta_idx = paths_partition * pc.n_macrotiles + macrotile_id;

          MacrotileMetadata meta = macrotile_metas[meta_idx];

          partition_offsets[lidx] = meta.off;
          path_count = meta.count;
        }

        // TODO: Prefix sum the element counts

        // available_elements = sh_part_count[WG_SIZE - 1u] 

        part_idx += WG_SIZE;
      }

      // TODO: use binary search to find draw object to read

      written_elements = min(consumed_elements + n_elements_per_batch, available_elements);
      if(written_elements - consumed_elements >= n_elements_per_batch || 
          (written_elements >= available_elements && part_idx >= n_parts)
          ) break;

      barrier();
    }

    // TODO: do a lot of other stuff

    consumed_elements = written_elements; 
  }
}
