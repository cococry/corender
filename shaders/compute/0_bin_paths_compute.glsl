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

#define ADD_PATH_TO_MACROTILE(tile) {                                      \
    uint current_bitmap = CONTRIBUTION_BITMAP(my_slice, tile);       \
    /* Get the local rank for the current lane. This rank is */     \
    /* basically asking: within my lane's 32 lane slice, how many */\
    /* contributing lanes before me also wrote to this tile? */      \
    uint rank = bitCount(current_bitmap & (my_mask - 1u));          \
    if (my_slice > 0u) {                                            \
        /* Add prefix count from previous slices to get */          \
        /* a workgroup- or tile-local rank/offset */                 \
        uint previous = my_slice - 1u;                              \
        uint prefix_count_packed = MACROTILE_COUNT(previous / 2u, tile);   \
        /* rank becomes the tile-local offset of this path */        \
        /* inside that tile's list. */                               \
        rank += (                                                   \
                prefix_count_packed >>                              \
                (16u * (previous & 1u))                             \
                ) & 0xffffu;                                        \
    }                                                               \
    /* Now add the local offset/rank to the global macrotile offset */\
    binned_paths[macrotile_offset[tile] + rank] = path_id;    \
}

layout(local_size_x = WG_SIZE) in;

layout(set = 0, binding = PATHS_BINDING, std430) readonly buffer Paths {
    DrawPath paths[];
};

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 9, std430) writeonly buffer BinnedPaths {
    uint binned_paths[];
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

void main() {
    uint lidx = gl_LocalInvocationID.x;
    uint path_id = gl_GlobalInvocationID.x;

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
    // paths to a given tile.

    // Example:
    // CONTRIBUTION_BITMAP(0, 12) = [0, 0, 1, 1, 0, 1, ...] (simplified) 
    // Means: For macrotile 12, in slice 0, lanes 2, 3, and 5 contribute paths.

    // Since each lane corresponds to one path in this workgroup, it tells us 
    // which paths (of the paths processed by this workgroup) overlap which macrotile

    ivec2 t0 = ivec2(0);
    ivec2 t1 = ivec2(-1);

    uint my_slice = lidx / 32u;
    uint my_mask = 1u << (lidx & 31u);

    bool single_tile = false;

    if (path_id < pc.n_paths) {
        DrawPath path = paths[path_id];

        vec2 mn = path_bboxes[path_id].mn;
        vec2 mx = path_bboxes[path_id].mx;

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

            if (tile < MAX_TILES) {
                atomicOr(CONTRIBUTION_BITMAP(my_slice, tile), my_mask);
            }
        } else {
            // add this element's contribution to all macrotiles it overlaps 
            for (int y = t0.y; y <= t1.y; y++) {
                uint tile = uint(y) * pc.n_macrotiles_x + uint(t0.x);
                for (int x = t0.x; x <= t1.x; x++, tile++) {

                    if (tile < MAX_TILES) {
                        atomicOr(CONTRIBUTION_BITMAP(my_slice, tile), my_mask);
                    }
                }
            }
        }
    }

    barrier();

    // Second (2.) Pass: The local invocation (lidx, lane) now acts as the 
    // flat macrotile index 
    if (lidx < n_macrotiles && lidx < MAX_TILES) {
        uint element_count = 0u;

        // prefix-sum the element contribution of all slices
        // - which was calculated for all slices of this workgroup 
        // in the first (1) pass - for this lane's tile.

        // this is done by packing the contribution of two slices into 
        // one 32 bit integer, to save memory and loop invocations
        for (uint i = 0u; i < N_SUBSLICE; i++) {
            element_count += bitCount(CONTRIBUTION_BITMAP(i * 2u, lidx));
            uint count_lo = element_count;

            element_count += bitCount(CONTRIBUTION_BITMAP(i * 2u + 1u, lidx));
            uint count_hi = element_count;

            uint count_packed = count_lo | (count_hi << 16u);
            MACROTILE_COUNT(i, lidx) = count_packed;
        }

        // increment global bump to get global offset into macrotile data 
        uint off = atomicAdd(bump.n_binned_paths, element_count);

        macrotile_offset[lidx] = off;

        uint draw_partition = gl_WorkGroupID.x;
        uint macrotile_page = gl_WorkGroupID.y;
        uint lidx = gl_LocalInvocationID.x;

        uint global_macrotile = macrotile_page * WG_SIZE + lidx;
        if (global_macrotile >= pc.n_macrotiles) {
          return;
        }

        uint meta_idx = draw_partition * pc.n_macrotiles + global_macrotile;

        macrotile_metas[meta_idx].count = element_count;
        macrotile_metas[meta_idx].off = off;
    }

    barrier();

    // Third (3.) Pass: Scatter the path to all tiles it overlaps. 
    // We compute slice-local and bin-local ranks before adding the 
    // global offset that was calculated in pass two (2) to get a 
    // unique index into binned_paths. See ADD_PATH_TO_TILE for more info.
    if (path_id < pc.n_paths) {

        if(single_tile) {
            uint tile = uint(t0.y) * pc.n_macrotiles_x + uint(t0.x);
            if (tile < MAX_TILES) {
                ADD_PATH_TO_MACROTILE(tile);
            }
        } else {
            for (int y = t0.y; y <= t1.y; y++) {
                uint tile = uint(y) * pc.n_macrotiles_x + uint(t0.x);
                for (int x = t0.x; x <= t1.x; x++, tile++) {
                    if (tile < MAX_TILES) {
                        ADD_PATH_TO_MACROTILE(tile);
                    }
                }
            }
        }
    }
}

#undef ADD_PATH_TO_MACROTILE
#undef CONTRIBUTION_BITMAP
#undef MACROTILE_COUNT

