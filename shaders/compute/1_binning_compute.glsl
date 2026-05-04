// Approach from: https://github.com/linebender/vello/blob/main/vello_shaders/shader/binning.wgsl  
#version 450 core

#extension GL_EXT_debug_printf : enable

#define MAX_TILES 135
#define WG_SIZE 256
#define N_SLICE (WG_SIZE / 32)
#define N_SUBSLICE 4

layout(local_size_x = WG_SIZE) in;

struct DrawPath {
    uint segment_offset;
    uint segment_count;

    vec2 mn;
    vec2 mx;

    uint id;
};

struct MacrotileMetadata {
    uint off;
    uint count;
};

layout(set = 0, binding = 18, std430) readonly buffer Paths {
    DrawPath paths[];
};

layout(set = 0, binding = 14, std430) writeonly buffer BinnedPaths {
    uint binned_paths[];
};

layout(set = 0, binding = 15, std430) buffer BumpAllocator {
    uint bump;
};

layout(set = 0, binding = 16, std430) writeonly buffer MacrotileMeta {
    MacrotileMetadata macrotile_metas[];
};

layout(push_constant) uniform push_constant {
    uint screen_w, screen_h;
    uint n_tiles_x, n_tiles_y;
    uint n_macrotiles_x, n_macrotiles_y;
    uint n_seg_blocks;
    uint n_bins;

    uint tile_size, macrotile_size;

    uint n_segments;
    uint n_paths;

    uint fill_rule;
} pc;

shared uint contribution_bitmap[N_SLICE][MAX_TILES];
shared uint bin_count[N_SUBSLICE][MAX_TILES];
shared uint macrotile_offset[MAX_TILES];

void main() {
    uint lidx = gl_LocalInvocationID.x;
    uint path_id = gl_GlobalInvocationID.x;

    uint n_macrotiles = pc.n_macrotiles_x * pc.n_macrotiles_y;

    // clear the bitmap 
    if (lidx < MAX_TILES) {
        for (uint i = 0u; i < N_SLICE; i++) {
            contribution_bitmap[i][lidx] = 0u;
        }

        for (uint i = 0u; i < N_SUBSLICE; i++) {
            bin_count[i][lidx] = 0u;
        }

        macrotile_offset[lidx] = 0u;
    }

    barrier();

    ivec2 t0 = ivec2(0);
    ivec2 t1 = ivec2(-1);

    uint my_slice = lidx / 32u;
    uint my_mask = 1u << (lidx & 31u);

    if (path_id < pc.n_paths) {
        DrawPath path = paths[path_id];

        vec2 mn = path.mn;
        vec2 mx = path.mx;

        debugPrintfEXT(
            "element %u aabb: [ %f,%f ] to [ %f,%f ] from %u to %u\n",
            path.id,
            mn.x, mn.y,
            mx.x, mx.y,
            path.segment_offset,
            path.segment_offset + path.segment_count
        );

        ivec2 ip0 = ivec2(floor(mn));
        ivec2 ip1 = ivec2(floor(mx));

        // Divide by 256, as macrotiles will always be 256x256 pixels (safes div op) 
        t0 = ip0 >> 8;
        t1 = ip1 >> 8;

        // clamp to valid ranges
        t0 = clamp(
            t0,
            ivec2(0),
            ivec2(
                int(pc.n_macrotiles_x - 1u),
                int(pc.n_macrotiles_y - 1u)
            )
        );

        t1 = clamp(
            t1,
            ivec2(0),
            ivec2(
                int(pc.n_macrotiles_x - 1u),
                int(pc.n_macrotiles_y - 1u)
            )
        );

        debugPrintfEXT(
            "element covering area from: [ %i,%i ] to [ %i,%i ]\n",
            t0.x, t0.y,
            t1.x, t1.y
        );

        // add this element's contribution to all macrotiles it overlaps 
        for (int y = t0.y; y <= t1.y; y++) {
            for (int x = t0.x; x <= t1.x; x++) {
                uint bin = uint(y) * pc.n_macrotiles_x + uint(x);

                if (bin < MAX_TILES) {
                    atomicOr(contribution_bitmap[my_slice][bin], my_mask);
                }
            }
        }
    }

    barrier();

    // from here on out: lidx = the flat tile index
    if (lidx < n_macrotiles && lidx < MAX_TILES) {
        uint element_count = 0u;

        // count up the element contribution from this macrotile.
        for (uint i = 0u; i < N_SUBSLICE; i++) {
            element_count += bitCount(contribution_bitmap[i * 2u][lidx]);
            uint count_lo = element_count;

            element_count += bitCount(contribution_bitmap[i * 2u + 1u][lidx]);
            uint count_hi = element_count;

            uint count_packed = count_lo | (count_hi << 16u);
            bin_count[i][lidx] = count_packed;
        }

        uint off = atomicAdd(bump, element_count);
        // TODO: care for bump overflow

        macrotile_offset[lidx] = off;

        if (element_count != 0u) {
            debugPrintfEXT(
                "bin %u has %u elements.\n",
                lidx,
                element_count
            );
        }

        macrotile_metas[lidx].count = element_count;
        macrotile_metas[lidx].off = off;
    }

    barrier();

    if (path_id < pc.n_paths) {
        DrawPath path = paths[path_id];

        for (int y = t0.y; y <= t1.y; y++) {
            for (int x = t0.x; x <= t1.x; x++) {
                uint bin = uint(y) * pc.n_macrotiles_x + uint(x);

                if (bin >= MAX_TILES) {
                    continue;
                }

                uint current_bitmap = contribution_bitmap[my_slice][bin];

                // rank or offset within our local slice
                uint rank = bitCount(current_bitmap & (my_mask - 1u));

                // add the workgroup-local offset (inbetween slices)
                if (my_slice > 0u) {
                    // adding the (prefix-summed) count of all previous slices
                    uint previous = my_slice - 1u;
                    uint prefix_count_packed = bin_count[previous / 2u][bin];

                    rank += (
                        prefix_count_packed >>
                        (16u * (previous & 1u))
                    ) & 0xffffu;
                }

                // add the path to the bin at the calculated offset
                binned_paths[macrotile_offset[bin] + rank] = path.id;
            }
        }
    }
}
