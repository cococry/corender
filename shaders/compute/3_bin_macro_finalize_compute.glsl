#version 450

layout(local_size_x = 64) in;

layout(set = 0, binding = 1, std430) buffer MacrotileNSegments {
    uint macrotile_n_segments[];
};

layout(set = 0, binding = 2, std430) buffer MacrotileOffsets {
    uint macrotile_offsets[];
};

layout(set = 0, binding = 11, std430) readonly buffer MacroBlockCounts {
    uint macro_block_counts[];
};

layout(set = 0, binding = 12, std430) readonly buffer MacroBlockOffsets {
    uint macro_block_offsets[];
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

void main()
{
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= pc.n_bins)
        return;

    uint total = 0u;

    if (pc.n_seg_blocks > 0u) {
        uint last = pc.n_seg_blocks - 1u;
        uint idx  = tid * pc.n_seg_blocks + last;
        total = macro_block_offsets[idx] + macro_block_counts[idx];
    }

    macrotile_n_segments[tid] = total;
    macrotile_offsets[tid]    = total; 
}
