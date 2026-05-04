#version 450

layout(local_size_x = 256) in;

struct Segment {
    vec2 p0;
    vec2 p1;
};

struct SegmentMacroRange {
    uint min_xy;
    uint max_xy;
};

layout(set = 0, binding = 0, std430) readonly buffer Segments {
    Segment segments[];
};

layout(set = 0, binding = 13, std430) buffer SegmentMacroRanges {
    SegmentMacroRange segment_macro_ranges[];
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

uint pack_xy(uint x, uint y)
{
    return (y << 16u) | x;
}

void main()
{
    // For every segment, get the macrotile coverage area (AABB) of the segment

    uint seg_id = gl_GlobalInvocationID.x;

    if (seg_id >= pc.n_segments)
        return;

    Segment s = segments[seg_id];

    vec2 mn = min(s.p0, s.p1);
    vec2 mx = max(s.p0, s.p1);


    ivec2 ip0 = ivec2(floor(mn));
    ivec2 ip1 = ivec2(floor(mx));

    // Divide by 256, as macrotiles will always be 256x256 pixels (safes div op) 
    ivec2 t0 = ip0 >> 8;
    ivec2 t1 = ip1 >> 8;

    // clamp to valid ranges
    t0 = clamp(t0, ivec2(0),
               ivec2(int(pc.n_macrotiles_x - 1u),
                     int(pc.n_macrotiles_y - 1u)));

    t1 = clamp(t1, ivec2(0),
               ivec2(int(pc.n_macrotiles_x - 1u),
                     int(pc.n_macrotiles_y - 1u)));

    segment_macro_ranges[seg_id].min_xy =
        pack_xy(uint(t0.x), uint(t0.y));

    segment_macro_ranges[seg_id].max_xy =
        pack_xy(uint(t1.x), uint(t1.y));
}
