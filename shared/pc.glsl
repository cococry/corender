#define COMP_PIPELINE_BINDING_BASE  4 // first valid location after segments,paths,img and indirect
#define SEGMENTS_BINDING            0 
#define PATHS_BINDING               1 
#define IMG_BINDING                 2 
#define INDIRECT_BINDING            3 
#define SUBGROUP_TMP_BINDING        COMP_PIPELINE_BINDING_BASE + 3 

#define FAIL_TOUCH_OVERFLOW         1u << 0
#define FAIL_TILE_SEGMENT_OVERFLOW  1u << 1
#define FAIL_ACTIVE_TILE_OVERFLOW   1u << 2
#define FAIL_RANK_OVERFLOW          1u << 3
#define FAIL_TILE_ID_OVERFLOW       1u << 4
#define FAIL_SCATTER_OOB            1u << 5
#define FAIL_BAD_TOUCH_TILE         1u << 6
#define FAIL_TILE_EVENT_OOB         1u << 7
#define FAIL_RANK_OOB               1u << 8
#define FAIL_FINE_TILE_SIZE_MISMATCH 1u << 9 
#define FAIL_FINE_OOB 1u << 10 
#define AVG_TOUCHES_PER_TILE        100


struct TileInfo {
    uint base;
    uint count;
    int winding;
    uint flags;
};

struct TileEdge {
    uint p0;   // x: low 16, y: high 16, tile-local 8.8 fixed
    uint p1;   // x: low 16, y: high 16, tile-local 8.8 fixed
    uint meta; // bit 0 valid, bit 1 negative winding delta
    uint y_edge;  
};


struct Segment {
    vec2 p0;
    vec2 p1;
};

#define TILE_EMPTY 0u
#define TILE_SOLID (1u << 0)
#define TILE_FINE  (1u << 1)
layout(push_constant) uniform push_constant {
    uint screen_w, screen_h;
    uint n_tiles_x, n_tiles_y, n_tiles;

    uint tile_size;

    uint max_tile_storage;

    uint n_segments;
    uint n_paths;
} pc;

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 0, std430) buffer Bump {
    uint n_touches;
    uint n_active_tiles;
    uint n_tile_segment_slots;
    uint failed;
} bump;

