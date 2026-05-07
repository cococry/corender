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

#define AVG_TOUCHES_PER_TILE        100

layout(push_constant) uniform push_constant {
    uint screen_w, screen_h;
    uint n_tiles_x, n_tiles_y;
    uint n_macrotiles_x, n_macrotiles_y;
    uint n_seg_blocks;
    uint n_path_partitions;
    uint n_bins;

    uint tile_size, macrotile_size;

    uint n_segments;
    uint n_paths;

    uint fill_rule;
} pc;


layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 0, std430) buffer Bump {
    uint n_touches;
    uint n_active_tiles;
    uint n_tile_segment_slots;
    uint failed;
} bump;

