#define SEGMENTS_BINDING            0 
#define PATHS_BINDING               1 
#define IMG_BINDING                 2 
#define INDIRECT_BINDING            3 
#define MSAA8_LUT_BINDING           4 
#define MSAA16_LUT_BINDING          5
#define COMP_PIPELINE_BINDING_BASE  6 // first valid location after segments,paths,img and indirect
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

#ifndef CR_ENABLE_GPU_STATS
#define CR_ENABLE_GPU_STATS 0
#endif

#define STATS_BINDING (COMP_PIPELINE_BINDING_BASE + 9)
#define STATS_HIST_BINS 32u

#define TILE_DENSE_THRESHOLD 32
#define EVEN_ODD_WINDING 1

struct GpuStats {
  uint failed;

  uint n_tiles_seen;

  uint empty_tiles;
  uint solid_tiles;
  uint fine_tiles;
  uint active_tiles;

  uint total_tile_segments;
  uint total_fine_segments;

  uint max_tile_segments;
  uint max_fine_segments;

  uint max_abs_winding;

  uint contention_tiles;
  uint contended_tile_segments;
  uint excess_tile_segments;
  uint tile_atomic_pair_pressure;

  uint valid_edges;
  uint invalid_edges;
  uint y_edge_edges;
  uint horizontal_edges;
  uint zero_length_edges;
  uint tile_mismatch_edges;

  uint estimated_fine_edge_evals;

  uint hist_tile_segments[32];
  uint hist_fine_segments[32];
};

#if CR_ENABLE_GPU_STATS

#define CR_STAT_ADD(field, value) atomicAdd(stats.field, value)
#define CR_STAT_MAX(field, value) atomicMax(stats.field, value)
#define CR_STAT_HIST_ADD(field, index, value) atomicAdd(stats.field[(index)], value)

#else

#define CR_STAT_ADD(field, value)
#define CR_STAT_MAX(field, value)
#define CR_STAT_HIST_ADD(field, index, value)

#endif

struct TileInfo {
  uint base;
  uint count;
  int winding;
  uint flags;
};

struct TileEdge {
  uint p0; 
  uint p1; 
};

#define TOUCH_SEG_BITS   20u
#define TOUCH_RANK_BITS  12u
#define TOUCH_TILE_BITS  16u
#define TOUCH_LOCAL_TILE_BITS 16u

#define TOUCH_SEG_MASK   ((1u << TOUCH_SEG_BITS) - 1u)
#define TOUCH_RANK_MASK  ((1u << TOUCH_RANK_BITS) - 1u)
#define TOUCH_TILE_MASK  ((1u << TOUCH_TILE_BITS) - 1u)
#define TOUCH_LOCAL_TILE_MASK ((1u << TOUCH_LOCAL_TILE_BITS) - 1u)

struct TileTouchRecord {
  uint a;
  uint b;
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
  uint n_active_tiles_sparse;
  uint n_active_tiles_dense;
  uint n_tile_segment_slots;
  uint failed;
} bump;

