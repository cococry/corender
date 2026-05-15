#version 450
#extension GL_GOOGLE_include_directive : require

#include "../../shared/pc.glsl"
#include "../../shared/segment_walking.glsl"

layout(local_size_x = 256) in;

const float ONE_MINUS_ULP = 0.99999994;
const float ROBUST_EPSILON = 2e-7;
const float EPSILON = 1e-6;
const float CLIP_NUDGE = 1e-3;

layout(set = 0, binding = SEGMENTS_BINDING, std430) readonly buffer Segments {
  Segment segments[];
};

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 1, std430) readonly buffer TileTouchRecords {
  TileTouchRecord tile_touch_records[];
};

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 5, std430) readonly buffer TileInfos {
  TileInfo tile_infos[];
};

layout(set = 0, binding = COMP_PIPELINE_BINDING_BASE + 8, std430) writeonly buffer TileEdges {
  TileEdge tile_edges[];
};

#if CR_ENABLE_GPU_STATS
layout(set = 0, binding = STATS_BINDING, std430) buffer StatsBuffer {
  GpuStats stats;
};
#endif

uint unpack_tile_id(uint packed_tile_rank) {
  return packed_tile_rank & 0xffffu;
}

uint unpack_rank(uint packed_tile_rank) {
  return packed_tile_rank >> 16u;
}


uint pack_coord(float v) {
  float ts = float(pc.tile_size);
  return uint(round(clamp(v, 0.0, ts) * (65535.0 / ts)));
}

uint pack_point(vec2 p) {
  uint x = pack_coord(p.x);
  uint y = pack_coord(p.y);
  return x | (y << 16u);
}

TileEdge make_invalid_edge() {
  TileEdge e;
  e.p0 = 0u;
  e.p1 = 0u;
  return e;
}

void write_invalid_edge(uint dst) {
  CR_STAT_ADD(invalid_edges, 1u);
  tile_edges[dst] = make_invalid_edge();
}

bool has_left_edge_signal(vec2 p0, vec2 p1) {
  float ts = float(pc.tile_size);

  bool p0_left =
    abs(p0.x) <= EPSILON &&
    p0.y > EPSILON &&
    p0.y < ts - EPSILON;

  bool p1_left =
    abs(p1.x) <= EPSILON &&
    p1.y > EPSILON &&
    p1.y < ts - EPSILON;

  return p0_left || p1_left;
}

bool is_horizontal_edge(vec2 p0, vec2 p1) {
  return abs(p1.y - p0.y) <= EPSILON;
}

bool make_tile_edge(
    Segment seg,
    uint local_tile,
    out ivec2 tile,
    out vec2 local_p0,
    out vec2 local_p1
    ) {
  float ts = float(pc.tile_size);

  bool is_down = seg.p1.y >= seg.p0.y;

  vec2 p0 = is_down ? seg.p0 : seg.p1;
  vec2 p1 = is_down ? seg.p1 : seg.p0;

  LineWalkParams lp;

  if (!make_line_walk_params(seg.p0, seg.p1, lp)) {
    return false;
  }

  if (local_tile >= lp.count) {
    return false;
  }

  float x_step;
  tile = tile_for_local_tile(lp, local_tile, x_step);

  vec2 tile_xy = vec2(tile) * ts;
  vec2 tile_xy1 = tile_xy + vec2(ts);

  if (local_tile > 0u) {
    float previous_x_step = floor(
        lp.x_step_rate * (float(local_tile) - 1.0) +
        lp.x_step_start_offset
        );

    if (previous_x_step == x_step) {
      /*
       * X did not change from the previous tile to this tile.
       * Therefore Y changed, so this tile was entered through the top edge.
       *
       * Segments are already normalized to move downward, so it cannot enter
       * through the bottom edge.
       */
      float denom = p1.y - p0.y;

      if (denom == 0.0) {
        return false;
      }

      // X coordinate where the segment entered this tile through the top edge.
      float t = (tile_xy.y - p0.y) / denom;
      float x_point = p0.x + (p1.x - p0.x) * t;

      // Clamp segment X point to tile bounds.
      // This should theoretically be optional.
      x_point = clamp(x_point, tile_xy.x + CLIP_NUDGE, tile_xy1.x);

      // Top Y is the clip boundary for the start point.
      p0 = vec2(x_point, tile_xy.y);
    } else {
      /*
       * X changed from the previous tile to this tile.
       * Therefore this tile was entered through a vertical edge.
       */
      float x_clip = lp.x_sign > 0.0
        ? tile_xy.x
        : tile_xy1.x;

      float denom = p1.x - p0.x;

      if (denom == 0.0) {
        return false;
      }

      // Y coordinate where the segment entered this tile through the vertical edge.
      float y_point =
        p0.y +
        (p1.y - p0.y) * (x_clip - p0.x) / denom;

      // Clamp segment Y point to tile bounds.
      // This should theoretically be optional.
      y_point = clamp(y_point, tile_xy.y + CLIP_NUDGE, tile_xy1.y);

      p0 = vec2(x_clip, y_point);
    }
  }

  /*
   * Clip end point to the boundary where this tile is exited.
   */
  if (local_tile < lp.count - 1u) {
    float next_x_step = floor(
        lp.x_step_rate * (float(local_tile) + 1.0) +
        lp.x_step_start_offset
        );

    if (x_step == next_x_step) {
      /*
       * X will not change from this tile to the next tile.
       * Therefore Y changes, so this tile exits through the bottom edge.
       */
      float denom = p1.y - p0.y;

      if (denom == 0.0) {
        return false;
      }

      float xt =
        p0.x +
        (p1.x - p0.x) * (tile_xy1.y - p0.y) / denom;

      xt = clamp(xt, tile_xy.x + CLIP_NUDGE, tile_xy1.x);

      p1 = vec2(xt, tile_xy1.y);
    } else {
      /*
       * X will change from this tile to the next tile.
       * Therefore this tile exits through a vertical edge.
       */
      float x_clip = lp.x_sign > 0.0
        ? tile_xy1.x
        : tile_xy.x;

      float denom = p1.x - p0.x;

      if (denom == 0.0) {
        return false;
      }

      float yt =
        p0.y +
        (p1.y - p0.y) * (x_clip - p0.x) / denom;

      yt = clamp(yt, tile_xy.y + CLIP_NUDGE, tile_xy1.y);

      p1 = vec2(x_clip, yt);
    }
  }

  local_p0 = p0 - tile_xy;
  local_p1 = p1 - tile_xy;

  /*
  if (local_p0.x == 0.0) {
    if (local_p1.x == 0.0) {
      local_p0.x = EPSILON;

      if (local_p0.y == 0.0) {
        local_p1.x = EPSILON;
        local_p1.y = ts;
      } else {
        local_p1.x = 2.0 * EPSILON;
        local_p1.y = local_p0.y;
      }
    } else if (local_p0.y == 0.0) {
      local_p0.x = EPSILON;
    } else {
    }
  } else if (local_p1.x == 0.0) {
    if (local_p1.y == 0.0) {
      local_p1.x = EPSILON;
    }
  }
  */

  /*
     Restore original edge direction.
   */
  if (!is_down) {
    vec2 tmp = local_p0;
    local_p0 = local_p1;
    local_p1 = tmp;
  }

  return true;
}

uint touch_seg_id(TileTouchRecord r) {
  return r.a & TOUCH_SEG_MASK;
}

uint touch_rank(TileTouchRecord r) {
  return (r.a >> TOUCH_SEG_BITS) & TOUCH_RANK_MASK;
}

uint touch_tile_id(TileTouchRecord r) {
  return r.b & TOUCH_TILE_MASK;
}

uint touch_local_tile(TileTouchRecord r) {
  return (r.b >> TOUCH_TILE_BITS) & TOUCH_LOCAL_TILE_MASK;
}

// dispatch over all segment-tile touch records 
// generated by 1_walk_segments 
void main() {
  uint touch_idx = gl_GlobalInvocationID.x;

  if (touch_idx >= bump.n_touches) {
    return;
  }

  TileTouchRecord touch = tile_touch_records[touch_idx];

  uint seg_id         = touch_seg_id(touch);
  uint rank           = touch_rank(touch);
  uint tile_id        = touch_tile_id(touch);
  uint local_tile     = touch_local_tile(touch);

  // enable GPU stats is used more as a 
  // debug/sanitizing flag here
#ifdef CR_ENABLE_GPU_STATS
  if (tile_id >= pc.n_tiles) {
    CR_STAT_ADD(invalid_edges, 1u);
    atomicOr(bump.failed, FAIL_SCATTER_OOB);
    return;
  }
#endif

  TileInfo info = tile_infos[tile_id];

  if (rank >= info.count) {
    CR_STAT_ADD(invalid_edges, 1u);
    atomicOr(bump.failed, FAIL_RANK_OOB);
    return;
  }

  uint dst = info.base + rank;

  if (dst >= pc.max_tile_storage) {
    CR_STAT_ADD(invalid_edges, 1u);
    atomicOr(bump.failed, FAIL_SCATTER_OOB);
    return;
  }

  Segment seg = segments[seg_id];

  ivec2 reconstructed_tile;
  vec2 local_p0;
  vec2 local_p1;

  if(!make_tile_edge(
      seg,
      local_tile,
      reconstructed_tile,
      local_p0,
      local_p1
      )) {
    
    write_invalid_edge(dst);
    return;
  }

  vec2 d = local_p1 - local_p0;

  CR_STAT_ADD(valid_edges, 1u);

#ifdef CR_ENABLE_GPU_STATS
  if (has_left_edge_signal(local_p0, local_p1)) {
    CR_STAT_ADD(y_edge_edges, 1u);
  }

  if (is_horizontal_edge(local_p0, local_p1)) {
    CR_STAT_ADD(horizontal_edges, 1u);
  }
#endif 

  TileEdge edge;
  edge.p0 = pack_point(local_p0);
  edge.p1 = pack_point(local_p1);

  tile_edges[dst] = edge;
}
