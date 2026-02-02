#version 450

#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable


#define SG_SIZE 32 

#define SGS_PER_WG 64 / SG_SIZE

shared uint subgroup_totals[SGS_PER_WG]; 
layout(local_size_x = 64) in;

struct Segment {
    vec2 p0;
    vec2 p1;
};

layout(set = 0, binding = 0, std430) readonly buffer Segments {
    Segment segments[];
};


layout(set = 0, binding = 1, std430) buffer macrotileNSegments {
    uint macrotile_n_segments[];
};

layout(set = 0, binding = 2, std430) buffer macrotileOffsets {
    uint macrotile_offsets[];
};

layout(set = 0, binding = 3, std430) buffer macrotileSegments {
    uint macrotile_segments[];
};


layout(set = 0, binding = 5, std430) buffer TileSegments {
    uint tile_segments[];
};


layout(set = 0, binding = 7, std430) buffer TileOffsets {
    uint tile_offsets_micro[];
};

layout(set = 0, binding = 12, std430) readonly buffer macrotileOffsetsMicro{
    uint macrotile_offsets_micro[];
};

layout(set = 0, binding = 13, std430) buffer TileCountsMacro {
    uint tile_counts_micro[];
};



layout(push_constant) uniform push_constant {
    uint screen_w, screen_h;
    uint n_tiles_x, n_tiles_y;
    uint n_macrotiles_x, n_macrotiles_y;
    uint tile_size, macrotile_size;

    uint n_segments;
    uint n_paths;
    uint fill_rule;
} pc;

shared uint tile_cursor_local[64];
shared uint tile_local_offset[64];
shared uint tile_counts[64];


void main() {
  uint lid = gl_LocalInvocationID.x;

  ivec2 macro_base = ivec2(int(gl_WorkGroupID.x) * 8, int(gl_WorkGroupID.y) * 8);
  ivec2 macro_end = macro_base + ivec2(7); 

  if(lid < 64) {
    tile_cursor_local[lid] = 0;
    uint macro_id =
      gl_WorkGroupID.y * pc.n_macrotiles_x +
      gl_WorkGroupID.x;
    tile_local_offset[lid] = tile_counts_micro[macro_id*64+lid];
    tile_counts[lid] = tile_counts_micro[macro_id*64+lid];
  }
  barrier();

  uint prefix = subgroupExclusiveAdd(tile_local_offset[lid]);

  uint sg_sum = subgroupAdd(tile_local_offset[lid]);

  if (subgroupElect())
    subgroup_totals[gl_SubgroupID] = sg_sum;

  barrier();

  uint sg_offset = 0;

  if (gl_SubgroupID == 0)
  {
    uint lane = gl_SubgroupInvocationID;
    uint val = (lane < SGS_PER_WG) ? subgroup_totals[lane] : 0;

    uint off = subgroupExclusiveAdd(val);

    if (lane < SGS_PER_WG)
      subgroup_totals[lane] = off;
  }

  barrier();

  sg_offset = subgroup_totals[gl_SubgroupID];

  prefix += sg_offset;

  tile_local_offset[lid] = prefix;
  barrier();


  uint macro_id = gl_WorkGroupID.y * pc.n_macrotiles_x + gl_WorkGroupID.x;

if(lid < 64)
{
    uint gx = uint(macro_base.x) + (lid % 8);
    uint gy = uint(macro_base.y) + (lid / 8);

    if(gx < pc.n_tiles_x && gy < pc.n_tiles_y)
    {
        uint tile_id = gy * pc.n_tiles_x + gx;

        tile_offsets_micro[tile_id] =
            macrotile_offsets_micro[macro_id] +
            tile_local_offset[lid];
    }
}
barrier();

  uint count = macrotile_n_segments[macro_id];
  uint base = macrotile_offsets[macro_id];

  for(uint i = gl_LocalInvocationID.x; i < count; i += gl_WorkGroupSize.x) {
    uint seg_id = macrotile_segments[base + i];
    Segment seg = segments[seg_id];

    vec2 p0 = seg.p0;
    vec2 p1 = seg.p1;

    vec2 seg_max = max(p0, p1);
    vec2 seg_min = min(p0, p1);
    float tile_size = float(pc.tile_size);

    ivec2 tile_start = ivec2(floor(seg_min / tile_size));
    ivec2 tile_end = ivec2(floor(seg_max/ tile_size));

    tile_start = clamp(tile_start, ivec2(0), ivec2(pc.n_tiles_x - 1, pc.n_tiles_y - 1));
    tile_end = clamp(tile_end, ivec2(0), ivec2(pc.n_tiles_x - 1, pc.n_tiles_y - 1));

    tile_start = clamp(tile_start, macro_base, macro_end); 
    tile_end = clamp(tile_end, macro_base, macro_end); 

    for(uint ty = tile_start.y; ty <= tile_end.y; ty++) {
      for(uint tx = tile_start.x; tx <= tile_end.x; tx++) {

        uint local_tile = (ty - uint(macro_base.y)) * 8 + (tx - uint(macro_base.x));

        uint tile_id = ty * pc.n_tiles_x + tx;
        uint idx = atomicAdd(tile_cursor_local[local_tile], 1);

        if(idx >= tile_counts[local_tile]) continue;
        uint micro_base = macrotile_offsets_micro[macro_id] + tile_local_offset[local_tile];

        tile_segments[micro_base + idx] = seg_id; 
      }
    }
  }

}
